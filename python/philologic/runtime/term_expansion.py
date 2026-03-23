#!/var/lib/philologic5/philologic_env/bin/python3
"""LMDB-based term expansion and autocomplete for PhiloLogic queries.

Handles all query-term expansion: normalized word lookups, regex pattern
scanning, LEMMA/ATTR expansion, NOT-term exclusion, and autocomplete.
"""

import os
import threading

import lmdb
import regex as re
from unidecode import unidecode


# Process-level cache: one LMDB env per lmdb_path, kept open for the
# lifetime of the worker process (avoids repeated open/close overhead).
_norm_lmdb_cache: dict[str, lmdb.Environment] = {}
_lmdb_cache_lock = threading.Lock()
# db_paths for which word_forms.lmdb is absent (no lemma/attr flat files)
_no_forms_lmdb: set[str] = set()

# Flat files (in frequencies/) that feed word_forms.lmdb
_FORMS_FLAT_FILES = ("lemmas", "word_attributes", "lemma_word_attributes")


def get_lmdb_env(lmdb_path: str) -> lmdb.Environment:
    """Return (and cache) a read-only LMDB environment for the given path."""
    env = _norm_lmdb_cache.get(lmdb_path)
    if env is not None:
        return env
    with _lmdb_cache_lock:
        # Double-check after acquiring lock (another thread may have created it)
        env = _norm_lmdb_cache.get(lmdb_path)
        if env is not None:
            return env
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, max_spare_txns=4)
        _norm_lmdb_cache[lmdb_path] = env
        return env


def _get_norm_env(freq_file: str) -> lmdb.Environment:
    """Return (and cache) the norm_word.lmdb env (built at index time by PostFilters)."""
    return get_lmdb_env(freq_file + ".lmdb")


def _get_forms_env(db_path: str) -> lmdb.Environment | None:
    """Return (and cache) the word_forms.lmdb env (built at index time by PostFilters).

    Returns None if the database has no word_forms.lmdb (no lemma/attr data).
    """
    lmdb_path = os.path.join(db_path, "frequencies", "word_forms.lmdb")
    if lmdb_path in _norm_lmdb_cache:
        return _norm_lmdb_cache[lmdb_path]
    if db_path in _no_forms_lmdb:
        return None
    if not os.path.exists(lmdb_path):
        _no_forms_lmdb.add(db_path)
        return None
    return get_lmdb_env(lmdb_path)


def _norm_key(token: str, lowercase: bool = True) -> bytes:
    if lowercase:
        token = token.lower()
    return "".join(unidecode(token)).encode("utf-8")


def _lmdb_lookup(txn, key: bytes) -> list[str]:
    """Return list of original forms for a normalized key, or []."""
    val = txn.get(key)
    if val is None:
        return []
    return bytes(val).decode("utf-8").split("\x00")


# ── Regex-pattern detection and LMDB cursor expansion ─────────────────────────

_REGEX_METACHARS = frozenset(".*+?[{(\\")


def _is_regex_pattern(token: str) -> bool:
    """Return True if token contains unescaped regex metacharacters."""
    i = 0
    while i < len(token):
        if token[i] == "\\" and i + 1 < len(token):
            i += 2  # skip escaped char
            continue
        if token[i] in _REGEX_METACHARS:
            return True
        i += 1
    return False


def _split_literal_prefix(token: str) -> tuple[str, str]:
    """Split token into (literal_prefix, meta_suffix) at the first unescaped metachar."""
    i = 0
    while i < len(token):
        if token[i] == "\\" and i + 1 < len(token):
            i += 2
            continue
        if token[i] in _REGEX_METACHARS:
            return token[:i], token[i:]
        i += 1
    return token, ""


def _normalize_pattern(token: str, lowercase: bool = True) -> tuple[bytes, str]:
    """Normalize a regex token for LMDB cursor scan + compiled-regex filter.

    Returns (cursor_prefix_bytes, full_pattern_str) where:
    - cursor_prefix_bytes: normalized literal prefix (for set_range + startswith)
    - full_pattern_str: complete regex pattern (normalized literal + raw meta suffix)
    """
    literal, meta = _split_literal_prefix(token)
    if lowercase:
        literal = literal.lower()
    norm_literal = "".join(unidecode(literal))
    return norm_literal.encode("utf-8"), norm_literal + meta


def _lmdb_expand_term(txn, norm_prefix: bytes, pattern_str: str | None = None,
                      max_results: int = 0) -> list[str]:
    """Cursor-scan norm_word.lmdb from norm_prefix, return original word forms.

    If pattern_str is given, applies re.match filter on normalized keys.
    When norm_prefix is empty, scans the whole DB filtered by pattern_str;
    max_results defaults to 10000 in that case to cap unbounded full-DB scans.
    max_results: stop after collecting that many forms (0 = unlimited).
    """
    if not norm_prefix and not pattern_str:
        return []
    if not norm_prefix and max_results == 0:
        max_results = 10000
    compiled = re.compile(pattern_str) if pattern_str else None
    results: list[str] = []
    cursor = txn.cursor()
    try:
        if norm_prefix:
            if not cursor.set_range(norm_prefix):
                return results
        else:
            if not cursor.first():
                return results
        while True:
            k = bytes(cursor.key())
            if norm_prefix and not k.startswith(norm_prefix):
                break
            if compiled is None or compiled.match(k.decode("utf-8", errors="replace")):
                for form in bytes(cursor.value()).decode("utf-8").split("\x00"):
                    results.append(form)
                    if max_results and len(results) >= max_results:
                        return results
            if not cursor.next():
                break
    finally:
        cursor.close()
    return results


def _lmdb_scan_keys(txn, prefix: bytes, pattern_str: str | None = None,
                    max_results: int = 0) -> list[str]:
    """Cursor-scan LMDB from prefix, return matching key strings.

    Used for LEMMA/ATTR/LEMMA_ATTR expansion against words.lmdb.
    Values (binary hit data) are ignored; only key strings are returned.
    If pattern_str is given, applies re.match filter on key strings.
    When prefix is empty, scans whole DB bounded by max_results.
    max_results: stop after collecting that many keys (0 = unlimited).
    """
    if not prefix and not pattern_str:
        return []
    compiled = re.compile(pattern_str) if pattern_str else None
    results: list[str] = []
    cursor = txn.cursor()
    try:
        if prefix:
            if not cursor.set_range(prefix):
                return results
        else:
            cursor.first()
        while True:
            k = bytes(cursor.key())
            if prefix and not k.startswith(prefix):
                break
            key_str = k.decode("utf-8", errors="replace")
            if compiled is None or compiled.match(key_str):
                results.append(key_str)
                if max_results and len(results) >= max_results:
                    break
            if not cursor.next():
                break
    finally:
        cursor.close()
    return results


def _expand_positive(kind: str, token: str, txn, ascii_conversion: bool, lowercase: bool,
                     forms_env: lmdb.Environment | None = None) -> list[str]:
    """Expand one positive token to the list of words.lmdb lookup keys.

    For TERM/QUOTE with ascii_conversion, expands via norm_word.lmdb (txn).
    Supports regex patterns (e.g. sens.*) via LMDB cursor scan.
    For LEMMA/ATTR/LEMMA_ATTR regex, scans word_forms.lmdb (forms_env).
    """
    if kind in ("TERM", "RANGE"):
        if ascii_conversion:
            if _is_regex_pattern(token):
                norm_prefix, pattern_str = _normalize_pattern(token, lowercase)
                return _lmdb_expand_term(txn, norm_prefix, pattern_str)
            return _lmdb_lookup(txn, _norm_key(token, lowercase))
        else:
            return [token]
    elif kind == "QUOTE":
        inner = token[1:-1]  # strip surrounding quotes
        if _is_regex_pattern(inner):
            norm_prefix, pattern_str = _normalize_pattern(inner, lowercase)
            return _lmdb_expand_term(txn, norm_prefix, pattern_str)
        return [inner]
    elif kind in ("LEMMA", "LEMMA_ATTR", "ATTR"):
        if _is_regex_pattern(token) and forms_env is not None:
            literal, meta = _split_literal_prefix(token)
            prefix_bytes = literal.encode("utf-8")
            with forms_env.begin(buffers=True) as f_txn:
                return _lmdb_scan_keys(f_txn, prefix_bytes, literal + meta)
        return [token]
    return []


def _expand_exclude(kind: str, token: str, txn, ascii_conversion: bool, lowercase: bool,
                    forms_env: lmdb.Environment | None = None) -> set[str]:
    """Expand one NOT token to the set of forms to exclude.

    Mirrors _expand_positive but returns a set for O(1) exclusion checks.
    """
    if kind in ("TERM", "RANGE"):
        if ascii_conversion:
            if _is_regex_pattern(token):
                norm_prefix, pattern_str = _normalize_pattern(token, lowercase)
                return set(_lmdb_expand_term(txn, norm_prefix, pattern_str))
            return set(_lmdb_lookup(txn, _norm_key(token, lowercase)))
        else:
            return {token}
    elif kind == "QUOTE":
        inner = token[1:-1]
        if _is_regex_pattern(inner):
            norm_prefix, pattern_str = _normalize_pattern(inner, lowercase)
            return set(_lmdb_expand_term(txn, norm_prefix, pattern_str))
        return {inner}
    elif kind in ("LEMMA", "LEMMA_ATTR", "ATTR"):
        if _is_regex_pattern(token) and forms_env is not None:
            literal, meta = _split_literal_prefix(token)
            prefix_bytes = literal.encode("utf-8")
            with forms_env.begin(buffers=True) as f_txn:
                return set(_lmdb_scan_keys(f_txn, prefix_bytes, literal + meta))
        return {token}
    return set()


def expand_query_not(split, freq_file, dest_fh, ascii_conversion, lowercase=True):
    """Expand search terms using LMDB index (replaces subprocess/rg pipeline).

    For each query group, expands positive tokens to all matching original word
    forms (including regex patterns like sens.*), subtracts any NOT-excluded
    forms, and writes the result to dest_fh.
    Groups are separated by blank lines (consumed by get_word_groups()).
    """
    env = _get_norm_env(freq_file)
    db_path = os.path.normpath(os.path.join(os.path.dirname(freq_file), ".."))
    forms_env = _get_forms_env(db_path)
    first = True

    with env.begin(buffers=True) as txn:
        for group in split:
            if not first:
                try:
                    dest_fh.write("\n")
                except TypeError:
                    dest_fh.write(b"\n")
                dest_fh.flush()
            first = False

            # Separate positive tokens from NOT-excluded tokens
            exclude_specs: list[tuple[str, str]] = []
            pos_group = list(group)
            for i, (kind, _) in enumerate(group):
                if kind == "NOT":
                    exclude_specs = list(group[i + 1:])
                    pos_group = list(group[:i])
                    break

            # Union of all positive-term expansions (order-preserving, deduped)
            seen: set[str] = set()
            pos_forms: list[str] = []
            for kind, token in pos_group:
                for form in _expand_positive(kind, token, txn, ascii_conversion, lowercase, forms_env):
                    if form not in seen:
                        seen.add(form)
                        pos_forms.append(form)

            # Set of forms to exclude
            excl: set[str] = set()
            for kind, token in exclude_specs:
                excl |= _expand_exclude(kind, token, txn, ascii_conversion, lowercase, forms_env)

            # Write filtered forms, one per line
            for form in pos_forms:
                if form not in excl:
                    try:
                        dest_fh.write(form + "\n")
                    except TypeError:
                        dest_fh.write((form + "\n").encode("utf-8"))


# ── Metadata inverted word index ──────────────────────────────────────────────

_META_LMDB_NAME = "metadata_word_index.lmdb"


def build_metadata_word_index(db_path: str) -> int:
    """Build inverted word index LMDB from all normalized_{field}_frequencies files.

    Key: {field}\\x00{word}  Value: NUL-joined original metadata values.
    Cap at 10000 values per word to bound stopword entries.
    Returns the number of keys written.
    """
    from collections import defaultdict

    freq_dir = os.path.join(db_path, "frequencies")
    lmdb_path = os.path.join(freq_dir, _META_LMDB_NAME)
    tmp_path = lmdb_path + ".tmp"

    index: dict[tuple[str, str], set[str]] = defaultdict(set)

    for fname in sorted(os.listdir(freq_dir)):
        if not fname.startswith("normalized_") or not fname.endswith("_frequencies"):
            continue
        if fname.endswith(".lmdb"):
            continue
        field = fname[len("normalized_"):-len("_frequencies")]
        if field == "word":
            continue

        fpath = os.path.join(freq_dir, fname)
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                tab = line.find("\t")
                if tab < 0:
                    continue
                norm_val = line[:tab]
                orig_val = line[tab + 1:].rstrip("\n")
                if not norm_val:
                    continue
                for w in re.findall(r"\w+", norm_val):
                    key = (field, w)
                    if len(index[key]) < 10000:
                        index[key].add(orig_val)

    tmp_env = lmdb.open(tmp_path, map_size=2 * 1024 * 1024 * 1024,
                        writemap=True, sync=False, metasync=False)
    with tmp_env.begin(write=True) as txn:
        for (field, word), originals in index.items():
            key = f"{field}\x00{word}".encode("utf-8")
            val = "\x00".join(originals).encode("utf-8")
            txn.put(key, val)
    tmp_env.sync(True)
    os.makedirs(lmdb_path, exist_ok=True)
    tmp_env.copy(lmdb_path, compact=True)
    tmp_env.close()
    os.system(f"rm -rf {tmp_path}")
    return len(index)


def _get_metadata_index_env(db_path: str) -> lmdb.Environment:
    """Return (and cache) the metadata_word_index.lmdb env (built at index time by PostFilters)."""
    lmdb_path = os.path.join(db_path, "frequencies", _META_LMDB_NAME)
    return get_lmdb_env(lmdb_path)


def metadata_word_lookup(db_path: str, field: str, term: str) -> list[str]:
    """Look up metadata values containing term as a whole word.

    Returns list of original metadata values from the inverted word index.
    """
    env = _get_metadata_index_env(db_path)
    key = f"{field}\x00{term}".encode("utf-8")
    with env.begin(buffers=True) as txn:
        val = txn.get(key)
        if val is None:
            return []
        return bytes(val).decode("utf-8").split("\x00")


def metadata_word_regex_scan(db_path: str, field: str, pattern: str) -> list[str]:
    """Scan metadata word index for words matching a regex pattern.

    Scans all keys for the given field and applies the regex against each
    indexed word.  Returns deduplicated list of original metadata values
    from all matching words.
    """
    env = _get_metadata_index_env(db_path)
    field_prefix = f"{field}\x00".encode("utf-8")
    compiled = re.compile(pattern)
    seen: set[str] = set()
    results: list[str] = []
    with env.begin(buffers=True) as txn:
        cursor = txn.cursor()
        try:
            if not cursor.set_range(field_prefix):
                return results
            while True:
                k = bytes(cursor.key())
                if not k.startswith(field_prefix):
                    break
                word = k[len(field_prefix):].decode("utf-8", errors="replace")
                if compiled.search(word):
                    for val in bytes(cursor.value()).decode("utf-8").split("\x00"):
                        if val not in seen:
                            seen.add(val)
                            results.append(val)
                if not cursor.next():
                    break
        finally:
            cursor.close()
    return results


def metadata_word_prefix_scan(db_path: str, field: str, prefix: str,
                              max_results: int = 100) -> list[str]:
    """Scan metadata word index for words starting with prefix.

    Returns deduplicated list of original metadata values from all matching words.
    Used for metadata autocomplete.
    """
    env = _get_metadata_index_env(db_path)
    key_prefix = f"{field}\x00{prefix}".encode("utf-8")
    seen: set[str] = set()
    results: list[str] = []
    with env.begin(buffers=True) as txn:
        cursor = txn.cursor()
        try:
            if not cursor.set_range(key_prefix):
                return results
            while True:
                k = bytes(cursor.key())
                if not k.startswith(key_prefix):
                    break
                for val in bytes(cursor.value()).decode("utf-8").split("\x00"):
                    if val not in seen:
                        seen.add(val)
                        results.append(val)
                        if len(results) >= max_results:
                            return results
                if not cursor.next():
                    break
        finally:
            cursor.close()
    return results


def expand_autocomplete(kind: str, token: str, frequency_file: str, db_path: str,
                        ascii_conversion: bool, lowercase: bool,
                        max_results: int = 100) -> list[str]:
    """Expand a single autocomplete token using LMDB cursor scans (no subprocess).

    Returns a list of matching word strings:
    - TERM/QUOTE: original word forms from norm_word.lmdb
    - LEMMA/ATTR/LEMMA_ATTR: key strings from words.lmdb (e.g. "lemma:être")

    Supports regex patterns (e.g. sens.*, lemma:virt.*) via cursor + re.match.
    """
    if kind in ("NOT", "OR", "NULL"):
        return []

    if kind in ("TERM", "QUOTE"):
        raw_token = token[1:-1] if kind == "QUOTE" else token
        if not raw_token:
            return []
        env = _get_norm_env(frequency_file)
        with env.begin(buffers=True) as txn:
            if _is_regex_pattern(raw_token):
                norm_prefix, pattern_str = _normalize_pattern(raw_token, lowercase and ascii_conversion)
                return _lmdb_expand_term(txn, norm_prefix, pattern_str, max_results)
            elif ascii_conversion:
                norm_prefix = _norm_key(raw_token, lowercase)
                return _lmdb_expand_term(txn, norm_prefix, None, max_results)
            else:
                # ascii_conversion=False: query token is the norm key as-is
                norm_prefix = raw_token.lower().encode("utf-8") if lowercase else raw_token.encode("utf-8")
                return _lmdb_expand_term(txn, norm_prefix, None, max_results)

    elif kind in ("LEMMA", "ATTR", "LEMMA_ATTR"):
        if not token:
            return []
        scan_env = _get_forms_env(db_path) or get_lmdb_env(os.path.join(db_path, "words.lmdb"))
        with scan_env.begin(buffers=True) as txn:
            if _is_regex_pattern(token):
                literal, meta = _split_literal_prefix(token)
                prefix_bytes = literal.encode("utf-8")
                return _lmdb_scan_keys(txn, prefix_bytes, literal + meta, max_results)
            else:
                return _lmdb_scan_keys(txn, token.encode("utf-8"), None, max_results)

    return []
