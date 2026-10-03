#!/var/lib/philologic5/philologic_env/bin/python3
"""LMDB-based term expansion and autocomplete for PhiloLogic queries.

Handles all query-term expansion: normalized word lookups, regex pattern
scanning, LEMMA/ATTR expansion, NOT-term exclusion, and autocomplete.
"""

import os
from contextlib import nullcontext

import lmdb
import regex as re
from unidecode import unidecode

from philologic.runtime.lmdb_env import lmdb_env
from philologic.runtime.QuerySyntax import quoted_text


# Flat files (in frequencies/) that feed word_forms.lmdb
_FORMS_FLAT_FILES = ("lemmas", "word_attributes", "lemma_word_attributes")

# A regex with no literal start scans the whole word index: it expands to this many word forms at most
REGEX_EXPANSION_CAP = 10000


class Forms(list):
    """The word forms a term expands to. cut is True when a cap left some out."""

    cut = False


def _norm(token: str, lowercase: bool = True) -> str:
    if lowercase:
        token = token.lower()
    return "".join(unidecode(token))


def _norm_key(token: str, lowercase: bool = True) -> bytes:
    return _norm(token, lowercase).encode("utf-8")


def _lmdb_lookup(txn, key: bytes) -> list[str]:
    """Return list of original forms for a normalized key, or []. A term normalized to nothing, as an emoji, is no
    key: LMDB fails on an empty one."""
    if not key:
        return []
    val = txn.get(key)
    if val is None:
        return []
    return bytes(val).decode("utf-8").split("\x00")


# ── Regex-pattern detection and LMDB cursor expansion ─────────────────────────

_REGEX_METACHARS = frozenset(".*+?[{(\\")
_QUANTIFIERS = frozenset("*+?{")
_OPTIONAL_QUANTIFIERS = frozenset("*?{")  # those that can match zero of what they apply to


def _is_regex_pattern(token: str) -> bool:
    """Return True if token contains unescaped regex metacharacters and compiles as a regex. A token that does not
    compile, such as "(Art" or "*nvit*", is looked up as a word (which, with its punctuation, it is seldom)."""
    i = 0
    while i < len(token):
        if token[i] == "\\" and i + 1 < len(token):
            i += 2  # skip escaped char
            continue
        if token[i] in _REGEX_METACHARS:
            break
        i += 1
    else:
        return False
    try:
        re.compile(token)
    except re.error:
        return False
    return True


def _split_literal_prefix(token: str) -> tuple[str, str]:
    """Split a regex token into (literal_prefix, meta_suffix) at its first metachar or backslash."""
    for i, char in enumerate(token):
        if char in _REGEX_METACHARS:
            return token[:i], token[i:]
    return token, ""


def _regex_scan_args(token: str, normalize) -> tuple[bytes, str]:
    """Normalize a regex token for LMDB cursor scan + compiled-regex filter.

    Returns (cursor_prefix_bytes, full_pattern_str) where:
    - cursor_prefix_bytes: what every match starts with (for set_range + startswith)
    - full_pattern_str: complete regex pattern (normalized, escaped literal + raw meta suffix)
    normalize turns the token's literal characters into those of the keys scanned.
    """
    literal, meta = _split_literal_prefix(token)
    if literal and meta[:1] in _QUANTIFIERS:
        # The quantifier applies to the literal's last character, which may normalize to several
        head, last = normalize(literal[:-1]), normalize(literal[-1])
        pattern = re.escape(head) + "(?:" + re.escape(last) + ")" + meta
        if meta[0] in _OPTIONAL_QUANTIFIERS:  # matches need not have that character: "couleu?r" matches "couler"
            return head.encode("utf-8"), pattern
        return (head + last).encode("utf-8"), pattern
    norm_literal = normalize(literal)
    return norm_literal.encode("utf-8"), re.escape(norm_literal) + meta


_REGEX_SYNTAX = frozenset(".^$*+?{}[]\\|()")


def _literal(char: str, normalize, quantified: bool) -> str:
    """The regex matching char once normalized: if a quantifier follows, several characters (œ: oe) as one group.
    Only regex syntax is escaped, so that the literal start of the regex, which scans start from, stays as long."""
    normalized = "".join("\\" + c if c in _REGEX_SYNTAX else c for c in normalize(char))
    return f"(?:{normalized})" if quantified and len(normalize(char)) != 1 else normalized


def _class_char(char: str) -> str:
    return "\\" + char if char in "\\]-^[" else char


def _normalize_class(pattern: str, start: int, normalize) -> tuple[int, str]:
    """The character class that starts at pattern[start] ("["), normalized, and the index after it. A character that
    normalizes to several (œ: oe) can't be in a class: it becomes an alternative to it. A class of one character once
    normalized ([éè]: e) is that character, so that a scan can start from it."""
    i = start + 1
    negated = pattern.startswith("^", i)
    i += negated
    items, alternatives, chars = [], [], set()
    first = True
    while i < len(pattern) and (pattern[i] != "]" or first):
        first = False
        if pattern[i] == "\\" and i + 1 < len(pattern):
            char, i = pattern[i + 1], i + 2
            if char.isascii() and char.isalnum():  # \d, \w...
                items.append("\\" + char)
                chars.add(None)
                continue
        else:
            char, i = pattern[i], i + 1
        if pattern.startswith("-", i) and i + 1 < len(pattern) and pattern[i + 1] != "]":  # a range
            end, i = pattern[i + 1], i + 2
            low, high = normalize(char), normalize(end)
            if len(low) == 1 and len(high) == 1:
                items.append(_class_char(min(low, high)) + "-" + _class_char(max(low, high)))
            else:
                items.append(_class_char(char) + "-" + _class_char(end))
            chars.add(None)
            continue
        normalized = normalize(char)
        if len(normalized) == 1:
            items.append(_class_char(normalized))
            chars.add(normalized)
        elif normalized:
            alternatives.append(re.escape(normalized))
    if i >= len(pattern):  # not closed: the token is no regex, and was taken for a word
        return len(pattern), pattern[start:]
    if not negated and not alternatives and len(chars) == 1 and None not in chars:
        return i + 1, re.escape(chars.pop())
    regex_class = "[" + "^" * negated + "".join(items) + "]" if items or negated else ""
    if alternatives and not negated:
        return i + 1, "(?:" + "|".join(([regex_class] if regex_class else []) + alternatives) + ")"
    return i + 1, regex_class


def _normalize_regex(pattern: str, normalize) -> str:
    """pattern with each of its literal characters (classes' too) normalized as the keys it is matched against are:
    "[éè]t" matches "et", "a[MN]our" "amour". Escapes (\\w, \\p{L}), quantifiers and group syntax stay as they are."""
    out = []
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == "\\" and i + 1 < len(pattern):
            escaped = pattern[i + 1]
            if escaped.isascii() and escaped.isalnum():  # \d, \w, \p{L}, \x{e9}, \1...
                end = i + 2
                if escaped in "pPNx" and pattern.startswith("{", end):
                    end = pattern.find("}", end) + 1 or len(pattern)
                out.append(pattern[i:end])
                i = end
            else:
                out.append(_literal(escaped, normalize, pattern[i + 2:i + 3] in _QUANTIFIERS))
                i += 2
        elif char == "[":
            i, regex_class = _normalize_class(pattern, i, normalize)
            out.append(regex_class)
        elif char == "(" and pattern.startswith("(?", i):  # (?:, (?=, (?<!, (?P<name>...
            end = i + 2
            while end < len(pattern) and pattern[end] not in ":=!)>" and not pattern[end].isspace():
                end += 1
            out.append(pattern[i:end + 1])
            i = end + 1
        elif char == "{" and (quantifier := re.match(r"\{\d*(,\d*)?\}", pattern[i:])):
            out.append(quantifier.group())
            i += quantifier.end()
        elif char in _REGEX_SYNTAX:
            out.append(char)
            i += 1
        else:
            out.append(_literal(char, normalize, pattern[i + 1:i + 2] in _QUANTIFIERS))
            i += 1
    return "".join(out)


def _normalize_pattern(token: str, lowercase: bool = True) -> tuple[bytes, str]:
    """_regex_scan_args for norm_word.lmdb, whose keys are normalized words: all the regex's literal characters are
    normalized (not only those before its first metacharacter), so that "pr.mière" matches "premiere"."""
    normalize = lambda s: _norm(s, lowercase)  # noqa: E731
    normalized = _normalize_regex(token, normalize)
    try:
        re.compile(normalized)
    except re.error:
        return _regex_scan_args(token, normalize)
    return _regex_scan_args(normalized, lambda s: s)


def _forms_pattern(token: str) -> tuple[bytes, str]:
    """_regex_scan_args for word_forms.lmdb or words.lmdb, whose keys are lemma and attribute strings as they are."""
    return _regex_scan_args(token, lambda s: s)


def _matcher(pattern_str: str | None, prefix_match: bool):
    """The function telling the keys pattern_str matches (None for all keys): those it matches whole, as egrep did
    on the word list, or with prefix_match those it matches the start of."""
    if not pattern_str:
        return None
    compiled = re.compile(pattern_str)
    return compiled.match if prefix_match else compiled.fullmatch


def _lmdb_expand_term(txn, norm_prefix: bytes, pattern_str: str | None = None,
                      max_results: int = 0, prefix_match: bool = False, form_pattern: str | None = None) -> list[str]:
    """Cursor-scan norm_word.lmdb from norm_prefix, return original word forms.

    If pattern_str is given, keeps the normalized keys it matches: whole, or with prefix_match (for autocomplete),
    at their start. If form_pattern is given, keeps the original forms it matches whole (for quoted terms, which
    are accent-sensitive).
    When norm_prefix is empty, scans the whole DB filtered by pattern_str;
    max_results defaults to REGEX_EXPANSION_CAP in that case to cap unbounded full-DB scans.
    max_results: stop after collecting that many forms (0 = unlimited). The Forms returned are cut if more matched.
    """
    if not norm_prefix and not pattern_str and not form_pattern:
        return Forms()
    form_match = re.compile(form_pattern).fullmatch if form_pattern else None
    if not norm_prefix and max_results == 0:
        max_results = REGEX_EXPANSION_CAP
    match = _matcher(pattern_str, prefix_match)
    results = Forms()
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
            if match is None or match(k.decode("utf-8", errors="replace")):
                for form in bytes(cursor.value()).decode("utf-8").split("\x00"):
                    if form_match is not None and not form_match(form):
                        continue
                    if max_results and len(results) >= max_results:  # one more than the cap allows
                        results.cut = True
                        return results
                    results.append(form)
            if not cursor.next():
                break
    finally:
        cursor.close()
    return results


def _lmdb_scan_keys(txn, prefix: bytes, pattern_str: str | None = None,
                    max_results: int = 0, prefix_match: bool = False) -> list[str]:
    """Cursor-scan LMDB from prefix, return matching key strings.

    Used for LEMMA/ATTR/LEMMA_ATTR expansion against words.lmdb.
    Values (binary hit data) are ignored; only key strings are returned.
    If pattern_str is given, keeps the keys it matches: whole, or with prefix_match (for autocomplete), at their start.
    When prefix is empty, scans whole DB bounded by max_results.
    max_results: stop after collecting that many keys (0 = unlimited).
    """
    if not prefix and not pattern_str:
        return []
    match = _matcher(pattern_str, prefix_match)
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
            if match is None or match(key_str):
                results.append(key_str)
                if max_results and len(results) >= max_results:
                    break
            if not cursor.next():
                break
    finally:
        cursor.close()
    return results


def _lemma_boundary_filter(kind: str, keys: list[str]) -> list[str]:
    """Keep LEMMA regex matches within the lemma form, not its attribute variants.

    A bare lemma key is 'lemma:<form>' (exactly one colon); attribute-qualified
    keys like 'lemma:<form>:pos:NOUN' have more. Filtering to a single colon makes
    'lemma:sentiment.*' match only lemma forms and stop at the attribute boundary.
    (Lemma forms never contain ':', so the colon count is an exact discriminator.)
    """
    if kind != "LEMMA":
        return keys
    return [k for k in keys if k.count(":") == 1]


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
        inner = quoted_text(token)
        if _is_regex_pattern(inner):  # accent-sensitive, as quoted words: matched against the forms as they are
            norm_prefix, _ = _normalize_pattern(inner, lowercase)
            return _lmdb_expand_term(txn, norm_prefix, form_pattern=inner)
        return [inner] if inner else []
    elif kind in ("LEMMA", "LEMMA_ATTR", "ATTR"):
        if _is_regex_pattern(token) and forms_env is not None:
            prefix_bytes, pattern_str = _forms_pattern(token)
            with forms_env.begin(buffers=True) as f_txn:
                keys = _lmdb_scan_keys(f_txn, prefix_bytes, pattern_str)
            return _lemma_boundary_filter(kind, keys)
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
        inner = quoted_text(token)
        if _is_regex_pattern(inner):
            norm_prefix, _ = _normalize_pattern(inner, lowercase)
            return set(_lmdb_expand_term(txn, norm_prefix, form_pattern=inner))
        return {inner}
    elif kind in ("LEMMA", "LEMMA_ATTR", "ATTR"):
        if _is_regex_pattern(token) and forms_env is not None:
            prefix_bytes, pattern_str = _forms_pattern(token)
            with forms_env.begin(buffers=True) as f_txn:
                keys = _lmdb_scan_keys(f_txn, prefix_bytes, pattern_str)
            return set(_lemma_boundary_filter(kind, keys))
        return {token}
    return set()


def _cap_applies(kind: str, token: str, ascii_conversion: bool, lowercase: bool) -> bool:
    """Whether REGEX_EXPANSION_CAP can cut the expansion of a token: a regex with no literal start."""
    if kind in ("TERM", "RANGE") and ascii_conversion:
        pattern = token
    elif kind == "QUOTE":
        pattern = quoted_text(token)
    else:
        return False
    return _is_regex_pattern(pattern) and not _normalize_pattern(pattern, lowercase)[0]


def cut_terms(split, freq_file, ascii_conversion, lowercase=True) -> list[tuple[str, bool]]:
    """The terms of the query groups split whose expansion REGEX_EXPANSION_CAP cut, as expand_query_not expands them,
    each with whether it follows a NOT. A cut term misses some of its forms; a cut NOT excludes too few."""
    tokens = []
    for group in split:
        negated = False
        for kind, token in group:
            if kind == "NOT":
                negated = True
            elif _cap_applies(kind, token, ascii_conversion, lowercase):
                tokens.append((kind, token, negated))
    if not tokens:  # no scan: the usual case
        return []
    cut = []
    with lmdb_env(freq_file + ".lmdb") as env, env.begin(buffers=True) as txn:
        for kind, token, negated in tokens:
            if getattr(_expand_positive(kind, token, txn, ascii_conversion, lowercase), "cut", False):
                cut.append((token, negated))
    return cut


def expand_query_not(split, freq_file, dest_fh, ascii_conversion, lowercase=True):
    """Expand search terms using LMDB index (replaces subprocess/rg pipeline).

    For each query group, expands positive tokens to all matching original word
    forms (including regex patterns like sens.*), subtracts any NOT-excluded
    forms, and writes the result to dest_fh.
    Groups are separated by blank lines (consumed by get_word_groups()).
    """
    db_path = os.path.normpath(os.path.join(os.path.dirname(freq_file), ".."))
    forms_lmdb_path = os.path.join(db_path, "frequencies", "word_forms.lmdb")
    forms = lmdb_env(forms_lmdb_path) if os.path.exists(forms_lmdb_path) else nullcontext()
    first = True

    with lmdb_env(freq_file + ".lmdb") as env, forms as forms_env, env.begin(buffers=True) as txn:
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



def metadata_word_lookup(db_path: str, field: str, term: str) -> list[str]:
    """Look up metadata values containing term as a whole word.

    Returns list of original metadata values from the inverted word index.
    """
    with lmdb_env(os.path.join(db_path, "frequencies", _META_LMDB_NAME)) as env:
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
    with lmdb_env(os.path.join(db_path, "frequencies", _META_LMDB_NAME)) as env:
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
                    if compiled.fullmatch(word):  # whole words, as in word search: e.* was any word with an e
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
    with lmdb_env(os.path.join(db_path, "frequencies", _META_LMDB_NAME)) as env:
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
        raw_token = quoted_text(token) if kind == "QUOTE" else token
        if not raw_token:
            return []
        with lmdb_env(frequency_file + ".lmdb") as env:
            with env.begin(buffers=True) as txn:
                if _is_regex_pattern(raw_token):
                    norm_prefix, pattern_str = _normalize_pattern(raw_token, lowercase and ascii_conversion)
                    return _lmdb_expand_term(txn, norm_prefix, pattern_str, max_results, prefix_match=True)
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
        forms_lmdb_path = os.path.join(db_path, "frequencies", "word_forms.lmdb")
        scan_path = forms_lmdb_path if os.path.exists(forms_lmdb_path) else os.path.join(db_path, "words.lmdb")
        with lmdb_env(scan_path) as scan_env:
            with scan_env.begin(buffers=True) as txn:
                if _is_regex_pattern(token):
                    prefix_bytes, pattern_str = _forms_pattern(token)
                    keys = _lmdb_scan_keys(txn, prefix_bytes, pattern_str, max_results, prefix_match=True)
                    return _lemma_boundary_filter(kind, keys)
                else:
                    return _lmdb_scan_keys(txn, token.encode("utf-8"), None, max_results)

    return []
