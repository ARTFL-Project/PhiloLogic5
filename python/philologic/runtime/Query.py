#!/var/lib/philologic5/philologic_env/bin/python3

import hashlib
import os
import sys
import threading
from bisect import bisect_left, bisect_right
from pathlib import Path

# Set Numba cache directory BEFORE importing numba — otherwise Numba resolves
# its cache locator using the default (write next to source file), which fails
# when the source is in a read-only site-packages directory.
_cache_dir = os.environ.get("NUMBA_CACHE_DIR", "/var/lib/philologic5/numba_cache")
if not os.access(_cache_dir, os.W_OK):
    _cache_dir = f"/tmp/philologic_numba_cache_{os.getuid()}"
    os.makedirs(_cache_dir, mode=0o755, exist_ok=True)
os.environ["NUMBA_CACHE_DIR"] = _cache_dir

import lmdb
import numba
import numpy as np
import regex as re

numba.config.CACHE_DIR = _cache_dir

from philologic.runtime import HitList
from philologic.runtime.QuerySyntax import group_terms, parse_query

# Cache LMDB environments per path to avoid "already open in this process" errors
# when multiple threads (gthread workers) query the same database concurrently.
_lmdb_envs: dict[str, lmdb.Environment] = {}
_lmdb_lock = threading.Lock()


def _get_words_env(db_path: str) -> lmdb.Environment:
    """Return a shared read-only LMDB environment for words.lmdb."""
    lmdb_path = f"{db_path}/words.lmdb"
    env = _lmdb_envs.get(lmdb_path)
    if env is not None:
        return env
    with _lmdb_lock:
        env = _lmdb_envs.get(lmdb_path)
        if env is not None:
            return env
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
        _lmdb_envs[lmdb_path] = env
        return env


@numba.jit(nopython=True, cache=True, nogil=True)
def _merge_two_sorted_arrays(arr1, arr2):
    """Merge two sorted hit arrays using two-way merge (O(n)).

    Arrays must be sorted by (doc_id, byte_offset) where doc_id is column 0
    and byte_offset is column 8.
    """
    n1, n2 = len(arr1), len(arr2)
    result = np.empty((n1 + n2, 9), dtype=np.uint32)

    i, j, k = 0, 0, 0
    while i < n1 and j < n2:
        # Compare (doc_id, byte_offset)
        doc1, byte1 = arr1[i, 0], arr1[i, 8]
        doc2, byte2 = arr2[j, 0], arr2[j, 8]

        if doc1 < doc2 or (doc1 == doc2 and byte1 <= byte2):
            result[k] = arr1[i]
            i += 1
        else:
            result[k] = arr2[j]
            j += 1
        k += 1

    # Copy remaining elements
    while i < n1:
        result[k] = arr1[i]
        i += 1
        k += 1
    while j < n2:
        result[k] = arr2[j]
        j += 1
        k += 1

    return result


def _kway_merge_sorted_arrays(arrays):
    """K-way merge of sorted hit arrays using repeated two-way merges.

    Sorts arrays by size so small arrays merge first, ensuring the largest
    array is only touched once in the final merge step.
    """
    if len(arrays) == 0:
        return np.empty((0, 9), dtype=np.uint32)
    if len(arrays) == 1:
        return np.ascontiguousarray(arrays[0])
    if len(arrays) == 2:
        return _merge_two_sorted_arrays(
            np.ascontiguousarray(arrays[0]),
            np.ascontiguousarray(arrays[1])
        )

    # Sort by size so small arrays merge first. The largest array
    # is only touched once in the final merge.
    arrays = sorted(arrays, key=len)

    while len(arrays) > 1:
        merged = []
        for i in range(0, len(arrays), 2):
            if i + 1 < len(arrays):
                merged.append(_merge_two_sorted_arrays(
                    np.ascontiguousarray(arrays[i]),
                    np.ascontiguousarray(arrays[i + 1])
                ))
            else:
                merged.append(arrays[i])
        arrays = merged

    return arrays[0]


def _load_word_group_hits(db_path, txn, words, overflow_words):
    """Load and merge hits for a word group (handles OR within group).

    Uses np.frombuffer to create views into LMDB buffers without copying.
    IMPORTANT: The returned array is only valid while the transaction is open.

    Args:
        db_path: Path to the database
        txn: LMDB transaction (must remain open while using returned array)
        words: List of words in the group
        overflow_words: Set of overflow words

    Returns:
        Numpy array view of hits, sorted by (doc_id, byte_offset)
    """
    arrays = []
    for word in words:
        if word not in overflow_words:
            buffer = txn.get(word.encode("utf8"))
            if buffer:
                # Use frombuffer for zero-copy view into LMDB's mmap
                arr = np.frombuffer(buffer, dtype=np.uint32).reshape(-1, 9)
                arrays.append(arr)
        else:
            # Overflow words use memmap for zero-copy access
            file_path = os.path.join(db_path, "overflow_words",
                                     f"{hashlib.sha256(word.encode('utf8')).hexdigest()}.bin")
            arr = np.memmap(file_path, dtype=np.uint32, mode='r').reshape(-1, 9)
            arrays.append(arr)

    if not arrays:
        return np.empty((0, 9), dtype=np.uint32)

    if len(arrays) == 1:
        return arrays[0]

    # Use k-way merge instead of concatenate+sort
    # This is O(n log k) instead of O(n log n) where k is number of word variants
    # and n is total hits. Since k is typically small (2-10), this is much faster.
    return _kway_merge_sorted_arrays(arrays)


def _load_word_arrays(db_path, txn, words, overflow_words):
    """Load per-word hit arrays without merging.

    Same loading logic as _load_word_group_hits but returns the raw list of
    arrays so callers can do a partial merge before the full merge.
    """
    arrays = []
    for word in words:
        if word not in overflow_words:
            buffer = txn.get(word.encode("utf8"))
            if buffer:
                arr = np.frombuffer(buffer, dtype=np.uint32).reshape(-1, 9)
                arrays.append(arr)
        else:
            file_path = os.path.join(db_path, "overflow_words",
                                     f"{hashlib.sha256(word.encode('utf8')).hexdigest()}.bin")
            arr = np.memmap(file_path, dtype=np.uint32, mode='r').reshape(-1, 9)
            arrays.append(arr)
    return arrays


def _partial_merge_first_n(arrays, n):
    """Extract the first n globally-sorted hits from k sorted arrays.

    Uses numpy sampling: the first n globally-sorted hits from k sorted arrays
    must come from the first n elements of each individual array.

    Returns (first_n_array, remaining_slices) where remaining_slices[i] is the
    unconsumed tail of arrays[i].
    """
    non_empty = [(i, arr) for i, arr in enumerate(arrays) if len(arr) > 0]
    if not non_empty:
        return np.empty((0, 9), dtype=np.uint32), [arr[0:0] for arr in arrays]

    samples, source_ids = [], []
    for i, arr in non_empty:
        take = min(n, len(arr))
        samples.append(np.ascontiguousarray(arr[:take]))
        source_ids.append(np.full(take, i, dtype=np.int64))

    combined = np.concatenate(samples, axis=0)
    sources = np.concatenate(source_ids)

    # Sort by (doc_id, byte_offset) — same key as _merge_two_sorted_arrays
    sort_idx = np.lexsort((combined[:, 8], combined[:, 0]))
    actual_n = min(n, len(combined))
    first_n = np.ascontiguousarray(combined[sort_idx[:actual_n]])

    # Count how many elements from each original array were consumed
    consumed = np.bincount(sources[sort_idx[:actual_n]], minlength=len(arrays)).astype(int)
    remaining = [arrays[i][consumed[i]:] for i in range(len(arrays))]

    return first_n, remaining


def _find_doc_boundaries(hits):
    """Find start indices for each document in sorted hits array.

    Returns: (unique_doc_ids, start_indices) where start_indices[i] is the
    start index for unique_doc_ids[i], with an implicit end at len(hits).
    """
    if len(hits) == 0:
        return np.array([], dtype=np.uint32), np.array([0], dtype=np.int64)

    doc_ids = hits[:, 0]
    # Find where doc_id changes
    changes = np.where(doc_ids[1:] != doc_ids[:-1])[0] + 1
    starts = np.concatenate([[0], changes])
    unique_docs = doc_ids[starts]
    # Add end index
    starts = np.concatenate([starts, [len(hits)]])
    return unique_docs, starts


def __filter_philo_ids_on_void(corpus_philo_ids, philo_ids):
    """Filter philo_ids based on corpus metadata."""

    def __rows_as_void(rows):
        arr_contiguous = np.ascontiguousarray(rows) # ensure array is C-contiguous for a valid view
        # Create a dtype for a row; itemsize is element_size_in_bytes * num_columns
        void_dtype = np.dtype((np.void, arr_contiguous.dtype.itemsize * arr_contiguous.shape[1]))
        return arr_contiguous.view(void_dtype).ravel()

    corpus_philo_ids_void = __rows_as_void(corpus_philo_ids)
    philo_ids_void = __rows_as_void(philo_ids)
    matching_indices_void = np.isin(philo_ids_void, corpus_philo_ids_void)
    return matching_indices_void


def filter_philo_ids(corpus_file, philo_ids) -> np.ndarray:
    """Filter philo_ids to only include hits matching corpus metadata.

    Both arrays are sorted by doc_id (column 0). We use bisect binary search
    directly on the strided column view — O(m·log n) with no contiguous copy,
    avoiding numpy searchsorted's implicit copy of the full column.
    """
    if len(philo_ids) == 0:
        return philo_ids
    with open(corpus_file, "rb") as corpus:
        buffer = corpus.read()
        corpus_philo_ids = np.frombuffer(buffer, dtype="u4").reshape(-1, 7)
    if len(corpus_philo_ids) == 0:
        return np.empty((0, philo_ids.shape[1]), dtype=philo_ids.dtype)


    # Narrow philo_ids to the doc range covered by the corpus.
    # bisect on the strided view avoids numpy's O(n) contiguous copy.
    doc_col = philo_ids[:, 0]
    first_doc = int(corpus_philo_ids[0, 0])
    last_doc = int(corpus_philo_ids[-1, 0])
    lo = bisect_left(doc_col, first_doc)
    hi = bisect_right(doc_col, last_doc)
    philo_ids = philo_ids[lo:hi]
    if len(philo_ids) == 0:
        return philo_ids

    # Determine corpus object level (number of non-zero columns per row).
    corpus_padded = np.pad(corpus_philo_ids, ((0, 0), (0, 1)), "constant", constant_values=0)
    actual_corpus_lengths = np.argmax(corpus_padded == 0, axis=1)
    uniform = np.all(actual_corpus_lengths == actual_corpus_lengths[0])
    object_level = int(actual_corpus_lengths[0]) if uniform else None

    # Collect hits for each corpus doc via bisect on the narrowed strided view.
    doc_col = philo_ids[:, 0]
    empty = np.empty((0, philo_ids.shape[1]), dtype=philo_ids.dtype)

    if object_level == 1:
        # Doc-level corpus: bisect gives exact matches, no further filtering.
        slices = []
        for doc_id in corpus_philo_ids[:, 0]:
            d = int(doc_id)
            s = bisect_left(doc_col, d)
            e = bisect_right(doc_col, d)
            if e > s:
                slices.append(philo_ids[s:e])
        return np.concatenate(slices) if slices else empty

    # Multi-level or mixed-level corpus: narrow by doc_id first, then use
    # np.isin on the reduced array for precise multi-column matching.
    corpus_unique_docs = np.unique(corpus_philo_ids[:, 0])
    slices = []
    for doc_id in corpus_unique_docs:
        d = int(doc_id)
        s = bisect_left(doc_col, d)
        e = bisect_right(doc_col, d)
        if e > s:
            slices.append(philo_ids[s:e])
    if not slices:
        return empty
    philo_ids = np.concatenate(slices)

    if uniform:
        matching = __filter_philo_ids_on_void(corpus_philo_ids[:, :object_level], philo_ids[:, :object_level])
        return philo_ids[matching]
    else:
        unique_lengths = np.unique(actual_corpus_lengths)
        overall_mask = np.zeros(len(philo_ids), dtype=bool)
        for current_len in unique_lengths:
            relevant = corpus_philo_ids[actual_corpus_lengths == current_len, :current_len]
            overall_mask |= __filter_philo_ids_on_void(relevant, philo_ids[:, :current_len])
        return philo_ids[overall_mask]


_EARLY_FLUSH_BYTES = 100 * 9 * 4  # 100 hits × 36 bytes — enough for first page


def _write_with_early_flush(output_file, data):
    """Write data with an early flush so the HitList reader can see the first page immediately."""
    mv = memoryview(data)
    output_file.write(mv[:_EARLY_FLUSH_BYTES])
    output_file.flush()
    if len(mv) > _EARLY_FLUSH_BYTES:
        output_file.write(mv[_EARLY_FLUSH_BYTES:])


def _stream_file_with_early_flush(source_path, output_file):
    """Stream from a file with an early flush after the first chunk."""
    with open(source_path, "rb") as src:
        first_chunk = src.read(_EARLY_FLUSH_BYTES)
        if first_chunk:
            output_file.write(first_chunk)
            output_file.flush()
        while True:
            chunk = src.read(8 * 1024 * 1024)
            if not chunk:
                break
            output_file.write(chunk)


def search_word(db_path, hitlist_filename, overflow_words, corpus_file=None):
    """Search for a single word in the database."""
    with open(f"{hitlist_filename}.terms", "r") as terms_file:
        words = terms_file.read().split()
    env = _get_words_env(db_path)
    if len(words) == 1:
        with env.begin(buffers=True) as txn, open(hitlist_filename, "wb") as output_file:
            word = words[0]
            if corpus_file is None:
                if word not in overflow_words:
                    buffer = txn.get(word.encode("utf8"))
                    if buffer is not None:
                        _write_with_early_flush(output_file, buffer)
                else:
                    file_path = os.path.join(db_path, "overflow_words", f"{hashlib.sha256(word.encode('utf8')).hexdigest()}.bin")
                    _stream_file_with_early_flush(file_path, output_file)
            else:
                word_array = get_word_array(txn, word, overflow_words, db_path)
                filtered_philo_ids = filter_philo_ids(
                    corpus_file,
                    word_array,
                )
                _write_with_early_flush(output_file, filtered_philo_ids.tobytes())
    else:
        with env.begin(buffers=True) as txn, open(hitlist_filename, "wb") as output_file:
            arrays = _load_word_arrays(db_path, txn, words, overflow_words)
            if not arrays:
                pass
            elif corpus_file is None:
                # Phase 1: flush first 100 hits immediately via partial merge
                early_hits, remaining = _partial_merge_first_n(arrays, 100)
                output_file.write(early_hits.tobytes())
                output_file.flush()
                # Phase 2: full merge of remaining hits
                remaining = [r for r in remaining if len(r) > 0]
                if remaining:
                    rest = _kway_merge_sorted_arrays(remaining)
                    output_file.write(rest.tobytes())
            else:
                merged = _kway_merge_sorted_arrays(arrays) if len(arrays) > 1 else arrays[0]
                merged = filter_philo_ids(corpus_file, merged)
                _write_with_early_flush(output_file, merged.tobytes())


def get_word_array(txn, word, overflow_words, db_path):
    """Returns numpy array either from LMDB buffer or memmap"""
    if word not in overflow_words:
        buffer = txn.get(word.encode("utf8"))
        if buffer is None:
            return np.array([], dtype="u4").reshape(-1, 9)
        return np.frombuffer(buffer, dtype="u4").reshape(-1, 9)
    file_path = os.path.join(db_path, "overflow_words", f"{hashlib.sha256(word.encode('utf8')).hexdigest()}.bin")
    return np.memmap(file_path, dtype="u4", mode='r').reshape(-1, 9)


def get_word_groups(terms_file):
    word_groups = []
    with open(terms_file, "r") as terms_file:
        word_group = []
        for line in terms_file:
            word = line.strip()
            if word:
                word_group.append(word)
            elif word_group:
                word_groups.append(word_group)
                word_group = []
        if word_group:
            word_groups.append(word_group)
    return word_groups


def _run_search(db_path, filename, split, frequency_file, ascii_conversion, lowercase_index,
                method, method_arg, overflow_words, object_level, corpus_file):
    """Run search in a background thread. Always writes .done file, even on error."""
    # Lazy imports to avoid circular dependency with multi_word_search
    from philologic.runtime.term_expansion import expand_query_not
    from philologic.runtime.multi_word_search import (
        search_phrase, search_within_word_span, search_within_text_object,
    )

    try:
        with open(f"{filename}.terms", "w") as terms_file:
            expand_query_not(split, frequency_file, terms_file, ascii_conversion, lowercase_index)

        method_arg = int(method_arg) if method_arg else 0
        if method == "single_term":
            search_word(db_path, filename, overflow_words, corpus_file=corpus_file)
        elif method == "phrase_ordered":
            search_phrase(db_path, filename, overflow_words, corpus_file=corpus_file)
        elif method == "phrase_unordered":
            search_within_word_span(db_path, filename, overflow_words, method_arg or 1, False, False, corpus_file=corpus_file)
        elif method == "proxy_ordered":
            search_within_word_span(db_path, filename, overflow_words, method_arg or 1, True, False, corpus_file=corpus_file)
        elif method == "proxy_unordered":
            search_within_word_span(db_path, filename, overflow_words, method_arg or 1, False, False, corpus_file=corpus_file)
        elif method == "exact_cooc_ordered":
            search_within_word_span(db_path, filename, overflow_words, method_arg or 1, True, True, corpus_file=corpus_file)
        elif method == "exact_cooc_unordered":
            search_within_word_span(db_path, filename, overflow_words, method_arg or 1, False, True, corpus_file=corpus_file)
        elif method == "sentence_ordered":
            search_within_text_object(db_path, filename, overflow_words, object_level, True, corpus_file=corpus_file)
        elif method == "sentence_unordered":
            search_within_text_object(db_path, filename, overflow_words, object_level, False, corpus_file=corpus_file)
    finally:
        with open(filename + ".done", "w") as flag:
            flag.write(f"{method} search complete\n")
            flag.flush()


def query(
    db,
    terms,
    corpus_file=None,
    method=None,
    method_arg=None,
    filename="",
    query_debug=False,
    sort_order=None,
    raw_results=False,
    raw_bytes=False,
    ascii_conversion=True,
    object_level="sent",
):
    """Runs concordance queries"""
    sys.stdout.flush()
    parsed = parse_query(terms, query_patterns=db.locals.query_patterns)
    grouped = group_terms(parsed)
    split = split_terms(grouped)
    words_per_hit = len(split)
    if not filename:
        hfile = str(os.getpid()) + ".hitlist"
    dir = db.path + "/hitlists/"
    filename = filename or (dir + hfile)
    if not os.path.exists(filename):
        Path(filename).touch()
    frequency_file = db.path + "/frequencies/normalized_word_frequencies"

    # Run search in a background thread.
    thread = threading.Thread(
        target=_run_search,
        args=(
            db.path, filename, split, frequency_file,
            db.locals.ascii_conversion, db.locals["lowercase_index"],
            method, method_arg, db.locals.overflow_words, object_level, corpus_file,
        ),
        daemon=True,
    )
    thread.start()

    hits = HitList.HitList(
        filename,
        words_per_hit,
        db,
        method=method,
        sort_order=sort_order,
        raw=raw_results,
        raw_bytes=raw_bytes,
        ascii_conversion=ascii_conversion,
    )
    return hits


def get_expanded_query(hitlist):
    fn = hitlist.filename + ".terms"
    query = []
    term = []
    try:
        grep_results = open(fn, "r", encoding="utf8")
    except:
        return []
    for line in grep_results:
        if line == "\n":
            query.append(term)
            term = []
        else:
            term.append('"' + line[:-1] + '"')
    if term:
        query.append(term)
    return query


def split_terms(grouped):
    split = []
    for group in grouped:
        if len(group) == 1:
            kind, token = group[0]
            if kind == "QUOTE" and token.find(" ") > 1:  # we can split quotes on spaces if there is no OR
                for split_tok in token[1:-1].split(" "):
                    split.append((("QUOTE", '"' + split_tok + '"'),))
            elif kind == "RANGE":
                split.append((("TERM", token),))
            else:
                split.append(group)
        else:
            split.append(group)
    return split


def query_parse(query_terms, config):
    """Parse query function."""
    for pattern, replacement in config.query_parser_regex:
        query_terms = re.sub(rf"{pattern}", rf"{replacement}", query_terms, re.U)
    return query_terms


def resolve_method(q, method, method_arg, cooc_order):
    """Resolve user-facing search parameters into internal method name and arg.

    Takes the raw query string, method name, method_arg, and cooc_order from
    the request and returns the (method, arg) pair used by the search engine.
    """
    words = [w for w in q.split() if w]
    method = method or "proxy"
    try:
        arg = int(method_arg)
    except (ValueError, TypeError):
        arg = 0
    if len(words) == 1:
        method = "single_term"
    elif arg == 0 and method in ("proxy", "exact_cooc"):
        if cooc_order == "yes":
            method = "phrase_ordered"
        else:
            method = "phrase_unordered"
    if method == "proxy":
        if cooc_order == "yes" and arg > 0:
            method = "proxy_ordered"
        else:
            method = "proxy_unordered"
    elif method == "exact_cooc":
        if cooc_order == "yes":
            method = "exact_cooc_ordered"
        else:
            method = "exact_cooc_unordered"
    elif method == "sentence":
        arg = 6
        if cooc_order == "yes":
            method = "sentence_ordered"
        else:
            method = "sentence_unordered"
    return method, str(arg)


if __name__ == "__main__":
    path = sys.argv[1]
    terms = sys.argv[2:]
    parsed = parse_query(" ".join(terms))
    print("PARSED:", parsed, file=sys.stderr)
    grouped = group_terms(parsed)
    print("GROUPED:", grouped, file=sys.stderr)
    split = split_terms(grouped)
    print("parsed %d terms:" % len(split), split, file=sys.stderr)
