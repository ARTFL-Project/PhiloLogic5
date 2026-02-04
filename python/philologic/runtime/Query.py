#!/var/lib/philologic5/philologic_env/bin/python3

import hashlib
import os
import subprocess
import sys
from pathlib import Path

# Set umask before importing numba to ensure cache files are world-writable
os.umask(0o000)

import lmdb
import msgspec
import numba
import numpy as np
import regex as re
from unidecode import unidecode

from philologic.runtime import HitList
from philologic.runtime.QuerySyntax import group_terms, parse_query

# Set Numba cache directory
# Try shared cache first, fall back to /tmp if permission denied
cache_dir = "/var/lib/philologic5/numba_cache"
if not os.access(cache_dir, os.W_OK):
    # In hardened containers, use per-user temp cache
    cache_dir = f"/tmp/philologic_numba_cache_{os.getuid()}"
    os.makedirs(cache_dir, mode=0o755, exist_ok=True)
os.environ["NUMBA_CACHE_DIR"] = cache_dir
numba.config.CACHE_DIR = cache_dir


# =============================================================================
# Document-level search helper functions
# =============================================================================


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

    # For multiple words in OR group, we need to merge and sort
    # This requires a copy, but OR groups are typically small
    combined = np.concatenate(arrays)
    sort_key = combined[:, 0].astype(np.uint64) << 32 | combined[:, 8]
    return combined[np.argsort(sort_key)]


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


def _find_common_docs(doc_lists):
    """Find documents that appear in all word groups."""
    if not doc_lists:
        return np.array([], dtype=np.uint32)

    common = doc_lists[0]
    for docs in doc_lists[1:]:
        common = np.intersect1d(common, docs)
    return common


def _find_sentence_boundaries(hits, cooc_slice=6):
    """Find sentence boundaries within a document's hits.

    Returns: (unique_sentences, start_indices) similar to _find_doc_boundaries.
    Sentence ID is the first `cooc_slice` columns.
    """
    if len(hits) == 0:
        return np.empty((0, cooc_slice), dtype=np.uint32), np.array([0], dtype=np.int64)

    sent_ids = hits[:, :cooc_slice]
    # Find where sentence changes
    changes = np.where(np.any(sent_ids[1:] != sent_ids[:-1], axis=1))[0] + 1
    starts = np.concatenate([[0], changes])
    unique_sents = sent_ids[starts]
    starts = np.concatenate([starts, [len(hits)]])
    return unique_sents, starts


def _find_common_sentences_numpy(hits_list, cooc_slice=6):
    """Find common sentences across all word groups using numpy.

    Args:
        hits_list: List of hit arrays, one per word group (all from same document)
        cooc_slice: Number of columns defining sentence ID (6 for sent, 5 for para)

    Returns:
        List of (group_hits_for_common_sentences, sizes) for each word group,
        or None if no common sentences
    """
    n_groups = len(hits_list)

    # Get sentence boundaries for each group
    sent_info = []
    for hits in hits_list:
        unique_sents, starts = _find_sentence_boundaries(hits, cooc_slice)
        sent_info.append((unique_sents, starts, hits))

    # Find common sentences using structured array view
    dt = np.dtype((np.void, hits_list[0].dtype.itemsize * cooc_slice))

    # Start with first group's sentences
    common_sents = sent_info[0][0]
    common_view = np.ascontiguousarray(common_sents).view(dt).ravel()

    # Intersect with each subsequent group
    group_indices = [None] * n_groups  # Will store indices into each group's unique_sents

    for g in range(n_groups):
        group_sents = sent_info[g][0]
        group_view = np.ascontiguousarray(group_sents).view(dt).ravel()

        if g == 0:
            _, idx_common, idx_group = np.intersect1d(common_view, group_view, return_indices=True)
            group_indices[0] = idx_group
            # Update common to be the intersection
            common_view = common_view[idx_common]
        else:
            _, idx_common, idx_group = np.intersect1d(common_view, group_view, return_indices=True)
            # Update all previous group indices
            for prev_g in range(g):
                group_indices[prev_g] = group_indices[prev_g][idx_common]
            group_indices[g] = idx_group
            common_view = common_view[idx_common]

    n_common = len(common_view)
    if n_common == 0:
        return None

    # Collect hits for common sentences from each group
    result = []
    for g in range(n_groups):
        unique_sents, starts, hits = sent_info[g]
        indices = group_indices[g]

        # Get start/end for each common sentence
        group_starts = starts[indices]
        group_ends = starts[indices + 1]
        counts = group_ends - group_starts

        # Gather hits
        total_hits = counts.sum()
        if total_hits == 0:
            result.append((np.empty((0, 9), dtype=np.uint32), np.array([], dtype=np.int32)))
            continue

        # Build gather indices
        offsets = np.arange(total_hits) - np.repeat(np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
        gather_idx = np.repeat(group_starts, counts) + offsets

        gathered_hits = hits[gather_idx]
        result.append((gathered_hits, counts.astype(np.int32)))

    return result


@numba.jit(nopython=True, cache=True)
def _process_n_groups_numba(all_hits, all_sizes, group_offsets, n_groups, n_sentences,
                            cooc_order, mapping_order, max_distance, exact_distance):
    """Process N word groups using Numba with odometer-style iteration.

    Args:
        all_hits: Concatenated hits array (total_hits x 9)
        all_sizes: 2D array of sizes (n_groups x n_sentences)
        group_offsets: Starting offset for each group in all_hits (n_groups,)
        n_groups: Number of word groups
        n_sentences: Number of sentences to process
        cooc_order: Whether to enforce query order
        mapping_order: Mapping from frequency order to query order
        max_distance: Maximum word distance (0 = no limit)
        exact_distance: If True, distance must equal max_distance

    Returns:
        Output array of valid hits
    """
    # Calculate output row size: 9 + 2*(n_groups-1)
    out_cols = 9 + 2 * (n_groups - 1)

    # Estimate max output size (will trim later)
    max_output = 0
    for sent_idx in range(n_sentences):
        combo_count = 1
        for g in range(n_groups):
            combo_count *= all_sizes[g, sent_idx]
        max_output += combo_count

    if max_output == 0:
        return np.empty((0, out_cols), dtype=np.uint32)

    output = np.empty((max_output, out_cols), dtype=np.uint32)
    out_idx = 0

    # Track current position in each group's hits
    group_pos = np.zeros(n_groups, dtype=np.int64)

    # Arrays for current sentence processing
    indices = np.zeros(n_groups, dtype=np.int64)
    positions = np.zeros(n_groups, dtype=np.int64)
    sorted_indices = np.zeros(n_groups, dtype=np.int64)

    for sent_idx in range(n_sentences):
        # Get sizes for this sentence
        sent_sizes = np.empty(n_groups, dtype=np.int64)
        sent_starts = np.empty(n_groups, dtype=np.int64)
        total_combos = 1

        for g in range(n_groups):
            sent_sizes[g] = all_sizes[g, sent_idx]
            sent_starts[g] = group_offsets[g] + group_pos[g]
            total_combos *= sent_sizes[g]
            group_pos[g] += sent_sizes[g]

        if total_combos == 0:
            continue

        # Track seen outputs for deduplication within sentence
        seen_start = out_idx

        # Reset indices for odometer
        for g in range(n_groups):
            indices[g] = 0

        # Iterate through all combinations
        for _ in range(total_combos):
            # Get positions for current combination
            for g in range(n_groups):
                hit_idx = sent_starts[g] + indices[g]
                positions[g] = all_hits[hit_idx, 7]

            valid = True

            if cooc_order:
                # Map positions to query order and check if sorted
                for g in range(n_groups):
                    sorted_indices[mapping_order[g]] = positions[g]

                # Check if in ascending order
                for g in range(n_groups - 1):
                    if sorted_indices[g] >= sorted_indices[g + 1]:
                        valid = False
                        break
            else:
                # Unordered: just check for duplicate positions
                for g in range(n_groups):
                    for g2 in range(g + 1, n_groups):
                        if positions[g] == positions[g2]:
                            valid = False
                            break
                    if not valid:
                        break

            if valid and max_distance > 0:
                # Find min and max position
                min_pos = positions[0]
                max_pos = positions[0]
                for g in range(1, n_groups):
                    if positions[g] < min_pos:
                        min_pos = positions[g]
                    if positions[g] > max_pos:
                        max_pos = positions[g]

                span = max_pos - min_pos
                if exact_distance:
                    if span != max_distance:
                        valid = False
                else:
                    if span > max_distance:
                        valid = False

            if valid:
                # Sort hits by position for output
                # Simple insertion sort since n_groups is small
                for g in range(n_groups):
                    sorted_indices[g] = g

                for i in range(1, n_groups):
                    key_pos = positions[sorted_indices[i]]
                    key_idx = sorted_indices[i]
                    j = i - 1
                    while j >= 0 and positions[sorted_indices[j]] > key_pos:
                        sorted_indices[j + 1] = sorted_indices[j]
                        j -= 1
                    sorted_indices[j + 1] = key_idx

                # Build output row
                row = np.empty(out_cols, dtype=np.uint32)

                # First word: all 9 columns
                first_g = sorted_indices[0]
                first_hit_idx = sent_starts[first_g] + indices[first_g]
                for c in range(9):
                    row[c] = all_hits[first_hit_idx, c]

                # Remaining words: columns 7-8 only
                col = 9
                for g_idx in range(1, n_groups):
                    g = sorted_indices[g_idx]
                    hit_idx = sent_starts[g] + indices[g]
                    row[col] = all_hits[hit_idx, 7]
                    row[col + 1] = all_hits[hit_idx, 8]
                    col += 2

                # Check for duplicate within this sentence
                is_dup = False
                for k in range(seen_start, out_idx):
                    match = True
                    for m in range(out_cols):
                        if output[k, m] != row[m]:
                            match = False
                            break
                    if match:
                        is_dup = True
                        break

                if not is_dup:
                    output[out_idx] = row
                    out_idx += 1

            # Increment odometer
            for g in range(n_groups - 1, -1, -1):
                indices[g] += 1
                if indices[g] < sent_sizes[g]:
                    break
                indices[g] = 0

    return output[:out_idx]


def _process_n_groups_python(hits_list, sizes_list, cooc_order, mapping_order,
                              max_distance, exact_distance, n_groups):
    """Process N word groups (general case) - calls Numba implementation.

    Returns numpy array directly (not list of bytes) for efficient file writing.
    """
    n_sentences = len(sizes_list[0])

    # Prepare data for Numba function
    # Concatenate all hits into single array
    total_hits = sum(len(h) for h in hits_list)
    all_hits = np.empty((total_hits, 9), dtype=np.uint32)
    group_offsets = np.empty(n_groups, dtype=np.int64)

    offset = 0
    for g in range(n_groups):
        group_offsets[g] = offset
        all_hits[offset:offset + len(hits_list[g])] = hits_list[g]
        offset += len(hits_list[g])

    # Convert sizes to 2D array
    all_sizes = np.array(sizes_list, dtype=np.int64)

    # Convert mapping_order to numpy array
    mapping_order_arr = np.array(mapping_order, dtype=np.int64)

    # Call Numba function - returns numpy array directly
    return _process_n_groups_numba(
        all_hits, all_sizes, group_offsets, n_groups, n_sentences,
        cooc_order, mapping_order_arr, max_distance, exact_distance
    )


class WordData(msgspec.Struct):
    full_array: np.ndarray = None
    array: np.ndarray = None
    start: int = 0
    first_doc: int = 0


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
    parsed = parse_query(terms)
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
    pid = os.fork()
    if pid == 0:  # In child process
        os.umask(0)
        os.setsid()
        pid = os.fork()  # double fork to detach completely from parent
        if pid > 0:
            os._exit(0)
        else:
            with open(f"{filename}.terms", "w") as terms_file:
                expand_query_not(
                    split, frequency_file, terms_file, db.locals.ascii_conversion, db.locals["lowercase_index"]
                )
            method_arg = int(method_arg) if method_arg else 0
            if method == "single_term":
                # Search for one term
                search_word(db.path, filename, db.locals.overflow_words, corpus_file=corpus_file)
            elif method == "phrase_ordered":
                # Phrase searching where words need to be in a specific order with no words in between
                search_phrase(db.path, filename, db.locals.overflow_words, corpus_file=corpus_file)
            elif method == "phrase_unordered":
                # Phrase searching where words need to be in a specific order with possible words in between
                search_within_word_span(db.path, filename, db.locals.overflow_words, method_arg or 1, False, False, corpus_file=corpus_file)
            elif method == "proxy_ordered":
                # Proximity searching with possible words in between
                search_within_word_span(db.path, filename, db.locals.overflow_words, method_arg or 1, True, False, corpus_file=corpus_file)
            elif method == "proxy_unordered":
                # Proximity searching with possible words in between unordered
                search_within_word_span(db.path, filename, db.locals.overflow_words, method_arg or 1, False, False, corpus_file=corpus_file)
            elif method == "exact_cooc_ordered":
                # Co-occurrence searching where words need to be within n words of each other
                search_within_word_span(db.path, filename, db.locals.overflow_words, method_arg or 1, True, True, corpus_file=corpus_file)
            elif method == "exact_cooc_unordered":
                # Co-occurrence searching where words need to be within n words of each othera and can be unordered
                search_within_word_span(db.path, filename, db.locals.overflow_words, method_arg or 1, False, True, corpus_file=corpus_file)
            elif method == "sentence_ordered":  # no support for para search for now
                # Co-occurrence searching where words need to be within an object irrespective of word order
                search_within_text_object(db.path, filename, db.locals.overflow_words, object_level, True, corpus_file=corpus_file)
            elif method == "sentence_unordered":
                # Co-occurrence searching where words need to be within an object irrespective of word order
                search_within_text_object(db.path, filename, db.locals.overflow_words, object_level, False, corpus_file=corpus_file)

            with open(filename + ".done", "w") as flag:  # do something to mark query as finished
                flag.write(" ".join(sys.argv) + "\n")
                flag.flush()  # make sure the file is written to disk. Otherwise we get an infinite loop with 0 hits
            os._exit(0)  # Exit child process
    else:
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


def get_word_array(txn, word, overflow_words, db_path):
    """Returns numpy array either from LMDB buffer or memmap"""
    if word not in overflow_words:
        buffer = txn.get(word.encode("utf8"))
        if buffer is None:
            return np.array([], dtype="u4").reshape(-1, 9)
        return np.frombuffer(buffer, dtype="u4").reshape(-1, 9)
    file_path = os.path.join(db_path, "overflow_words", f"{hashlib.sha256(word.encode('utf8')).hexdigest()}.bin")
    return np.memmap(file_path, dtype="u4", mode='r').reshape(-1, 9)


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
    """Filter philo_ids based on corpus metadata."""
    with open(corpus_file, "rb") as corpus:
        buffer = corpus.read()
        corpus_philo_ids = np.frombuffer(buffer, dtype="u4").reshape(-1, 7)
    corpus_padded = np.pad(corpus_philo_ids, ((0, 0), (0, 1)), 'constant', constant_values=0)
    actual_corpus_lengths = np.argmax(corpus_padded == 0, axis=1)
    if np.all(actual_corpus_lengths == actual_corpus_lengths[0]): # check if all rows have the same length
        object_level = actual_corpus_lengths[0]
        matching_indices = __filter_philo_ids_on_void(corpus_philo_ids[:, :object_level], philo_ids[:, :object_level])
        return philo_ids[matching_indices]
    else:
        unique_lengths = np.unique(actual_corpus_lengths) # get unique lengths
        num_philo_rows = philo_ids.shape[0]
        overall_match_mask = np.zeros(num_philo_rows, dtype=bool)

        for current_len in unique_lengths:
            # Create a mask for the corpus_philo_ids that match the current length
            corpus_rows_for_this_len_mask = (actual_corpus_lengths == current_len)

            # Extract these actual corpus prefixes (all are of length current_len)
            relevant_corpus_prefixes = corpus_philo_ids[corpus_rows_for_this_len_mask, :current_len]
            philo_ids_prefixes = philo_ids[:, :current_len]
            current_matching_indices = __filter_philo_ids_on_void(relevant_corpus_prefixes, philo_ids_prefixes)
            overall_match_mask |= current_matching_indices
        return philo_ids[overall_match_mask]


def search_word(db_path, hitlist_filename, overflow_words, corpus_file=None):
    """Search for a single word in the database."""
    with open(f"{hitlist_filename}.terms", "r") as terms_file:
        words = terms_file.read().split()
    env = lmdb.open(f"{db_path}/words.lmdb", readonly=True, lock=False, readahead=False)
    if len(words) == 1:
        with env.begin(buffers=True) as txn, open(hitlist_filename, "wb") as output_file:
            word = words[0]
            if corpus_file is None:
                if word not in overflow_words:
                    buffer = txn.get(word.encode("utf8"))
                    output_file.write(buffer)
                else:
                    file_path = os.path.join(db_path, "overflow_words", f"{hashlib.sha256(word.encode('utf8')).hexdigest()}.bin")
                    with open(file_path, "rb") as overflow_file:
                        output_file.write(overflow_file.read())
            else:
                word_array = get_word_array(txn, word, overflow_words, db_path)
                filtered_philo_ids = filter_philo_ids(
                    corpus_file,
                    word_array,
                )
                output_file.write(filtered_philo_ids.tobytes())
    else:
        with env.begin(buffers=True) as txn, open(hitlist_filename, "wb") as output_file:
            for philo_ids in merge_word_group(db_path, txn, words, overflow_words):
                if corpus_file is None:
                    output_file.write(philo_ids.tobytes())
                else:
                    filtered_philo_ids = filter_philo_ids(
                        corpus_file,
                        philo_ids,
                    )
                    output_file.write(filtered_philo_ids.tobytes())
    env.close()


def search_phrase(db_path, hitlist_filename, overflow_words, corpus_file=None):
    """Phrase searches where words need to be in a specific order"""
    word_groups = get_word_groups(f"{hitlist_filename}.terms")

    # For phrase search: positions must be consecutive (distance == n_groups - 1)
    max_distance = len(word_groups) - 1
    _search_two_groups_batched(db_path, hitlist_filename, word_groups, overflow_words,
                               cooc_order=True, corpus_file=corpus_file,
                               max_distance=max_distance, exact_distance=True)


def search_within_word_span(db_path, hitlist_filename, overflow_words, n, cooc_order, exact_distance, corpus_file=None):
    """Search for co-occurrences of multiple words within n words of each other in the database."""
    word_groups = get_word_groups(f"{hitlist_filename}.terms")

    if len(word_groups) > 1 and n == 1:
        n = len(word_groups) - 1

    # Use document-level approach for all cases
    _search_two_groups_batched(db_path, hitlist_filename, word_groups, overflow_words,
                               cooc_order=cooc_order, corpus_file=corpus_file,
                               max_distance=n, exact_distance=exact_distance)


def search_within_text_object(db_path, hitlist_filename, overflow_words, level, cooc_order, corpus_file=None):
    """Search for co-occurrences of multiple words in the same sentence in the database."""
    word_groups = get_word_groups(f"{hitlist_filename}.terms")

    # Use document-level approach for all cases
    _search_two_groups_batched(db_path, hitlist_filename, word_groups, overflow_words,
                               cooc_order=cooc_order, corpus_file=corpus_file, level=level)


def _search_two_groups_batched(db_path, hitlist_filename, word_groups, overflow_words,
                                cooc_order, corpus_file, level=None, distance_check=None,
                                max_distance=0, exact_distance=False):
    """Document-level co-occurrence search using numpy optimization.

    Processes documents one at a time for streaming results while using numpy
    vectorized operations within each document for efficiency.

    Uses zero-copy np.frombuffer views into LMDB's memory-mapped pages,
    keeping memory usage minimal even for very large corpora.

    Args:
        db_path: Path to the database
        hitlist_filename: Output hitlist file path
        word_groups: List of word groups from the query
        overflow_words: Set of overflow words
        cooc_order: Whether to enforce query order
        corpus_file: Optional corpus filter file
        level: Text object level ("sent", "para") - only used for sentence/para search
        distance_check: Deprecated, use max_distance/exact_distance instead
        max_distance: Maximum word distance (0 = no limit, 1 = consecutive, n = within n words)
        exact_distance: If True, distance must equal max_distance; if False, distance <= max_distance
    """
    cooc_slice = 6 if level in (None, "sent") else 5
    n_groups = len(word_groups)
    mapping_order = list(range(n_groups))

    # Keep transaction open for the entire processing to use zero-copy views
    env = lmdb.open(f"{db_path}/words.lmdb", readonly=True, lock=False, readahead=False)
    with env.begin(buffers=True) as txn:
        # Load word hits as views (no copy - uses LMDB's mmap)
        all_hits = []
        for group in word_groups:
            hits = _load_word_group_hits(db_path, txn, group, overflow_words)
            all_hits.append(hits)

        # Apply corpus filter if provided (this creates copies for filtered results)
        if corpus_file is not None:
            all_hits = [filter_philo_ids(corpus_file, hits) for hits in all_hits]

        # Find document boundaries for each group
        doc_info = []
        for hits in all_hits:
            doc_ids, starts = _find_doc_boundaries(hits)
            doc_info.append((doc_ids, starts, hits))

        # Find common documents
        doc_lists = [info[0] for info in doc_info]
        common_docs = _find_common_docs(doc_lists)

        with open(hitlist_filename, "wb") as output_file:
            # Process each common document
            for doc_id in common_docs:
                # Extract hits for this document from each group
                doc_hits = []
                for doc_ids, starts, hits in doc_info:
                    idx = np.searchsorted(doc_ids, doc_id)
                    if idx < len(doc_ids) and doc_ids[idx] == doc_id:
                        doc_hits.append(hits[starts[idx]:starts[idx + 1]])
                    else:
                        doc_hits.append(np.empty((0, 9), dtype=np.uint32))

                # Skip if any group has no hits in this document
                if any(len(h) == 0 for h in doc_hits):
                    continue

                # Find common sentences within this document
                common_sent_data = _find_common_sentences_numpy(doc_hits, cooc_slice)
                if common_sent_data is None:
                    continue

                # Process based on number of groups
                if n_groups == 2:
                    hits_w1, sizes_w1 = common_sent_data[0]
                    hits_w2, sizes_w2 = common_sent_data[1]

                    if len(sizes_w1) == 0:
                        continue

                    # For ordered search, first group should have earlier position
                    first_earlier = cooc_order  # True if ordered, False if unordered

                    # Convert sizes to 2D array for Numba function
                    sizes = np.column_stack([sizes_w1, sizes_w2]).astype(np.int64)

                    result = _process_two_groups_batch(
                        hits_w1, hits_w2, sizes,
                        first_earlier, cooc_order, max_distance, exact_distance
                    )

                    if len(result) > 0:
                        output_file.write(result.tobytes())
                else:
                    # General N-group case
                    hits_list = [data[0] for data in common_sent_data]
                    sizes_list = [data[1] for data in common_sent_data]

                    result = _process_n_groups_python(
                        hits_list, sizes_list, cooc_order, mapping_order,
                        max_distance, exact_distance, n_groups
                    )

                    if len(result) > 0:
                        output_file.write(result.tobytes())

    env.close()


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


@numba.jit(nopython=True, cache=True)
def _process_two_groups_batch(all_w1, all_w2, batch_sizes, first_earlier, cooc_order,
                                     max_distance, exact_distance):
    """Process a batch of two-word-group co-occurrences using Numba.

    Builds cartesian products, checks position constraints, and deduplicates hits.

    Args:
        all_w1: Concatenated word1 arrays from all sentences in batch (N x 9)
        all_w2: Concatenated word2 arrays from all sentences in batch (M x 9)
        batch_sizes: Array of (n1, n2) sizes for each sentence (S x 2)
        first_earlier: Whether first group should appear earlier in text (for ordered search)
        cooc_order: Whether to enforce ordering
        max_distance: Maximum word distance (0 = no limit, 1 = consecutive, n = within n words)
        exact_distance: If True, distance must equal max_distance; if False, distance <= max_distance

    Returns:
        Output array of shape (H, 11) with uint32 values for H valid hits
    """
    # Calculate total combinations for allocation
    n_sentences = len(batch_sizes)
    total = 0
    for i in range(n_sentences):
        total += batch_sizes[i, 0] * batch_sizes[i, 1]

    if total == 0:
        return np.empty((0, 11), dtype=np.uint32)

    # Allocate max size output
    output = np.empty((total, 11), dtype=np.uint32)
    out_idx = 0

    w1_offset = 0
    w2_offset = 0

    for sent_idx in range(n_sentences):
        n1 = batch_sizes[sent_idx, 0]
        n2 = batch_sizes[sent_idx, 1]

        seen_start = out_idx  # For deduplication within sentence

        for i in range(n1):
            w1 = all_w1[w1_offset + i]
            for j in range(n2):
                w2 = all_w2[w2_offset + j]

                pos1 = w1[7]
                pos2 = w2[7]

                # Check ordering constraint
                if cooc_order:
                    if first_earlier:
                        if pos1 >= pos2:
                            continue
                    else:
                        if pos1 <= pos2:
                            continue
                else:
                    # Unordered: just need different positions
                    if pos1 == pos2:
                        continue

                # Check distance constraint
                if max_distance > 0:
                    if pos1 > pos2:
                        dist = pos1 - pos2
                    else:
                        dist = pos2 - pos1

                    if exact_distance:
                        if dist != max_distance:
                            continue
                    else:
                        if dist > max_distance:
                            continue

                # Determine first/second by position for output
                if pos1 < pos2:
                    first = w1
                    second = w2
                else:
                    first = w2
                    second = w1

                # Build output row: first[0:9] + second[7:9]
                row = np.empty(11, dtype=np.uint32)
                for c in range(9):
                    row[c] = first[c]
                row[9] = second[7]
                row[10] = second[8]

                # Check for duplicate within this sentence (check all columns)
                is_dup = False
                for k in range(seen_start, out_idx):
                    match = True
                    for m in range(11):
                        if output[k, m] != row[m]:
                            match = False
                            break
                    if match:
                        is_dup = True
                        break

                if not is_dup:
                    output[out_idx] = row
                    out_idx += 1

        w1_offset += n1
        w2_offset += n2

    return output[:out_idx]


def merge_word_group(db_path: str, txn, words: list[str], overflow_words: set[str], chunk_size=None):
    # Initialize data structures for each word
    word_data = {}
    if chunk_size is None:
        chunk_size = 10000  # 10000 hits: happy median between performance and memory usage, potentially reevaluate.

    # Load initial chunks
    for word in words:
        if word not in overflow_words:
            buffer = txn.get(word.encode("utf8"))
            array = np.frombuffer(buffer[:3600], dtype="u4").reshape(-1, 9)
            first_doc = array[0, 0]
            word_data[word] = WordData(full_array=np.frombuffer(buffer, dtype="u4").reshape(-1, 9), array=array, start=0, first_doc=first_doc)
        else:
            file_path = os.path.join(db_path, "overflow_words", f"{hashlib.sha256(word.encode('utf8')).hexdigest()}.bin")
            mmap_array = np.memmap(file_path, dtype="u4", mode='r').reshape(-1, 9)
            array = mmap_array[:100] # 100 36 byte rows
            first_doc = array[0, 0]
            word_data[word] = WordData(full_array=mmap_array, array=array, start=0, first_doc=first_doc)

    def build_first_last_rows():
        first_finishing_doc = np.iinfo(np.uint32).max
        first_finishing_byte = np.iinfo(np.uint32).max
        result = []

        for word, data in word_data.items():
            if data.first_doc > first_finishing_doc or (
                data.first_doc == first_finishing_doc and data.array[0, -1] > first_finishing_byte
            ):  # row starts after finishing row
                continue
            if data.array[-1, 0] < first_finishing_doc or (
                data.array[-1, 0] == first_finishing_doc and data.array[-1, -1] < first_finishing_byte
            ):  # row ends before finishing row
                first_finishing_doc = data.array[-1, 0]
                first_finishing_byte = data.array[-1, -1]
                first_word = word
            result.append((word, data.array[0, ::8], data.array[-1, ::8]))

        return result, np.array([first_finishing_doc, first_finishing_byte], dtype="u4"), first_word

    # Merge sort and write loop
    while word_data:
        # Determine which words start before the first finishing word ends
        # Save index of first row that exceeds the first finishing word
        words_first_last_row, first_finishing_row, first_word_to_finish = build_first_last_rows()

        words_to_keep = []
        for other_word, other_first_row, _ in words_first_last_row:
            if other_word == first_word_to_finish:
                words_to_keep.append((other_word, None))
                continue
            elif other_first_row[0] > first_finishing_row[0] or (
                other_first_row[0] == first_finishing_row[0] and other_first_row[1] > first_finishing_row[1]
            ):  # dismiss words that start before the first finishing word ends
                continue
            else:
                first_exceeding_index = np.searchsorted(
                    word_data[other_word].array[:, 0], first_finishing_row[0], side="right"
                )
                if first_exceeding_index < word_data[other_word].array.shape[0]:
                    remaining_array = word_data[other_word].array[:first_exceeding_index]
                else:
                    remaining_array = word_data[other_word].array
                if np.all(remaining_array[:, 0] < first_finishing_row[0]):  # all doc_ids are less than finishing doc id
                    words_to_keep.append((other_word, remaining_array.shape[0]))
                    continue
                # Are there equal doc_ids? If so, we need to break the tie by comparing byte offsets
                equal_doc_rows = np.where(remaining_array[:, 0] == first_finishing_row[0])
                last_equal_index = equal_doc_rows[0][-1] + 1  # +1 to include the last equal row
                remaining_array = word_data[other_word].array[:last_equal_index]
                exceeding_rows_mask = (remaining_array[:, 0] == first_finishing_row[0]) & (
                    remaining_array[:, -1] > first_finishing_row[1]
                )
                potential_exceeding_indices = np.where(exceeding_rows_mask)
                if potential_exceeding_indices[0].size != 0:
                    first_exceeding_index = potential_exceeding_indices[0][0]
                    words_to_keep.append((other_word, first_exceeding_index))
                else:
                    words_to_keep.append((other_word, last_equal_index))

        # Merge sort partial philo_id arrays
        combined_arrays = np.concatenate(
            [word_data[word].array[:index] for word, index in words_to_keep],
            dtype="u4",
        )
        # Sort by doc id and byte offset, 3x faster than np.lexsort((combined_arrays[:, -1], combined_arrays[:, 0]))
        composite_key = combined_arrays[:, 0].astype(np.uint64) << 32 | combined_arrays[:, -1]
        yield combined_arrays[np.argsort(composite_key, kind="stable")]

        # Load next chunks for all words based on the indices we saved
        for word, index in words_to_keep:
            if index is None:
                index = word_data[word].array.shape[0]  # no need to slice, we have the full array
            word_data[word].start += index
            end = word_data[word].start + chunk_size
            word_data[word].array = word_data[word].full_array[word_data[word].start:end]
            if word_data[word].array.size > 0:
                word_data[word].first_doc = word_data[word].array[0, 0]
            else:
                del word_data[word]


def expand_query_not(split, freq_file, dest_fh, ascii_conversion, lowercase=True):
    """Expand search term"""
    first = True
    grep_proc = None
    for group in split:
        if first == True:
            first = False
        else:  # bare newline starts a new group, except the first
            try:
                dest_fh.write("\n")
            except TypeError:
                dest_fh.write(b"\n")
            dest_fh.flush()

        # find all the NOT terms and separate them out by type
        exclude = []
        for i, g in enumerate(group):
            kind, token = g
            if kind == "NOT":
                exclude = group[i + 1 :]
                group = group[:i]
                break
        cut_proc = subprocess.Popen("cut -f 2 | sort | uniq", stdin=subprocess.PIPE, stdout=dest_fh, shell=True)
        filter_inputs = [cut_proc.stdin]
        filter_procs = [cut_proc]

        # We will chain all NOT operators backward from the main filter.
        for kind, token in exclude:
            if kind == "TERM" and ascii_conversion is True:
                proc = invert_grep(token, subprocess.PIPE, filter_inputs[0], lowercase)
            if kind == "TERM" and ascii_conversion is True:
                proc = invert_grep_exact(token, subprocess.PIPE, filter_inputs[0])
            if kind == "QUOTE":
                token = token[1:-1]
                proc = invert_grep_exact(token, subprocess.PIPE, filter_inputs[0])
            filter_inputs = [proc.stdin] + filter_inputs
            filter_procs = [proc] + filter_procs

        # then we append output from all the greps into the front of that filter chain.
        for kind, token in group:  # or, splits, and ranges should have been taken care of by now.
            if (kind == "TERM" and ascii_conversion is True) or kind == "RANGE":
                grep_proc = grep_word(token, freq_file, filter_inputs[0], lowercase)
                grep_proc.wait()
            elif kind == "TERM" and ascii_conversion is False:
                grep_proc = grep_exact(token, freq_file, filter_inputs[0])
                grep_proc.wait()
            elif kind == "QUOTE":
                token = token[1:-1]
                grep_proc = grep_exact(token, freq_file, filter_inputs[0])
                grep_proc.wait()
            elif kind == "LEMMA":
                grep_proc = grep_word_attributes(token, freq_file, filter_inputs[0], "lemmas")
                grep_proc.wait()
            elif kind == "LEMMA_ATTR":
                grep_proc = grep_word_attributes(token, freq_file, filter_inputs[0], "lemma_word_attributes")
                grep_proc.wait()
            elif kind == "ATTR":
                grep_proc = grep_word_attributes(token, freq_file, filter_inputs[0], "word_attributes")
                grep_proc.wait()
        # close all the pipes and wait for procs to finish.
        for pipe, proc in zip(filter_inputs, filter_procs):
            pipe.close()
            proc.wait()


def grep_word_attributes(token, freq_file, dest_fh, token_type):
    """Grep on lemmas or word attributes"""
    forms_file = os.path.join(os.path.dirname(freq_file), token_type)
    try:
        grep_proc = subprocess.Popen(["rg", "-a", b"^%s$" % token, forms_file], stdout=dest_fh)
    except (UnicodeEncodeError, TypeError):
        grep_proc = subprocess.Popen(["rg", "-a", b"^%s$" % token.encode("utf8"), forms_file], stdout=dest_fh)
    return grep_proc


def grep_word(token, freq_file, dest_fh, lowercase=True):
    """Grep on normalized words"""
    if lowercase:
        token = token.lower()
    norm_tok_uni_chars = unidecode(token)
    norm_tok = "".join(norm_tok_uni_chars)
    try:
        grep_command = ["rg", "-a", "^%s[[:blank:]]" % norm_tok, freq_file]
        grep_proc = subprocess.Popen(grep_command, stdout=dest_fh)
    except (UnicodeEncodeError, TypeError):
        grep_command = ["rg", "-a", b"^%s[[:blank:]]" % norm_tok.encode("utf8"), freq_file]
        grep_proc = subprocess.Popen(grep_command, stdout=dest_fh)
    return grep_proc


def invert_grep(token, in_fh, dest_fh, lowercase=True):
    """NOT grep"""
    if lowercase:
        token = token.lower()
    norm_tok_uni_chars = unidecode(token)
    norm_tok = "".join(norm_tok_uni_chars)
    try:
        grep_command = ["rg", "-a", "-v", "^%s[[:blank:]]" % norm_tok]
        grep_proc = subprocess.Popen(grep_command, stdin=in_fh, stdout=dest_fh)
    except (UnicodeEncodeError, TypeError):
        grep_command = ["rg", "-a", "-v", b"^%s[[:blank:]]" % norm_tok.encode("utf8")]
        grep_proc = subprocess.Popen(grep_command, stdin=in_fh, stdout=dest_fh)
    return grep_proc


def grep_exact(token, freq_file, dest_fh):
    """Exact grep"""
    try:
        grep_proc = subprocess.Popen(["rg", "-a", b"[[:blank:]]%s$" % token, freq_file], stdout=dest_fh)
    except (UnicodeEncodeError, TypeError):
        grep_proc = subprocess.Popen(["rg", "-a", b"[[:blank:]]%s$" % token.encode("utf8"), freq_file], stdout=dest_fh)
    return grep_proc


def invert_grep_exact(token, in_fh, dest_fh):
    """NOT exact grep"""
    # don't strip accent or case, exact match only.
    try:
        grep_proc = subprocess.Popen(["rg", "-a", "-v", b"[[:blank:]]%s$" % token], stdin=in_fh, stdout=dest_fh)
    except (UnicodeEncodeError, TypeError):
        grep_proc = subprocess.Popen(
            ["rg", "-a", "-v", b"[[:blank:]]%s$" % token.encode("utf8")], stdin=in_fh, stdout=dest_fh
        )
    # can't wait because input isn't ready yet.
    return grep_proc


def query_parse(query_terms, config):
    """Parse query function."""
    for pattern, replacement in config.query_parser_regex:
        query_terms = re.sub(rf"{pattern}", rf"{replacement}", query_terms, re.U)
    return query_terms


if __name__ == "__main__":
    path = sys.argv[1]
    terms = sys.argv[2:]
    parsed = parse_query(" ".join(terms))
    print("PARSED:", parsed, file=sys.stderr)
    grouped = group_terms(parsed)
    print("GROUPED:", grouped, file=sys.stderr)
    split = split_terms(grouped)
    print("parsed %d terms:" % len(split), split, file=sys.stderr)
