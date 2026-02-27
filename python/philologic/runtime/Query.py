#!/var/lib/philologic5/philologic_env/bin/python3

import hashlib
import os
import sys
import threading
import time
from bisect import bisect_left, bisect_right
from pathlib import Path

import lmdb
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


@numba.jit(nopython=True, cache=True, nogil=True)
def _intersect_sorted_rows(arr1, arr2, n_cols):
    """O(n+m) intersection of two sorted 2D arrays by first n_cols columns.

    Returns indices into arr1 and arr2 where rows match.
    Assumes both arrays are sorted and have unique rows (by first n_cols).
    """
    n1, n2 = len(arr1), len(arr2)
    if n1 == 0 or n2 == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    # Pre-allocate for worst case
    max_matches = min(n1, n2)
    idx1 = np.empty(max_matches, dtype=np.int64)
    idx2 = np.empty(max_matches, dtype=np.int64)

    i, j, k = 0, 0, 0
    while i < n1 and j < n2:
        # Compare rows lexicographically by first n_cols columns
        cmp = 0
        for c in range(n_cols):
            if arr1[i, c] < arr2[j, c]:
                cmp = -1
                break
            elif arr1[i, c] > arr2[j, c]:
                cmp = 1
                break

        if cmp == 0:  # Equal
            idx1[k] = i
            idx2[k] = j
            k += 1
            i += 1
            j += 1
        elif cmp < 0:
            i += 1
        else:
            j += 1

    return idx1[:k], idx2[:k]


def _find_common_sentences(hits_list, cooc_slice=6):
    """Find common sentences using Numba-optimized sorted intersection.

    Same interface as _find_common_sentences_numpy but uses O(n+m) merge
    instead of O(n log n) intersect1d.
    """
    n_groups = len(hits_list)

    # Get sentence boundaries for each group
    sent_info = []
    for hits in hits_list:
        unique_sents, starts = _find_sentence_boundaries(hits, cooc_slice)
        sent_info.append((unique_sents, starts, hits))

    # Start with first group's sentences as the "common" set
    common_sents = np.ascontiguousarray(sent_info[0][0])
    group_indices = [None] * n_groups
    group_indices[0] = np.arange(len(common_sents), dtype=np.int64)

    # Intersect with each subsequent group
    for g in range(1, n_groups):
        group_sents = np.ascontiguousarray(sent_info[g][0])

        # Find intersection indices
        idx_common, idx_group = _intersect_sorted_rows(common_sents, group_sents, cooc_slice)

        if len(idx_common) == 0:
            return None

        # Update all previous group indices to map through the intersection
        for prev_g in range(g):
            group_indices[prev_g] = group_indices[prev_g][idx_common]
        group_indices[g] = idx_group

        # Narrow common_sents to the intersection
        common_sents = common_sents[idx_common]

    n_common = len(common_sents)
    if n_common == 0:
        return None

    # Sort common sentences by void dtype order for deterministic output
    dt = np.dtype((np.void, common_sents.dtype.itemsize * cooc_slice))
    common_view = np.ascontiguousarray(common_sents).view(dt).ravel()
    sort_order = np.argsort(common_view)

    # Apply sort order to all group indices
    for g in range(n_groups):
        group_indices[g] = group_indices[g][sort_order]

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


@numba.jit(nopython=True, cache=True, nogil=True)
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


def _process_n_groups(hits_list, sizes_list, cooc_order, mapping_order,
                       max_distance, exact_distance, n_groups):
    """Process N word groups (general case) - prepares data for Numba kernel."""
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


@numba.jit(nopython=True, cache=True, nogil=True)
def _phrase_match_doc(rare_hits, all_doc_hits, group_offsets, group_sizes,
                      n_groups, rare_group_idx, out_cols):
    """Match phrase hits within a single document using binary search.

    For each hit from the rarest word group, computes the expected positions
    for all other groups (consecutive word positions within the same sentence)
    and binary-searches for them.

    Args:
        rare_hits: Rare group's hits in this doc (R x 9, uint32)
        all_doc_hits: Concatenated hits for ALL groups in this doc (T x 9, uint32)
        group_offsets: Start offset of each group in all_doc_hits
        group_sizes: Number of hits for each group in all_doc_hits
        n_groups: Total number of word groups
        rare_group_idx: Which group index is the rare word
        out_cols: Output columns per match (9 + 2 * (n_groups - 1))

    Returns:
        Numpy array of matched phrase hits (M x out_cols, uint32)
    """
    n_rare = len(rare_hits)
    if n_rare == 0:
        return np.empty((0, out_cols), dtype=np.uint32)

    # Pre-allocate output (generous estimate, will truncate)
    max_out = n_rare
    output = np.empty((max_out, out_cols), dtype=np.uint32)
    out_idx = 0

    # Previous output row for dedup
    prev_row = np.zeros(out_cols, dtype=np.uint32)
    have_prev = False

    for r in range(n_rare):
        rare_sent = rare_hits[r, :6]  # sentence ID (cols 0-5)
        rare_pos = rare_hits[r, 7]    # word position within sentence

        # Compute word0_pos: the position of the first word in the phrase
        # rare word is at group rare_group_idx, so word0 is at rare_pos - rare_group_idx
        if rare_pos < rare_group_idx:
            continue  # Can't form phrase - would need negative position

        word0_pos = rare_pos - rare_group_idx

        # Check all groups
        all_found = True
        # We'll store matched hit indices for building output
        matched_indices = np.empty(n_groups, dtype=np.int64)

        for g in range(n_groups):
            if g == rare_group_idx:
                # We already have this hit - store the index in all_doc_hits
                # The rare hits are at group_offsets[rare_group_idx] + rare_idx_within_group
                # But we need the index in all_doc_hits. Since rare_hits is a slice,
                # map r back to all_doc_hits index
                matched_indices[g] = group_offsets[rare_group_idx] + r
                continue

            target_pos = word0_pos + g

            # Binary search in group g's slice for (sent_id, position)
            g_start = group_offsets[g]
            g_size = group_sizes[g]

            lo = 0
            hi = g_size - 1
            found = False

            while lo <= hi:
                mid = (lo + hi) // 2

                # Compare sentence ID first (cols 0-5)
                row_idx = g_start + mid
                cmp = 0
                for c in range(6):
                    if all_doc_hits[row_idx, c] < rare_sent[c]:
                        cmp = -1
                        break
                    elif all_doc_hits[row_idx, c] > rare_sent[c]:
                        cmp = 1
                        break

                if cmp == 0:
                    # Same sentence, compare position (col 7)
                    if all_doc_hits[row_idx, 7] < target_pos:
                        cmp = -1
                    elif all_doc_hits[row_idx, 7] > target_pos:
                        cmp = 1

                if cmp == 0:
                    found = True
                    matched_indices[g] = row_idx
                    break
                elif cmp < 0:
                    lo = mid + 1
                else:
                    hi = mid - 1

            if not found:
                all_found = False
                break

        if not all_found:
            continue

        # Build output row: group 0's full 9 cols,
        # then for groups 1..n-1: (position, byte_offset)
        # Group 0 is at matched_indices[0]
        g0_idx = matched_indices[0]
        row = np.empty(out_cols, dtype=np.uint32)

        # First 9 cols from group 0 (all columns from hit array)
        for c in range(9):
            row[c] = all_doc_hits[g0_idx, c]

        # For each additional group: position, byte_offset
        col = 9
        for g in range(1, n_groups):
            g_idx = matched_indices[g]
            row[col] = all_doc_hits[g_idx, 7]      # word position
            row[col + 1] = all_doc_hits[g_idx, 8]   # byte_offset
            col += 2

        # Simple dedup: skip if identical to previous output row
        if have_prev:
            is_dup = True
            for c in range(out_cols):
                if row[c] != prev_row[c]:
                    is_dup = False
                    break
            if is_dup:
                continue

        # Grow output if needed
        if out_idx >= max_out:
            max_out *= 2
            new_output = np.empty((max_out, out_cols), dtype=np.uint32)
            for i in range(out_idx):
                for c2 in range(out_cols):
                    new_output[i, c2] = output[i, c2]
            output = new_output

        for c in range(out_cols):
            output[out_idx, c] = row[c]
            prev_row[c] = row[c]
        have_prev = True
        out_idx += 1

    return output[:out_idx]


@numba.jit(nopython=True, cache=True, nogil=True)
def _cooc_match_doc_two_groups(w1_hits, w2_hits, cooc_slice,
                                cooc_order, max_distance, exact_distance):
    """Find co-occurring pairs of two word groups within the same text object in one document.

    Computes sentence boundaries for the smaller group, then binary-searches
    the larger group for matching sentence ranges.

    Args:
        w1_hits: Group 0's hits for this document (N1 x 9, sorted by byte_offset)
        w2_hits: Group 1's hits for this document (N2 x 9, sorted by byte_offset)
        cooc_slice: Number of columns defining the text object (6=sent, 5=para)
        cooc_order: Whether group 0 must precede group 1
        max_distance: Max word distance (0=no limit)
        exact_distance: If True, distance must equal max_distance

    Returns:
        Output array (M x 11, uint32) of valid co-occurrence pairs
    """
    n1 = len(w1_hits)
    n2 = len(w2_hits)

    if n1 == 0 or n2 == 0:
        return np.empty((0, 11), dtype=np.uint32)

    # Determine which group to scan for sentences (smaller) vs binary search (larger)
    if n1 <= n2:
        scan_hits = w1_hits
        search_hits = w2_hits
        scan_is_w1 = True
    else:
        scan_hits = w2_hits
        search_hits = w1_hits
        scan_is_w1 = False

    n_scan = len(scan_hits)
    n_search = len(search_hits)

    # Pre-allocate output (will grow if needed)
    max_out = n_scan * 4
    if max_out < 64:
        max_out = 64
    output = np.empty((max_out, 11), dtype=np.uint32)
    out_idx = 0

    # Process each unique sentence in scan_hits
    s = 0
    while s < n_scan:
        # Current sentence ID
        sent = scan_hits[s, :cooc_slice]

        # Find end of this sentence in scan_hits
        s_end = s + 1
        while s_end < n_scan:
            same = True
            for c in range(cooc_slice):
                if scan_hits[s_end, c] != sent[c]:
                    same = False
                    break
            if not same:
                break
            s_end += 1

        # Binary search for lower bound of this sentence in search_hits
        lo = 0
        hi = n_search
        while lo < hi:
            mid = (lo + hi) // 2
            cmp = 0
            for c in range(cooc_slice):
                if search_hits[mid, c] < sent[c]:
                    cmp = -1
                    break
                elif search_hits[mid, c] > sent[c]:
                    cmp = 1
                    break
            if cmp < 0:
                lo = mid + 1
            else:
                hi = mid
        search_start = lo

        # Binary search for upper bound
        lo2 = search_start
        hi2 = n_search
        while lo2 < hi2:
            mid = (lo2 + hi2) // 2
            cmp = 0
            for c in range(cooc_slice):
                if search_hits[mid, c] < sent[c]:
                    cmp = -1
                    break
                elif search_hits[mid, c] > sent[c]:
                    cmp = 1
                    break
            if cmp <= 0:
                lo2 = mid + 1
            else:
                hi2 = mid
        search_end = lo2

        # Produce all valid pairs from this sentence
        if search_start < search_end:
            sent_output_start = out_idx

            for si in range(s, s_end):
                for sj in range(search_start, search_end):
                    # Map back to w1/w2
                    if scan_is_w1:
                        pos1 = scan_hits[si, 7]
                        pos2 = search_hits[sj, 7]
                    else:
                        pos1 = search_hits[sj, 7]
                        pos2 = scan_hits[si, 7]

                    # Check ordering constraint
                    if cooc_order:
                        if pos1 >= pos2:
                            continue
                    else:
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

                    # Build output row: earlier by position first
                    if scan_is_w1:
                        if pos1 < pos2:
                            first_hit = scan_hits[si]
                            second_hit = search_hits[sj]
                        else:
                            first_hit = search_hits[sj]
                            second_hit = scan_hits[si]
                    else:
                        if pos1 < pos2:
                            first_hit = search_hits[sj]
                            second_hit = scan_hits[si]
                        else:
                            first_hit = scan_hits[si]
                            second_hit = search_hits[sj]

                    # Check for duplicate within this sentence
                    is_dup = False
                    for k in range(sent_output_start, out_idx):
                        match = True
                        for c in range(9):
                            if output[k, c] != first_hit[c]:
                                match = False
                                break
                        if match:
                            if output[k, 9] == second_hit[7] and output[k, 10] == second_hit[8]:
                                is_dup = True
                                break
                        # Already not matching, continue
                    if is_dup:
                        continue

                    # Grow output if needed
                    if out_idx >= max_out:
                        max_out *= 2
                        new_output = np.empty((max_out, 11), dtype=np.uint32)
                        for i in range(out_idx):
                            for c2 in range(11):
                                new_output[i, c2] = output[i, c2]
                        output = new_output

                    for c in range(9):
                        output[out_idx, c] = first_hit[c]
                    output[out_idx, 9] = second_hit[7]
                    output[out_idx, 10] = second_hit[8]
                    out_idx += 1

        s = s_end

    return output[:out_idx]


@numba.jit(nopython=True, cache=True, nogil=True)
def _cooc_match_all_docs_two_groups(
    w1_all, w2_all,
    g0_starts, g0_ends, g1_starts, g1_ends,
    n_docs, cooc_slice, cooc_order, max_distance, exact_distance,
):
    """Fused document-loop kernel for 2-group co-occurrence matching.

    Processes ALL documents in a single Numba call, eliminating Python-level
    per-document dispatch overhead (~22μs × N_docs).

    Args:
        w1_all, w2_all: Full contiguous hit arrays for groups 0 and 1
        g0_starts, g0_ends: Per-document index ranges into w1_all (int64[n_docs])
        g1_starts, g1_ends: Per-document index ranges into w2_all (int64[n_docs])
        n_docs: Number of documents to process
        cooc_slice: Columns defining text object (6=sent, 5=para)
        cooc_order: Whether group 0 must precede group 1
        max_distance: Max word distance (0=no limit)
        exact_distance: If True, distance must equal max_distance
    """
    max_out = 4096
    output = np.empty((max_out, 11), dtype=np.uint32)
    out_idx = 0

    for d_idx in range(n_docs):
        g0_s = g0_starts[d_idx]
        g0_e = g0_ends[d_idx]
        g1_s = g1_starts[d_idx]
        g1_e = g1_ends[d_idx]

        n1 = g0_e - g0_s
        n2 = g1_e - g1_s
        if n1 <= 0 or n2 <= 0:
            continue

        # Pick smaller group to scan, larger to binary-search
        if n1 <= n2:
            scan_arr = w1_all
            search_arr = w2_all
            scan_s = g0_s
            search_s = g1_s
            n_scan = n1
            n_search = n2
            scan_is_w1 = True
        else:
            scan_arr = w2_all
            search_arr = w1_all
            scan_s = g1_s
            search_s = g0_s
            n_scan = n2
            n_search = n1
            scan_is_w1 = False

        # Process each unique sentence in scan group
        s = 0
        while s < n_scan:
            si_abs = scan_s + s

            # Find end of this sentence
            s_end = s + 1
            while s_end < n_scan:
                si2_abs = scan_s + s_end
                same = True
                for c in range(cooc_slice):
                    if scan_arr[si2_abs, c] != scan_arr[si_abs, c]:
                        same = False
                        break
                if not same:
                    break
                s_end += 1

            # Binary search: lower bound
            lo, hi = 0, n_search
            while lo < hi:
                mid = (lo + hi) // 2
                mid_abs = search_s + mid
                cmp = 0
                for c in range(cooc_slice):
                    if search_arr[mid_abs, c] < scan_arr[si_abs, c]:
                        cmp = -1
                        break
                    elif search_arr[mid_abs, c] > scan_arr[si_abs, c]:
                        cmp = 1
                        break
                if cmp < 0:
                    lo = mid + 1
                else:
                    hi = mid
            search_start = lo

            # Binary search: upper bound
            lo2, hi2 = search_start, n_search
            while lo2 < hi2:
                mid = (lo2 + hi2) // 2
                mid_abs = search_s + mid
                cmp = 0
                for c in range(cooc_slice):
                    if search_arr[mid_abs, c] < scan_arr[si_abs, c]:
                        cmp = -1
                        break
                    elif search_arr[mid_abs, c] > scan_arr[si_abs, c]:
                        cmp = 1
                        break
                if cmp <= 0:
                    lo2 = mid + 1
                else:
                    hi2 = mid
            search_end = lo2

            if search_start < search_end:
                sent_output_start = out_idx

                for si in range(s, s_end):
                    si_a = scan_s + si
                    for sj in range(search_start, search_end):
                        sj_a = search_s + sj

                        if scan_is_w1:
                            pos1 = scan_arr[si_a, 7]
                            pos2 = search_arr[sj_a, 7]
                        else:
                            pos1 = search_arr[sj_a, 7]
                            pos2 = scan_arr[si_a, 7]

                        if cooc_order:
                            if pos1 >= pos2:
                                continue
                        else:
                            if pos1 == pos2:
                                continue

                        if max_distance > 0:
                            dist = pos1 - pos2 if pos1 > pos2 else pos2 - pos1
                            if exact_distance:
                                if dist != max_distance:
                                    continue
                            else:
                                if dist > max_distance:
                                    continue

                        # Determine output order (earlier position first)
                        if scan_is_w1:
                            if pos1 < pos2:
                                f_a, f_arr, s_a, s_arr = si_a, scan_arr, sj_a, search_arr
                            else:
                                f_a, f_arr, s_a, s_arr = sj_a, search_arr, si_a, scan_arr
                        else:
                            if pos1 < pos2:
                                f_a, f_arr, s_a, s_arr = sj_a, search_arr, si_a, scan_arr
                            else:
                                f_a, f_arr, s_a, s_arr = si_a, scan_arr, sj_a, search_arr

                        # Duplicate check within this sentence
                        is_dup = False
                        for k in range(sent_output_start, out_idx):
                            match = True
                            for c in range(9):
                                if output[k, c] != f_arr[f_a, c]:
                                    match = False
                                    break
                            if match:
                                if output[k, 9] == s_arr[s_a, 7] and output[k, 10] == s_arr[s_a, 8]:
                                    is_dup = True
                                    break
                        if is_dup:
                            continue

                        # Grow output if needed
                        if out_idx >= max_out:
                            max_out *= 2
                            new_output = np.empty((max_out, 11), dtype=np.uint32)
                            for i in range(out_idx):
                                for c2 in range(11):
                                    new_output[i, c2] = output[i, c2]
                            output = new_output

                        for c in range(9):
                            output[out_idx, c] = f_arr[f_a, c]
                        output[out_idx, 9] = s_arr[s_a, 7]
                        output[out_idx, 10] = s_arr[s_a, 8]
                        out_idx += 1

            s = s_end

    return output[:out_idx]


def _run_search(db_path, filename, split, frequency_file, ascii_conversion, lowercase_index,
                method, method_arg, overflow_words, object_level, corpus_file):
    """Run search in a background thread. Always writes .done file, even on error."""
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
    env = lmdb.open(f"{db_path}/words.lmdb", readonly=True, lock=False, readahead=False)
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
    env.close()


def _phrase_process_doc(group_doc_slices, n_groups, rare_group_idx, out_cols):
    """Run phrase kernel on one document's per-group hit slices. Returns result array."""
    total_doc_hits = sum(len(s) for s in group_doc_slices)
    all_doc_hits = np.empty((total_doc_hits, 9), dtype=np.uint32)
    group_offsets = np.empty(n_groups, dtype=np.int64)
    group_sizes_arr = np.empty(n_groups, dtype=np.int64)

    offset = 0
    for g in range(n_groups):
        group_offsets[g] = offset
        group_sizes_arr[g] = len(group_doc_slices[g])
        all_doc_hits[offset:offset + group_sizes_arr[g]] = group_doc_slices[g]
        offset += group_sizes_arr[g]

    rare_in_concat = all_doc_hits[group_offsets[rare_group_idx]:
                                  group_offsets[rare_group_idx] + group_sizes_arr[rare_group_idx]]
    return _phrase_match_doc(
        rare_in_concat, all_doc_hits, group_offsets, group_sizes_arr,
        n_groups, rare_group_idx, out_cols
    )


def search_phrase(db_path, hitlist_filename, overflow_words, corpus_file=None):
    """Phrase searches where words need to be in a specific order.

    Scans only the rarest word group for document boundaries, then uses
    binary search within each document to find consecutive word positions
    forming the phrase.
    """
    word_groups = get_word_groups(f"{hitlist_filename}.terms")
    n_groups = len(word_groups)

    # Single word: delegate to search_word
    if n_groups <= 1:
        search_word(db_path, hitlist_filename, overflow_words, corpus_file=corpus_file)
        return

    out_cols = 9 + 2 * (n_groups - 1)

    env = lmdb.open(f"{db_path}/words.lmdb", readonly=True, lock=False, readahead=False)
    with env.begin(buffers=True) as txn:
        if corpus_file is None:
            # Merge-free Phase 1 + merged Phase 2 for early flush
            group_arrays = [_load_word_arrays(db_path, txn, g, overflow_words) for g in word_groups]
            group_counts = [sum(len(a) for a in arrays) for arrays in group_arrays]

            if any(c == 0 for c in group_counts):
                env.close()
                return

            rare_group_idx = int(np.argmin(np.array(group_counts)))
            rare_arrays = group_arrays[rare_group_idx]

            rare_doc_ids = np.unique(np.concatenate([arr[:, 0] for arr in rare_arrays]))
            batch = min(500, len(rare_doc_ids))

            with open(hitlist_filename, "wb") as output_file:
                # Phase 1: per-form array extraction, no merge needed
                if batch > 0:
                    batch_docs = rare_doc_ids[:batch]
                    all_form_lefts = []
                    all_form_rights = []
                    for g in range(n_groups):
                        fl = [np.searchsorted(arr[:, 0], batch_docs, side='left') for arr in group_arrays[g]]
                        fr = [np.searchsorted(arr[:, 0], batch_docs, side='right') for arr in group_arrays[g]]
                        all_form_lefts.append(fl)
                        all_form_rights.append(fr)

                    flushed = False
                    for d_idx in range(batch):
                        skip_doc = False
                        group_doc_slices = [None] * n_groups
                        for g in range(n_groups):
                            parts = []
                            for k in range(len(group_arrays[g])):
                                left = all_form_lefts[g][k][d_idx]
                                right = all_form_rights[g][k][d_idx]
                                if right > left:
                                    parts.append(group_arrays[g][k][left:right])
                            if not parts:
                                skip_doc = True
                                break
                            doc_hits = np.concatenate(parts) if len(parts) > 1 else parts[0]
                            if len(parts) > 1:
                                doc_hits = np.ascontiguousarray(doc_hits[np.argsort(doc_hits[:, 8])])
                            group_doc_slices[g] = doc_hits

                        if skip_doc:
                            continue

                        result = _phrase_process_doc(group_doc_slices, n_groups, rare_group_idx, out_cols)
                        if len(result) > 0:
                            output_file.write(result.tobytes())
                            if not flushed:
                                output_file.flush()
                                flushed = True

                # Phase 2: full merge for remaining docs
                n_remaining = len(rare_doc_ids) - batch
                if n_remaining > 0:
                    all_hits = []
                    for g in range(n_groups):
                        arrays = group_arrays[g]
                        if len(arrays) > 1:
                            all_hits.append(_kway_merge_sorted_arrays(arrays))
                        elif len(arrays) == 1:
                            all_hits.append(arrays[0])
                        else:
                            all_hits.append(np.empty((0, 9), dtype=np.uint32))

                    rare_docs, rare_starts = _find_doc_boundaries(all_hits[rare_group_idx])
                    phase2_start = np.searchsorted(rare_docs, rare_doc_ids[batch - 1], side='right') if batch > 0 else 0

                    lefts = {}
                    rights = {}
                    for g in range(n_groups):
                        if g == rare_group_idx:
                            continue
                        g_doc_ids = all_hits[g][:, 0]
                        lefts[g] = np.searchsorted(g_doc_ids, rare_docs, side='left')
                        rights[g] = np.searchsorted(g_doc_ids, rare_docs, side='right')

                    for d_idx in range(phase2_start, len(rare_docs)):
                        rare_doc_hits = all_hits[rare_group_idx][rare_starts[d_idx]:rare_starts[d_idx + 1]]

                        skip_doc = False
                        group_doc_slices = [None] * n_groups
                        group_doc_slices[rare_group_idx] = rare_doc_hits

                        for g in range(n_groups):
                            if g == rare_group_idx:
                                continue
                            g_left = lefts[g][d_idx]
                            g_right = rights[g][d_idx]
                            if g_left >= g_right:
                                skip_doc = True
                                break
                            group_doc_slices[g] = all_hits[g][g_left:g_right]

                        if skip_doc:
                            continue

                        result = _phrase_process_doc(group_doc_slices, n_groups, rare_group_idx, out_cols)
                        if len(result) > 0:
                            output_file.write(result.tobytes())
        else:
            # Corpus-filtered: use merged path
            all_hits = []
            for group in word_groups:
                hits = _load_word_group_hits(db_path, txn, group, overflow_words)
                all_hits.append(hits)

            all_hits = [filter_philo_ids(corpus_file, hits) for hits in all_hits]

            if any(len(h) == 0 for h in all_hits):
                env.close()
                return

            sizes = [len(h) for h in all_hits]
            rare_group_idx = int(np.argmin(np.array(sizes)))
            rare_docs, rare_starts = _find_doc_boundaries(all_hits[rare_group_idx])

            lefts = {}
            rights = {}
            for g in range(n_groups):
                if g == rare_group_idx:
                    continue
                g_doc_ids = all_hits[g][:, 0]
                lefts[g] = np.searchsorted(g_doc_ids, rare_docs, side='left')
                rights[g] = np.searchsorted(g_doc_ids, rare_docs, side='right')

            with open(hitlist_filename, "wb") as output_file:
                flushed = False
                for d_idx in range(len(rare_docs)):
                    rare_doc_hits = all_hits[rare_group_idx][rare_starts[d_idx]:rare_starts[d_idx + 1]]

                    skip_doc = False
                    group_doc_slices = [None] * n_groups
                    group_doc_slices[rare_group_idx] = rare_doc_hits

                    for g in range(n_groups):
                        if g == rare_group_idx:
                            continue
                        g_left = lefts[g][d_idx]
                        g_right = rights[g][d_idx]
                        if g_left >= g_right:
                            skip_doc = True
                            break
                        group_doc_slices[g] = all_hits[g][g_left:g_right]

                    if skip_doc:
                        continue

                    result = _phrase_process_doc(group_doc_slices, n_groups, rare_group_idx, out_cols)
                    if len(result) > 0:
                        output_file.write(result.tobytes())
                        if not flushed:
                            output_file.flush()
                            flushed = True

    env.close()


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
        # For 2-group case: use merge-free early flush then full merge
        if n_groups == 2 and corpus_file is None:
            # Load per-form arrays without merging (instant, zero-copy)
            group_arrays = [_load_word_arrays(db_path, txn, g, overflow_words) for g in word_groups]
            group_counts = [sum(len(a) for a in arrays) for arrays in group_arrays]

            # Determine rare group (fewer total hits)
            rare_idx = 0 if group_counts[0] <= group_counts[1] else 1
            common_idx = 1 - rare_idx
            rare_arrays = group_arrays[rare_idx]
            common_arrays = group_arrays[common_idx]

            # Phase 1: merge-free early flush using per-form arrays for BOTH groups
            # Extract unique doc IDs from rare group without merging
            if rare_arrays:
                rare_doc_ids = np.unique(np.concatenate([arr[:, 0] for arr in rare_arrays]))
            else:
                rare_doc_ids = np.empty(0, dtype=np.uint32)

            batch = min(500, len(rare_doc_ids))

            with open(hitlist_filename, "wb") as output_file:
                if batch > 0 and common_arrays:
                    batch_docs = rare_doc_ids[:batch]
                    # Pre-compute per-doc ranges in each form array for BOTH groups
                    rare_form_lefts = [np.searchsorted(arr[:, 0], batch_docs, side='left')
                                       for arr in rare_arrays]
                    rare_form_rights = [np.searchsorted(arr[:, 0], batch_docs, side='right')
                                        for arr in rare_arrays]
                    common_form_lefts = [np.searchsorted(arr[:, 0], batch_docs, side='left')
                                         for arr in common_arrays]
                    common_form_rights = [np.searchsorted(arr[:, 0], batch_docs, side='right')
                                          for arr in common_arrays]

                    flushed = False
                    for d_idx in range(batch):
                        # Gather rare-group hits from all form arrays
                        r_parts = []
                        for k in range(len(rare_arrays)):
                            left, right = rare_form_lefts[k][d_idx], rare_form_rights[k][d_idx]
                            if right > left:
                                r_parts.append(rare_arrays[k][left:right])
                        if not r_parts:
                            continue
                        rare_doc = np.concatenate(r_parts) if len(r_parts) > 1 else r_parts[0]
                        if len(r_parts) > 1:
                            rare_doc = np.ascontiguousarray(rare_doc[np.argsort(rare_doc[:, 8])])

                        # Gather common-group hits from all form arrays
                        c_parts = []
                        for k in range(len(common_arrays)):
                            left, right = common_form_lefts[k][d_idx], common_form_rights[k][d_idx]
                            if right > left:
                                c_parts.append(common_arrays[k][left:right])
                        if not c_parts:
                            continue
                        common_doc = np.concatenate(c_parts) if len(c_parts) > 1 else c_parts[0]
                        if len(c_parts) > 1:
                            common_doc = np.ascontiguousarray(common_doc[np.argsort(common_doc[:, 8])])

                        # Call per-document kernel (group 0 = first positional group)
                        if rare_idx == 0:
                            result = _cooc_match_doc_two_groups(
                                rare_doc, common_doc, cooc_slice,
                                cooc_order, max_distance, exact_distance)
                        else:
                            result = _cooc_match_doc_two_groups(
                                common_doc, rare_doc, cooc_slice,
                                cooc_order, max_distance, exact_distance)
                        if len(result) > 0:
                            output_file.write(result.tobytes())
                            if not flushed:
                                output_file.flush()
                                flushed = True

                # Phase 2: full merge both groups, process remaining docs with fused kernel
                n_remaining_docs = len(rare_doc_ids) - batch
                if n_remaining_docs > 0:
                    # Merge both groups fully
                    all_hits = []
                    for g in range(n_groups):
                        arrays = group_arrays[g]
                        if len(arrays) > 1:
                            all_hits.append(_kway_merge_sorted_arrays(arrays))
                        elif len(arrays) == 1:
                            all_hits.append(arrays[0])
                        else:
                            all_hits.append(np.empty((0, 9), dtype=np.uint32))

                    # Recompute rare group info from merged array
                    rare_docs, rare_starts = _find_doc_boundaries(all_hits[rare_idx])
                    n_docs = len(rare_docs)

                    # Find where the batch docs end in the merged rare array
                    # (docs are sorted, so we can find the offset)
                    phase2_start = np.searchsorted(rare_docs, rare_doc_ids[batch - 1], side='right') if batch > 0 else 0
                    remaining = n_docs - phase2_start

                    if remaining > 0:
                        other_g = common_idx
                        g_doc_ids = all_hits[other_g][:, 0]
                        cm_lefts = np.searchsorted(g_doc_ids, rare_docs[phase2_start:], side='left')
                        cm_rights = np.searchsorted(g_doc_ids, rare_docs[phase2_start:], side='right')

                        if rare_idx == 0:
                            g0_starts = rare_starts[phase2_start:-1].astype(np.int64)
                            g0_ends = rare_starts[phase2_start + 1:].astype(np.int64)
                            g1_starts = cm_lefts.astype(np.int64)
                            g1_ends = cm_rights.astype(np.int64)
                        else:
                            g0_starts = cm_lefts.astype(np.int64)
                            g0_ends = cm_rights.astype(np.int64)
                            g1_starts = rare_starts[phase2_start:-1].astype(np.int64)
                            g1_ends = rare_starts[phase2_start + 1:].astype(np.int64)

                        w1 = np.ascontiguousarray(all_hits[0])
                        w2 = np.ascontiguousarray(all_hits[1])

                        # Phase 2a: small batch for early flush
                        batch2 = min(100, remaining)
                        result2a = _cooc_match_all_docs_two_groups(
                            w1, w2, g0_starts[:batch2], g0_ends[:batch2],
                            g1_starts[:batch2], g1_ends[:batch2],
                            batch2, cooc_slice, cooc_order, max_distance, exact_distance,
                        )
                        if len(result2a) > 0:
                            output_file.write(result2a.tobytes())
                            output_file.flush()

                        # Phase 2b: remaining documents
                        remaining2 = remaining - batch2
                        if remaining2 > 0:
                            result2b = _cooc_match_all_docs_two_groups(
                                w1, w2, g0_starts[batch2:], g0_ends[batch2:],
                                g1_starts[batch2:], g1_ends[batch2:],
                                remaining2, cooc_slice, cooc_order, max_distance, exact_distance,
                            )
                            if len(result2b) > 0:
                                output_file.write(result2b.tobytes())
        else:
            # N-group or corpus-filtered: use original merged path
            all_hits = []
            for group in word_groups:
                hits = _load_word_group_hits(db_path, txn, group, overflow_words)
                all_hits.append(hits)

            if corpus_file is not None:
                all_hits = [filter_philo_ids(corpus_file, hits) for hits in all_hits]

            sizes = [len(h) for h in all_hits]
            rare_idx = int(np.argmin(np.array(sizes)))
            rare_docs, rare_starts = _find_doc_boundaries(all_hits[rare_idx])

            other_lefts = {}
            other_rights = {}
            for g in range(n_groups):
                if g == rare_idx:
                    continue
                g_doc_ids = all_hits[g][:, 0]
                other_lefts[g] = np.searchsorted(g_doc_ids, rare_docs, side='left')
                other_rights[g] = np.searchsorted(g_doc_ids, rare_docs, side='right')

            with open(hitlist_filename, "wb") as output_file:
                if n_groups == 2:
                    n_docs = len(rare_docs)
                    other_g = 1 - rare_idx
                    if rare_idx == 0:
                        g0_starts = rare_starts[:-1].astype(np.int64)
                        g0_ends = rare_starts[1:].astype(np.int64)
                        g1_starts = other_lefts[other_g].astype(np.int64)
                        g1_ends = other_rights[other_g].astype(np.int64)
                    else:
                        g0_starts = other_lefts[other_g].astype(np.int64)
                        g0_ends = other_rights[other_g].astype(np.int64)
                        g1_starts = rare_starts[:-1].astype(np.int64)
                        g1_ends = rare_starts[1:].astype(np.int64)

                    w1 = np.ascontiguousarray(all_hits[0])
                    w2 = np.ascontiguousarray(all_hits[1])
                    result = _cooc_match_all_docs_two_groups(
                        w1, w2, g0_starts, g0_ends, g1_starts, g1_ends,
                        n_docs, cooc_slice, cooc_order, max_distance, exact_distance,
                    )
                    if len(result) > 0:
                        _write_with_early_flush(output_file, result.tobytes())

                else:
                    # N-group case: per-document loop
                    flushed = False
                    for d_idx in range(len(rare_docs)):
                        empty = np.empty((0, 9), dtype=np.uint32)
                        doc_hits = [empty] * n_groups
                        doc_hits[rare_idx] = all_hits[rare_idx][rare_starts[d_idx]:rare_starts[d_idx + 1]]

                        skip_doc = False
                        for g in range(n_groups):
                            if g == rare_idx:
                                continue
                            left = other_lefts[g][d_idx]
                            right = other_rights[g][d_idx]
                            if left >= right:
                                skip_doc = True
                                break
                            doc_hits[g] = all_hits[g][left:right]

                        if skip_doc:
                            continue

                        common_sent_data = _find_common_sentences(doc_hits, cooc_slice)
                        if common_sent_data is None:
                            continue

                        hits_list = [data[0] for data in common_sent_data]
                        sizes_list = [data[1] for data in common_sent_data]

                        result = _process_n_groups(
                            hits_list, sizes_list, cooc_order, mapping_order,
                            max_distance, exact_distance, n_groups
                        )

                        if len(result) > 0:
                            output_file.write(result.tobytes())
                            if not flushed:
                                output_file.flush()
                                flushed = True

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


# Process-level cache: one LMDB env per lmdb_path, kept open for the
# lifetime of the worker process (avoids repeated open/close overhead).
_norm_lmdb_cache: dict[str, lmdb.Environment] = {}
# db_paths for which word_forms.lmdb is absent (no lemma/attr flat files)
_no_forms_lmdb: set[str] = set()

# Flat files (in frequencies/) that feed word_forms.lmdb
_FORMS_FLAT_FILES = ("lemmas", "word_attributes", "lemma_word_attributes")


def get_lmdb_env(lmdb_path: str) -> lmdb.Environment:
    """Return (and cache) a read-only LMDB environment for the given path."""
    env = _norm_lmdb_cache.get(lmdb_path)
    if env is not None:
        return env
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, max_spare_txns=4)
    _norm_lmdb_cache[lmdb_path] = env
    return env


def _get_norm_env(freq_file: str) -> lmdb.Environment:
    """Return (and cache) the norm_word.lmdb env, building it from the freq file if absent."""
    lmdb_path = freq_file + ".lmdb"
    if lmdb_path not in _norm_lmdb_cache and not os.path.exists(lmdb_path):
        _build_norm_lmdb(freq_file, lmdb_path)
    return get_lmdb_env(lmdb_path)


def _build_norm_lmdb(freq_file: str, lmdb_path: str) -> None:
    """Build norm→originals LMDB on first use (for databases pre-dating this feature)."""
    from collections import defaultdict

    tmp_path = lmdb_path + ".tmp"
    mapping: dict[bytes, list[bytes]] = defaultdict(list)
    with open(freq_file, "rb") as f:
        for line in f:
            tab = line.find(b"\t")
            if tab < 0:
                continue
            norm = line[:tab]
            orig = line[tab + 1:].rstrip(b"\n")
            if norm:
                mapping[norm].append(orig)

    tmp_env = lmdb.open(tmp_path, map_size=2 * 1024 * 1024 * 1024,
                        writemap=True, sync=False, metasync=False)
    with tmp_env.begin(write=True) as txn:
        for norm, originals in mapping.items():
            txn.put(norm, b"\x00".join(originals))
    tmp_env.sync(True)
    os.makedirs(lmdb_path, exist_ok=True)
    tmp_env.copy(lmdb_path, compact=True)
    tmp_env.close()
    os.system(f"rm -rf {tmp_path}")


def _build_forms_lmdb(db_path: str, lmdb_path: str) -> None:
    """Build key-only LMDB from lemma/attr flat files (lemmas, word_attributes, lemma_word_attributes).

    Keys are the form strings (e.g. b'lemma:love', b'love:pos:NOUN'); values are empty.
    Used for fast prefix/regex scanning without touching the large words.lmdb.
    """
    freq_dir = os.path.join(db_path, "frequencies")
    tmp_path = lmdb_path + ".tmp"
    tmp_env = lmdb.open(tmp_path, map_size=512 * 1024 * 1024,
                        writemap=True, sync=False, metasync=False)
    with tmp_env.begin(write=True) as txn:
        for fname in _FORMS_FLAT_FILES:
            fpath = os.path.join(freq_dir, fname)
            if not os.path.exists(fpath):
                continue
            with open(fpath, "rb") as f:
                for line in f:
                    key = line.rstrip(b"\n")
                    if key:
                        txn.put(key, b"")
    tmp_env.sync(True)
    os.makedirs(lmdb_path, exist_ok=True)
    tmp_env.copy(lmdb_path, compact=True)
    tmp_env.close()
    os.system(f"rm -rf {tmp_path}")


def _get_forms_env(db_path: str) -> lmdb.Environment | None:
    """Return (and cache) the word_forms.lmdb env, building it lazily if absent.

    Returns None if the database has no lemma/attr flat files (plain word-only corpus).
    """
    lmdb_path = os.path.join(db_path, "frequencies", "word_forms.lmdb")
    if lmdb_path in _norm_lmdb_cache:
        return _norm_lmdb_cache[lmdb_path]
    if db_path in _no_forms_lmdb:
        return None
    if not os.path.exists(lmdb_path):
        freq_dir = os.path.join(db_path, "frequencies")
        has_files = any(os.path.exists(os.path.join(freq_dir, f)) for f in _FORMS_FLAT_FILES)
        if not has_files:
            _no_forms_lmdb.add(db_path)
            return None
        _build_forms_lmdb(db_path, lmdb_path)
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
