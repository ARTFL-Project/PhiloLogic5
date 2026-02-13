#!/var/lib/philologic5/philologic_env/bin/python3

import hashlib
import os
import subprocess
import sys
import threading
from pathlib import Path

# Set umask before importing numba to ensure cache files are world-writable
os.umask(0o000)

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


# =============================================================================
# Document-level search helper functions
# =============================================================================


@numba.jit(nopython=True, cache=True)
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


@numba.jit(nopython=True, cache=True)
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


@numba.jit(nopython=True, cache=True)
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


@numba.jit(nopython=True, cache=True)
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

    from bisect import bisect_left, bisect_right

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
            merged = _load_word_group_hits(db_path, txn, words, overflow_words)
            if corpus_file is not None:
                merged = filter_philo_ids(corpus_file, merged)
            output_file.write(merged.tobytes())
    env.close()


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
        # Load all word group hits (zero-copy from LMDB)
        all_hits = []
        for group in word_groups:
            hits = _load_word_group_hits(db_path, txn, group, overflow_words)
            all_hits.append(hits)

        # Apply corpus filter if provided
        if corpus_file is not None:
            all_hits = [filter_philo_ids(corpus_file, hits) for hits in all_hits]

        # Early return if any group has 0 hits
        if any(len(h) == 0 for h in all_hits):
            env.close()
            return

        # Find the rarest word group (smallest hit count)
        sizes = [len(h) for h in all_hits]
        rare_group_idx = int(np.argmin(np.array(sizes)))

        # Scan only the rare word for document boundaries
        rare_docs, rare_starts = _find_doc_boundaries(all_hits[rare_group_idx])

        # Bulk searchsorted for other groups: find per-document ranges
        lefts = {}
        rights = {}
        for g in range(n_groups):
            if g == rare_group_idx:
                continue
            g_doc_ids = all_hits[g][:, 0]
            lefts[g] = np.searchsorted(g_doc_ids, rare_docs, side='left')
            rights[g] = np.searchsorted(g_doc_ids, rare_docs, side='right')

        with open(hitlist_filename, "wb") as output_file:
            for d_idx in range(len(rare_docs)):
                # Get rare group's hits for this document
                rare_doc_hits = all_hits[rare_group_idx][rare_starts[d_idx]:rare_starts[d_idx + 1]]

                # Get other groups' hits for this document, check if any are empty
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

                # Concatenate all groups' doc hits into a single array for the Numba kernel
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

                # Call Numba kernel - rare_hits within all_doc_hits starts at group_offsets[rare_group_idx]
                rare_in_concat = all_doc_hits[group_offsets[rare_group_idx]:
                                              group_offsets[rare_group_idx] + group_sizes_arr[rare_group_idx]]
                result = _phrase_match_doc(
                    rare_in_concat, all_doc_hits, group_offsets, group_sizes_arr,
                    n_groups, rare_group_idx, out_cols
                )

                if len(result) > 0:
                    output_file.write(result.tobytes())

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
        # Load word hits as views (no copy - uses LMDB's mmap)
        all_hits = []
        for group in word_groups:
            hits = _load_word_group_hits(db_path, txn, group, overflow_words)
            all_hits.append(hits)

        # Apply corpus filter if provided (this creates copies for filtered results)
        if corpus_file is not None:
            all_hits = [filter_philo_ids(corpus_file, hits) for hits in all_hits]

        # Scan only the rarest group for doc boundaries, use bulk searchsorted for others
        sizes = [len(h) for h in all_hits]
        rare_idx = int(np.argmin(np.array(sizes)))
        rare_docs, rare_starts = _find_doc_boundaries(all_hits[rare_idx])

        # Pre-compute per-document ranges for all non-rare groups
        other_lefts = {}
        other_rights = {}
        for g in range(n_groups):
            if g == rare_idx:
                continue
            g_doc_ids = all_hits[g][:, 0]
            other_lefts[g] = np.searchsorted(g_doc_ids, rare_docs, side='left')
            other_rights[g] = np.searchsorted(g_doc_ids, rare_docs, side='right')

        with open(hitlist_filename, "wb") as output_file:
            for d_idx in range(len(rare_docs)):
                # Extract hits for this document from each group
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

                # Process based on number of groups
                if n_groups == 2:
                    # Optimized 2-group: binary search sentences directly
                    result = _cooc_match_doc_two_groups(
                        doc_hits[0], doc_hits[1], cooc_slice,
                        cooc_order, max_distance, exact_distance
                    )

                    if len(result) > 0:
                        output_file.write(result.tobytes())
                else:
                    # General N-group case: use sentence intersection pipeline
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
