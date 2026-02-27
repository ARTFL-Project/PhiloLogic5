#!/var/lib/philologic5/philologic_env/bin/python3
"""Multi-word search: phrase, co-occurrence, and proximity queries.

Contains all Numba-jitted kernels and the search entry points for queries
involving two or more word groups (phrase_ordered, phrase_unordered,
proxy, exact_cooc, sentence).
"""

import os

import lmdb
import numba
import numpy as np

# Set Numba cache directory (idempotent — safe even if Query.py set it first)
cache_dir = "/var/lib/philologic5/numba_cache"
if not os.access(cache_dir, os.W_OK):
    cache_dir = f"/tmp/philologic_numba_cache_{os.getuid()}"
    os.makedirs(cache_dir, mode=0o755, exist_ok=True)
os.environ["NUMBA_CACHE_DIR"] = cache_dir
numba.config.CACHE_DIR = cache_dir

from philologic.runtime.Query import (
    _find_doc_boundaries,
    _kway_merge_sorted_arrays,
    _load_word_arrays,
    _load_word_group_hits,
    _write_with_early_flush,
    filter_philo_ids,
    get_word_groups,
    search_word,
)


# ── Numba kernels ─────────────────────────────────────────────────────────────

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
    per-document dispatch overhead (~22us x N_docs).

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


# ── Python search entry points ────────────────────────────────────────────────

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
