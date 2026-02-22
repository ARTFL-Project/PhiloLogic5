"""Get similar collocate distributions using excess-over-median cosine similarity
with top shared collocates as explanations."""

import os

import numba
import numpy as np

from philologic.runtime.reports.collocation import (
    load_map_field_cache,
    safe_pickle_load,
)

@numba.njit(cache=True)
def _nb_cosine_similarities(ref_tids, ref_weights, all_tids, all_weights, group_bounds, n_groups, max_tid):
    """Cosine similarity between a reference vector and each group's vector."""
    ref_lookup = np.zeros(max_tid + 1, dtype=np.float64)
    ref_norm_sq = 0.0
    for i in range(len(ref_tids)):
        ref_lookup[ref_tids[i]] = float(ref_weights[i])
        ref_norm_sq += float(ref_weights[i]) ** 2
    ref_norm = np.sqrt(ref_norm_sq)

    similarities = np.empty(n_groups, dtype=np.float64)
    for g in range(n_groups):
        gs = group_bounds[g]
        ge = group_bounds[g + 1]
        dot = 0.0
        norm_sq = 0.0
        for j in range(gs, ge):
            c = float(all_weights[j])
            norm_sq += c * c
            dot += ref_lookup[all_tids[j]] * c
        g_norm = np.sqrt(norm_sq)
        if ref_norm > 0.0 and g_norm > 0.0:
            similarities[g] = dot / (ref_norm * g_norm)
        else:
            similarities[g] = 0.0

    for i in range(len(ref_tids)):
        ref_lookup[ref_tids[i]] = 0.0

    return similarities


def get_similar_collocate_distributions(request, config):
    """Get similar collocate distributions"""
    # Load the map_field numpy cache (all groups)
    tids, counts, group_bounds, group_names, count_lemmas, attribute, attribute_value = load_map_field_cache(
        request.file_path
    )

    # Load reference collocates (Counter from a previous collocation query)
    if not request.primary_file_path:
        return {"similar": []}
    reference_collocates = safe_pickle_load(request.primary_file_path)

    # Build name<->tid lookups (use lemma vocab when counting lemmas)
    colloc_dir = os.path.join(config.db_path, "data", "collocations")
    if count_lemmas:
        count_offsets = np.load(os.path.join(colloc_dir, "attr_lemma_vocab_offsets.npy"), mmap_mode="r")
        with open(os.path.join(colloc_dir, "attr_lemma_vocab_strings.bin"), "rb") as f:
            vocab_data = f.read()
    else:
        count_offsets = np.load(os.path.join(colloc_dir, "vocab_offsets.npy"), mmap_mode="r")
        with open(os.path.join(colloc_dir, "vocab_strings.bin"), "rb") as f:
            vocab_data = f.read()
    used_tids = np.unique(tids)
    name_to_tid = {}
    tid_to_name = {}
    for tid in used_tids.tolist():
        s = int(count_offsets[tid])
        e = int(count_offsets[tid + 1])
        name = vocab_data[s:e].decode("utf-8")
        name_to_tid[name] = tid
        tid_to_name[tid] = name

    # Convert reference Counter to sparse arrays.
    # Reference Counter keys may have an attribute suffix (e.g. ":pos:noun")
    # that the vocab strings don't include -- strip it before looking up tids.
    attr_suffix = f":{attribute}:{attribute_value}" if attribute else ""
    ref_tids_list = []
    ref_counts_list = []
    for word, count in reference_collocates.items():
        key = word.removesuffix(attr_suffix) if attr_suffix else word
        tid = name_to_tid.get(key)
        if tid is not None:
            ref_tids_list.append(tid)
            ref_counts_list.append(count)

    if not ref_tids_list:
        return {"similar": []}

    ref_tids = np.array(ref_tids_list, dtype=np.uint32)
    ref_counts_arr = np.array(ref_counts_list, dtype=np.int32)
    max_tid = max(int(tids.max()), int(ref_tids.max()))
    n_groups = len(group_names)
    counts_f64 = counts.astype(np.float64)
    ref_f64 = ref_counts_arr.astype(np.float64)

    # --- Excess-over-median cosine ---
    # Per-tid median count across all groups (tids with df <= n_groups/2 have median 0)
    tids_i64 = tids.astype(np.int64)
    df = np.zeros(max_tid + 1, dtype=np.int32)
    np.add.at(df, tids_i64, 1)

    median_per_tid = np.zeros(max_tid + 1, dtype=np.float64)
    high_df_tids = np.where(df > n_groups // 2)[0]
    if len(high_df_tids) > 0:
        tid_order = np.argsort(tids)
        sorted_tids = tids_i64[tid_order]
        sorted_counts = counts_f64[tid_order]
        unique_tids, starts = np.unique(sorted_tids, return_index=True)
        ends = np.append(starts[1:], len(sorted_tids))
        high_set = set(high_df_tids.tolist())
        for i in range(len(unique_tids)):
            t = int(unique_tids[i])
            if t not in high_set:
                continue
            vals = np.sort(sorted_counts[starts[i]:ends[i]])
            n_zeros = n_groups - len(vals)
            if n_zeros > 0:
                vals = np.concatenate([np.zeros(n_zeros, dtype=np.float64), vals])
            median_per_tid[t] = np.median(vals)

    log_median = np.log1p(median_per_tid)

    # Excess: how much each group exceeds corpus median (log scale, clamped to 0)
    excess_all = np.maximum(0.0, np.log1p(counts_f64) - log_median[tids_i64])
    excess_ref = np.maximum(0.0, np.log1p(ref_f64) - log_median[ref_tids.astype(np.int64)])

    sims = _nb_cosine_similarities(
        ref_tids, excess_ref, tids, excess_all, group_bounds, n_groups, max_tid,
    )

    # --- Collect top-50 with shared collocate explanations ---
    ref_excess_lookup = np.zeros(max_tid + 1, dtype=np.float64)
    ref_excess_lookup[ref_tids] = excess_ref

    top_indices = _top_similar_indices(sims, 50)
    similar = []
    for idx in top_indices:
        gs, ge = int(group_bounds[idx]), int(group_bounds[idx + 1])
        words = _top_shared_collocates(
            ref_excess_lookup,
            tids[gs:ge], excess_all[gs:ge],
            tid_to_name,
        )
        similar.append(
            (group_names[idx], round(float(sims[idx]), 3), words)
        )

    return {"similar": similar}


def _top_similar_indices(similarities, n):
    """Return indices of top-n groups by similarity, skipping self-matches."""
    order = np.argsort(similarities)[::-1]
    result = []
    for idx in order:
        if float(similarities[idx]) < 0.99:
            result.append(int(idx))
            if len(result) == n:
                break
    return result


def _top_shared_collocates(ref_excess_lookup, group_tids, group_excess, tid_to_name, n_words=20):
    """Return the top shared collocates ranked by min excess-over-median.

    For each word both the reference and the group use above the corpus
    median, rank by min(excess_ref, excess_group) -- the bottleneck value.
    """
    shared = []
    for i in range(len(group_tids)):
        tid = int(group_tids[i])
        ref_w = ref_excess_lookup[tid]
        g_w = float(group_excess[i])
        if ref_w > 0 and g_w > 0:
            shared.append((tid, min(ref_w, g_w)))

    if not shared:
        return []

    shared.sort(key=lambda x: x[1], reverse=True)
    return [tid_to_name.get(tid, "") for tid, _ in shared[:n_words]]
