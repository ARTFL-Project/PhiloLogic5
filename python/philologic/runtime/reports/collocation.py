#!/var/lib/philologic5/philologic_env/bin/python3
"""Collocation results"""

import hashlib
import json
import os
import pickle
import struct
import tempfile
import time
import timeit
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numba
import numpy as np

from philologic.runtime.DB import DB
from philologic.runtime.exceptions import BadRequest, NotFound
from philologic.runtime.hitlist_dir import get_hitlist_dir
from philologic.runtime.MetadataQuery import bulk_load_metadata
from philologic.runtime.Query import MAX_DISTANCE, get_word_groups, phrase_lengths, rewrite_terms_file, split_terms
from philologic.runtime.QuerySyntax import group_terms, parse_query
from philologic.runtime.sql_validation import validate_request_column
from philologic.runtime.word_properties import (
    _nb_compare_keys,
    _nb_sentence_index,
    _nb_word_kept,
    property_masks,
    property_names,
    split_properties,
)

# Per-worker cache of corpus-wide sentence document-frequency arrays.
# Keyed by (db_path, count_lemmas) -> (df_array, n_sentences).
_SENTENCE_DF_CACHE: dict = {}

# Default and bounds for the document-frequency stopword filter (percent of
# sentences a word must exceed to be dropped).
DEFAULT_DF_PCT = 1.0
MIN_DF_PCT = 0.5
MAX_DF_PCT = 5.0


@numba.njit(parallel=True, cache=True)
def _nb_build_filter(vocab_hashes, filter_hashes_sorted):
    """Build boolean filter array by matching deterministic hashes."""
    n = len(vocab_hashes)
    nf = len(filter_hashes_sorted)
    is_filtered = np.zeros(n, dtype=numba.boolean)
    for i in numba.prange(n):
        h = vocab_hashes[i]
        lo, hi = 0, nf
        while lo < hi:
            mid = (lo + hi) // 2
            if filter_hashes_sorted[mid] < h:
                lo = mid + 1
            else:
                hi = mid
        if lo < nf and filter_hashes_sorted[lo] == h:
            is_filtered[i] = True
    return is_filtered


@numba.njit(parallel=True, cache=True)
def _nb_searchsorted(sent_keys, hit_keys):
    """Parallel binary search of hit keys against sorted sentence keys."""
    n_hits = hit_keys.shape[0]
    n_sents = sent_keys.shape[0]
    result = np.empty(n_hits, dtype=np.int64)
    match = np.empty(n_hits, dtype=numba.boolean)
    for i in numba.prange(n_hits):
        lo = _nb_sentence_index(sent_keys, hit_keys[i])
        result[i] = lo
        match[i] = lo < n_sents and _nb_compare_keys(sent_keys[lo], hit_keys[i]) == 0
    return result, match


@numba.njit(cache=True)
def _nb_span_with(pos, q_pos, q_end):
    """The distance from the first to the last of a hit's query words (at q_pos to q_end) and the word at pos. That word
    is within n words of the hit when it is at most n, as the words found by a search within n words are."""
    first = q_pos if q_pos < pos else pos
    last = q_end if q_end > pos else pos
    return np.int32(last) - np.int32(first)


@numba.njit(parallel=True, cache=True)
def _nb_fused_count(sent_indices, hit_q_pos, sent_offsets, token_ids, is_filtered, vocab_size):
    """Fused gather + filter + count in a single parallel pass (no intermediate arrays)."""
    n = len(sent_indices)
    nt = numba.config.NUMBA_NUM_THREADS
    local_counts = np.zeros((nt, vocab_size), dtype=np.int64)
    for i in numba.prange(n):
        tid = numba.get_thread_id()
        start = sent_offsets[sent_indices[i]]
        end = sent_offsets[sent_indices[i] + 1]
        q_pos = hit_q_pos[i]
        pos = np.uint32(1)
        for j in range(start, end):
            if pos != q_pos:
                token = token_ids[j]
                if not is_filtered[token]:
                    local_counts[tid, token] += 1
            pos += 1
    return local_counts


@numba.njit(parallel=True, cache=True)
def _nb_fused_count_distance(sent_indices, hit_q_pos, hit_q_end, sent_offsets, token_ids, is_filtered, vocab_size,
                             max_dist):
    """Fused count with distance constraint (see _nb_span_with: the hit's query words are at hit_q_pos to hit_q_end)."""
    n = len(sent_indices)
    nt = numba.config.NUMBA_NUM_THREADS
    local_counts = np.zeros((nt, vocab_size), dtype=np.int64)
    for i in numba.prange(n):
        tid = numba.get_thread_id()
        start = sent_offsets[sent_indices[i]]
        end = sent_offsets[sent_indices[i] + 1]
        q_pos = hit_q_pos[i]
        q_end = hit_q_end[i]
        pos = np.uint32(1)
        for j in range(start, end):
            if pos != q_pos:
                if _nb_span_with(pos, q_pos, q_end) <= max_dist:
                    token = token_ids[j]
                    if not is_filtered[token]:
                        local_counts[tid, token] += 1
            pos += 1
    return local_counts


@numba.njit(cache=True)
def _nb_other_kept(word, other_ids, other_excluded):
    """Whether the word at index word of the collocation arrays is not of other_excluded in the other vocabulary:
    always, without one (None). An "is None" test of its own, which numba takes out when compiling for None."""
    if other_ids is not None:
        return not other_excluded[other_ids[word]]
    return True


@numba.njit(parallel=True, cache=True)
def _nb_fused_count_filtered(
    sent_indices,
    hit_q_pos,
    hit_q_end,
    sent_offsets,
    count_ids,
    is_filtered,
    vocab_size,
    max_dist,
    prop_columns,
    prop_allowed,
    other_ids,
    other_excluded,
):
    """_nb_fused_count with the words filtered further: within max_dist of the hit when max_dist >= 0 (see
    _nb_span_with), with a kept value of each word property of prop_columns (see word_properties.property_masks;
    None for any), and not of other_excluded in the other vocabulary: other_ids has the lemma of each word when word
    forms are counted, its form when lemmas are (None to leave none out)."""
    n = len(sent_indices)
    nt = numba.config.NUMBA_NUM_THREADS
    local_counts = np.zeros((nt, vocab_size), dtype=np.int64)
    for i in numba.prange(n):
        tid = numba.get_thread_id()
        start = sent_offsets[sent_indices[i]]
        end = sent_offsets[sent_indices[i] + 1]
        q_pos = hit_q_pos[i]
        q_end = hit_q_end[i]
        pos = np.uint32(1)
        for j in range(start, end):
            if pos != q_pos and (max_dist < 0 or _nb_span_with(pos, q_pos, q_end) <= max_dist):
                token = count_ids[j]
                if not is_filtered[token] and _nb_word_kept(j, prop_columns, prop_allowed):
                    if _nb_other_kept(j, other_ids, other_excluded):
                        local_counts[tid, token] += 1
            pos += 1
    return local_counts


@numba.njit(parallel=True, cache=True)
def _nb_gather_grouped(
    sent_indices,
    hit_q_pos,
    hit_q_end,
    hit_gids,
    sent_offsets,
    count_ids,
    is_filtered,
    prop_columns,
    prop_allowed,
    other_ids,
    other_excluded,
    max_dist,
    hit_word_offsets,
    out_tokens,
    out_gids,
    out_valid,
):
    """Single-pass gather: annotate every word with token, group, validity.

    One kernel call for ALL hits (all groups combined). Produces flat arrays
    that are then split by group in numpy for per-group bincount. The words are filtered as by
    _nb_fused_count_filtered.
    """
    n = len(sent_indices)
    for i in numba.prange(n):
        si = sent_indices[i]
        s = numba.int64(sent_offsets[si])
        e = numba.int64(sent_offsets[si + 1])
        base = hit_word_offsets[i]
        qp = hit_q_pos[i]
        qe = hit_q_end[i]
        gid = hit_gids[i]
        pos = np.uint32(1)
        for j in range(s, e):
            w = base + (j - s)
            t = count_ids[j]
            out_tokens[w] = t
            out_gids[w] = gid
            ok = pos != qp
            if ok:
                ok = not is_filtered[t]
            if ok and max_dist >= np.int32(0):
                ok = _nb_span_with(pos, qp, qe) <= max_dist
            if ok:
                ok = _nb_word_kept(j, prop_columns, prop_allowed)
            if ok:
                ok = _nb_other_kept(j, other_ids, other_excluded)
            out_valid[w] = ok
            pos += 1


@numba.njit(cache=True)
def _nb_grouped_bincount(vg_sorted, vt_sorted, group_bounds, n_groups, max_tid):
    """Per-group bincount using a reusable counts buffer with dirty-list tracking.

    Expects vg_sorted/vt_sorted to be sorted by group_id. Returns flat arrays
    (out_g, out_t, out_c) of (group_id, token_id, count) triples, ordered by
    group_id then token_id within each group.
    """
    counts_buf = np.zeros(max_tid + 1, dtype=np.int32)
    dirty = np.empty(max_tid + 1, dtype=np.int64)

    # First pass: count total unique entries to size output arrays
    total_unique = 0
    for g in range(n_groups):
        gs = group_bounds[g]
        ge = group_bounds[g + 1]
        if gs >= ge:
            continue
        n_dirty = 0
        for j in range(gs, ge):
            t = vt_sorted[j]
            if counts_buf[t] == 0:
                dirty[n_dirty] = t
                n_dirty += 1
            counts_buf[t] += 1
        total_unique += n_dirty
        for d in range(n_dirty):
            counts_buf[dirty[d]] = 0

    # Second pass: fill output arrays
    out_g = np.empty(total_unique, dtype=np.int32)
    out_t = np.empty(total_unique, dtype=np.int64)
    out_c = np.empty(total_unique, dtype=np.int32)
    pos = 0
    for g in range(n_groups):
        gs = group_bounds[g]
        ge = group_bounds[g + 1]
        if gs >= ge:
            continue
        n_dirty = 0
        for j in range(gs, ge):
            t = vt_sorted[j]
            if counts_buf[t] == 0:
                dirty[n_dirty] = t
                n_dirty += 1
            counts_buf[t] += 1
        for d in range(n_dirty):
            idx = dirty[d]
            out_g[pos] = g
            out_t[pos] = idx
            out_c[pos] = counts_buf[idx]
            pos += 1
            counts_buf[idx] = 0

    return out_g, out_t, out_c


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only allows safe classes for collocation data."""

    ALLOWED_CLASSES = {
        ("collections", "Counter"),
    }

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED_CLASSES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Forbidden class: {module}.{name}")


def cache_file_path(db_path, file_path, ext):
    """The path of the collocation cache (a .pickle or .npz file, as ext) a client names by file_path, the path it
    was sent for it. Only the file name is taken from the client, to look for in the database's hitlist_dir, where
    collocation caches are. Raises NotFound if it is not there: the hitlist cleanup removes caches after a while."""
    name = os.path.basename(file_path or "")
    if not name.endswith(ext):
        raise BadRequest(f"Not a collocation cache: {name}")
    path = os.path.join(get_hitlist_dir(os.path.join(db_path, "data")), name)
    if not os.path.isfile(path):
        raise NotFound("These collocation results have expired: run the collocation again.")
    return path


def safe_pickle_load(file_path):
    """Load a pickle file with restricted classes."""
    with open(file_path, "rb") as f:
        return RestrictedUnpickler(f).load()


def fightin_words_zscores(y_a, y_b):
    """Log-proportion ratio with informative Dirichlet prior (Monroe et al. 2008).

    Returns per-word z-scores: positive means over-represented in y_a relative to y_b.
    """
    y_a = np.asarray(y_a, dtype=np.float64)
    y_b = np.asarray(y_b, dtype=np.float64)
    n_a = y_a.sum()
    n_b = y_b.sum()
    alpha_0 = n_a + n_b
    y_pooled = y_a + y_b
    alpha = alpha_0 * (y_pooled / y_pooled.sum())
    delta = np.log((y_a + alpha) / (n_a + alpha_0)) - np.log((y_b + alpha) / (n_b + alpha_0))
    sigma_sq = 1.0 / (y_a + alpha) + 1.0 / (y_b + alpha)
    return delta / np.sqrt(sigma_sq)


def fightin_words_zscores_vs_rest(y_a, y_b, n_a, n_b):
    """Closed-form fightin' words for the 'this slice vs. rest of corpus' case.

    When y_b is the rest-of-corpus on the same support as y_a (i.e., n_a + n_b
    is the corpus total), alpha_0 = n_a + n_b and alpha[i] simplifies to
    y_a[i] + y_b[i], so the prior pre-computation drops out. Faster than the
    general form when this assumption holds (single-group / single-hit
    "distinctiveness" scoring).

    y_a, y_b: 1D float arrays of equal length on the active vocab support.
    n_a, n_b: scalar totals (corpus_total = n_a + n_b).
    Returns per-token z-scores; positive means over-represented in y_a.
    """
    delta = (
        np.log((2.0 * y_a + y_b) / (2.0 * n_a + n_b))
        - np.log((y_a + 2.0 * y_b) / (n_a + 2.0 * n_b))
    )
    sigma_sq = 1.0 / (2.0 * y_a + y_b) + 1.0 / (y_a + 2.0 * y_b)
    return delta / np.sqrt(sigma_sq)


def score_with_top_k_positive(z, candidate_tids, top_k, min_distinctive=1):
    """Collapse a z-score vector to a single distinctiveness score + explainers.

    Take the tokens with z > 0, keep the top-k by z, sum their z-scores. The
    selected token ids are returned as explainers — the actual evidence behind
    the score.

    Returns (score, explainer_tids):
      - score: float, sum of top-k positive z-scores, or -np.inf if fewer than
        min_distinctive tokens have positive z (caller can use this as a skip).
      - explainer_tids: int64 array of the corresponding token ids, in
        z-descending order, length min(top_k, n_positive).
    """
    pos_mask = z > 0
    n_positive = int(pos_mask.sum())
    if n_positive < min_distinctive:
        return -np.inf, np.empty(0, dtype=np.int64)
    z_pos = z[pos_mask]
    t_pos = candidate_tids[pos_mask]
    order = np.argsort(z_pos)[::-1][:top_k]
    return float(z_pos[order].sum()), t_pos[order].astype(np.int64)


def _vectorized_collocation(
    db_path,
    hits,
    filter_list,
    count_lemmas,
    collocate_distance,
    map_field_info=None,
    properties=(),
    other_list=(),
):
    """Fully vectorized collocation counting using columnar numpy arrays.

    Returns either a Counter (when map_field_info is None) or
    a dict {metadata_value: Counter} (when map_field_info is set).
    Collocates are counted around each hit, so that each count is the number of hits a search for the query and the
    collocate finds (in the sentence or within collocate_distance words). The words of filter_list are not counted,
    nor those without a kept value of each word property of properties ((name, values) pairs), nor those of
    other_list in the other vocabulary (lemmas when word forms are counted, word forms when lemmas are).
    """
    colloc_dir = os.path.join(db_path, "collocations")

    # Memory-map arrays (zero-copy, OS manages page caching)
    token_ids_mmap = np.load(os.path.join(colloc_dir, "token_ids.npy"), mmap_mode="r")
    sent_offsets = np.load(os.path.join(colloc_dir, "sent_offsets.npy"), mmap_mode="r")
    sent_keys_native = np.load(os.path.join(colloc_dir, "sent_keys_native.npy"), mmap_mode="r")
    sent_offsets_arr = np.asarray(sent_offsets)

    # Bulk read hitlist
    hits.finish()

    with hits.open_raw() as f:
        raw = f.read()

    all_hits = np.frombuffer(raw, dtype=np.uint32).reshape(-1, hits.length)

    # Where the first and last of each hit's query words are in its sentence (a hit has the position of each in
    # columns 7, 9...), for collocate_distance
    positions = all_hits[:, 7::2]
    span_start = positions.min(axis=1) if len(all_hits) else np.empty(0, dtype=np.uint32)
    span_end = positions.max(axis=1) if len(all_hits) else np.empty(0, dtype=np.uint32)

    #  Load counting vocab: lemma IDs or token IDs
    if count_lemmas:
        count_ids = np.load(os.path.join(colloc_dir, "attr_lemma_ids.npy"), mmap_mode="r")
        count_hashes = np.load(os.path.join(colloc_dir, "attr_lemma_vocab_hashes.npy"), mmap_mode="r")
        count_offsets = np.load(os.path.join(colloc_dir, "attr_lemma_vocab_offsets.npy"), mmap_mode="r")
        with open(os.path.join(colloc_dir, "attr_lemma_vocab_strings.bin"), "rb") as f:
            count_data = f.read()
    else:
        count_ids = token_ids_mmap
        count_hashes = np.load(os.path.join(colloc_dir, "vocab_hashes.npy"), mmap_mode="r")
        count_offsets = np.load(os.path.join(colloc_dir, "vocab_offsets.npy"), mmap_mode="r")
        with open(os.path.join(colloc_dir, "vocab_strings.bin"), "rb") as f:
            count_data = f.read()

    count_vocab_size = len(count_hashes)

    #  Build identity filter (always — applies to both attribute and non-attribute paths)
    is_filtered = _nb_build_filter(count_hashes, _filter_hashes(filter_list))

    #  Word property arrays, and the query's words of the other vocabulary, if any
    prop_columns, prop_allowed = property_masks(db_path, properties)
    other_ids, other_excluded = _other_vocabulary_filter(colloc_dir, count_lemmas, other_list)
    filtered = prop_columns is not None or other_ids is not None

    #  Parallel binary search (all hits at once)
    hit_keys = np.ascontiguousarray(all_hits[:, :6])
    si, match = _nb_searchsorted(sent_keys_native, hit_keys)
    sent_indices = si[match]
    hit_q_pos = np.ascontiguousarray(span_start[match])
    hit_q_end = np.ascontiguousarray(span_end[match])

    #  Helpers: fused count + decode (shared by all modes)
    def _fused_count(group_si, group_qp, group_qe):
        if filtered:
            return _nb_fused_count_filtered(
                group_si,
                group_qp,
                group_qe,
                sent_offsets_arr,
                count_ids,
                is_filtered,
                count_vocab_size,
                np.int32(-1 if collocate_distance is None else collocate_distance),
                prop_columns,
                prop_allowed,
                other_ids,
                other_excluded,
            )
        if collocate_distance is not None:
            return _nb_fused_count_distance(
                group_si, group_qp, group_qe, sent_offsets_arr,
                count_ids, is_filtered, count_vocab_size, collocate_distance,
            )
        return _nb_fused_count(
            group_si, group_qp, sent_offsets_arr,
            count_ids, is_filtered, count_vocab_size,
        )

    def _decode_counts(local_counts):
        counts = local_counts.sum(axis=0)
        nonzero = np.nonzero(counts)[0]
        starts_list = count_offsets[nonzero].tolist()
        ends_list = count_offsets[nonzero + 1].tolist()
        nz_names = [count_data[s:e].decode("utf-8") for s, e in zip(starts_list, ends_list)]
        nz_counts = counts[nonzero].tolist()
        return Counter(dict(zip(nz_names, nz_counts)))

    #  Simple mode: one fused count over all hits
    if map_field_info is None:
        return _decode_counts(_fused_count(sent_indices, hit_q_pos, hit_q_end))

    #  map_field mode: single gather pass + per-group bincount
    metadata_cache, field_obj_index = map_field_info
    hit_prefixes = all_hits[match, :field_obj_index]

    # Assign integer group_ids to each hit
    group_names = []  # group_id -> metadata_value
    group_lookup = {}  # metadata_value -> group_id
    hit_gids = np.empty(len(sent_indices), dtype=np.int32)
    for i in range(len(sent_indices)):
        prefix = tuple(hit_prefixes[i].tolist())
        mv = metadata_cache.get(prefix)
        if mv not in (None, ""):  # hits with no value make no group: it would have no name to compare it by
            if mv not in group_lookup:
                group_lookup[mv] = len(group_names)
                group_names.append(mv)
            hit_gids[i] = group_lookup[mv]
        else:
            hit_gids[i] = -1

    # Filter to hits that have a valid group
    valid_mask = hit_gids >= 0
    v_si = np.ascontiguousarray(sent_indices[valid_mask])
    v_qp = np.ascontiguousarray(hit_q_pos[valid_mask])
    v_qe = np.ascontiguousarray(hit_q_end[valid_mask])
    v_gids = np.ascontiguousarray(hit_gids[valid_mask])

    # Pre-compute per-hit word ranges
    lengths = (sent_offsets_arr[v_si + 1] - sent_offsets_arr[v_si]).astype(np.int64)
    total_words = int(lengths.sum())
    hit_word_offsets = np.empty(len(v_si) + 1, dtype=np.int64)
    hit_word_offsets[0] = 0
    np.cumsum(lengths, out=hit_word_offsets[1:])

    # Allocate flat output arrays
    out_tokens = np.empty(total_words, dtype=np.uint32)
    out_gids = np.empty(total_words, dtype=np.int32)
    out_valid = np.empty(total_words, dtype=np.bool_)

    # Single gather kernel call (all hits, all groups)
    _max_dist = np.int32(collocate_distance) if collocate_distance is not None else np.int32(-1)
    _nb_gather_grouped(
        v_si,
        v_qp,
        v_qe,
        v_gids,
        sent_offsets_arr,
        count_ids,
        is_filtered,
        prop_columns,
        prop_allowed,
        other_ids,
        other_excluded,
        _max_dist,
        hit_word_offsets,
        out_tokens,
        out_gids,
        out_valid,
    )

    # Sort by group, then per-group bincount via numba dirty-list kernel
    vt = out_tokens[out_valid]
    vg = out_gids[out_valid]
    order = vg.argsort(kind="stable")
    vg_sorted = vg[order]
    vt_sorted = vt[order]

    n_groups = len(group_names)
    gb = np.searchsorted(vg_sorted, np.arange(n_groups + 1, dtype=np.int32))
    max_tid = int(vt.max()) if len(vt) > 0 else 0
    unique_gids, unique_tids, unique_counts = _nb_grouped_bincount(
        vg_sorted, vt_sorted, gb.astype(np.int64), n_groups, max_tid,
    )

    # Per-group query-word hit counts (used by outlier scoring as a representativeness floor)
    group_hits = np.bincount(v_gids, minlength=n_groups).astype(np.int32)

    # Return raw numpy arrays — string decode is deferred to downstream scripts
    group_bounds = np.searchsorted(unique_gids, np.arange(n_groups + 1, dtype=np.int32))
    # Ensure group_names are strings (metadata values may be int, e.g. year)
    group_names = [str(n) for n in group_names]
    return unique_tids, unique_counts, group_bounds, group_names, group_hits

def _filter_hashes(words):
    """The hashes of words, as the vocabularies have them (vocab_hashes.npy), sorted for _nb_build_filter"""
    return np.array(
        sorted(struct.unpack("<Q", hashlib.md5(w.encode("utf-8")).digest()[:8])[0] for w in words),
        dtype=np.uint64,
    )


def _other_vocabulary_filter(colloc_dir, count_lemmas, other_list):
    """The words of the other vocabulary (lemmas when word forms are counted, word forms when lemmas are) of
    other_list: the id of each word in it and the mask of those of other_list, or (None, None) without any."""
    if not other_list:
        return None, None
    prefix = "" if count_lemmas else "attr_lemma_"
    excluded = _nb_build_filter(
        np.load(os.path.join(colloc_dir, f"{prefix}vocab_hashes.npy"), mmap_mode="r"), _filter_hashes(other_list)
    )
    if not excluded.any():
        return None, None
    ids = np.load(os.path.join(colloc_dir, "token_ids.npy" if count_lemmas else "attr_lemma_ids.npy"), mmap_mode="r")
    return ids, excluded


def collocate_distance(request):
    """The number of words collocates are counted within, or None for the whole sentence. At most MAX_DISTANCE:
    larger ones overflowed the counts' 32-bit integers."""
    try:
        distance = int(request.method_arg)
    except ValueError:
        return None
    if distance < 0:
        raise BadRequest(f"Collocates can't be counted within a negative number of words: {request.method_arg}")
    return min(distance, MAX_DISTANCE)


def phrase_words(q, query_patterns=None):
    """The number of words of the query's phrases after their first: for the distance, a phrase is one word (see
    Query.phrase_lengths)."""
    return sum(length - 1 for length in phrase_lengths(group_terms(parse_query(q, query_patterns=query_patterns))))


def collocation_search_method(q, query_patterns=None, distance=None):
    """The search (method, method_arg) for the hits collocates are counted around, within distance words or in the
    sentence: each occurrence of the query's one term (which may be "a | b"), each occurrence of a quoted phrase, or
    else each co-occurrence of the query terms, unordered, within distance words or in a sentence."""
    grouped = group_terms(parse_query(q, query_patterns=query_patterns))
    if len(split_terms(grouped)) <= 1:
        return "single_term", "0"
    if len(grouped) == 1:  # one term, split: a quoted phrase
        return "phrase_ordered", "0"
    if distance is not None:
        return "proxy_unordered", str(distance)
    return "sentence_unordered", "0"


@dataclass(frozen=True)
class CollocateFilters:
    """How collocates are counted: as lemmas or as word forms, without the common words of common (frequency: the
    words of more sentences than filter_df_pct percent; stopwords: the database's stopwords), and only with a value
    of each word property of properties ((name, values) pairs, by name) among its values."""

    count_lemmas: bool
    common: tuple = ()
    properties: tuple = ()

    def key(self):
        """The filters as a string, for cache names"""
        return json.dumps([self.count_lemmas, self.common, self.properties])


# Request parameters of the values kept of a word property (colloc_attr_pos=NOUN | ADJ): any of them
PROPERTY_PARAM = "colloc_attr_"


def collocate_filters(request, db):
    """The collocate filters of a request. Without colloc_lemmas, collocates are counted as lemmas when the query has
    one, and colloc_filter_choice=attribute is the value q_attribute_value of q_attribute without any other filter:
    links made before the choices were there."""
    if request.colloc_lemmas == "":
        count_lemmas = "lemma:" in request.q
    elif request.colloc_lemmas in ("yes", "no"):
        count_lemmas = request.colloc_lemmas == "yes"
    else:
        raise BadRequest(f"Collocates are counted as lemmas (colloc_lemmas=yes) or not (no): {request.colloc_lemmas}")
    if count_lemmas and not os.path.exists(os.path.join(db.path, "collocations", "attr_lemma_ids.npy")):
        raise BadRequest("This database has no lemmas to count collocates as")
    choices = {choice.strip() for choice in request.colloc_filter_choice.split("|")} - {""} or {"frequency"}
    unknown = choices - {"frequency", "stopwords", "nofilter", "attribute"}
    if unknown:
        raise BadRequest(f"No such collocate filter: {', '.join(sorted(unknown))}")
    properties = {}
    if "attribute" in choices:
        if not request.q_attribute:
            raise BadRequest("Filtering collocates by attribute needs the attribute (q_attribute)")
        properties[request.q_attribute] = {request.q_attribute_value}
        choices = set()
    for key, value in request:
        if key.startswith(PROPERTY_PARAM):
            values = {v.strip() for v in str(value).split("|")} - {""}
            if values:
                properties.setdefault(key[len(PROPERTY_PARAM) :], set()).update(values)
    names = property_names(db.path)
    for name in properties:
        if name not in names:
            raise BadRequest(f"{name!r} is no word property of this database")
    return CollocateFilters(
        count_lemmas,
        tuple(choice for choice in ("frequency", "stopwords") if choice in choices),
        tuple((name, tuple(sorted(values))) for name, values in sorted(properties.items())),
    )


def common_words(request, config, filters):
    """The common words filters leave out, and whether the stopwords list was not found (it is then left out)"""
    words = []
    missing = False
    for choice in filters.common:
        found = build_filter_list(request, config, filters.count_lemmas, choice)
        if found is None:
            missing = True
        else:
            words.extend(found)
    return words, missing


def collocate_keys(collocate, filters):
    """The index keys a search for a collocate is made of, as the filters count it: collocate (a key of the counts:
    lemma:{lemma} for a lemma) with one of the kept values of each word property, if any. Their searches find the
    hits it was counted in."""
    keys = [collocate]
    for name, values in filters.properties:
        keys = [f"{key}:{name}:{value}" for key in keys for value in values]
    return keys


def query_exclusions(query_words, filters, db):
    """The query's words, which are not counted (their other occurrences in the sentence included): the keys of the
    count vocabulary, and those of the other one (lemmas when word forms are counted, word forms when lemmas are).
    A word with properties (aimer:pos:VERB) is left out as the word."""
    names = property_names(db.path)
    words = {split_properties(word, names)[0] for word in query_words}
    other = {word for word in words if word.startswith("lemma:") != filters.count_lemmas}
    return set(query_words) | (words - other), other


def collocation_results(request, config):
    """Fetch collocation results"""
    collocation_object: dict[str, Any] = {"query": dict([i for i in request])}
    db = DB(config.db_path + "/data/")

    map_field = request.map_field or None
    if map_field is not None:
        map_field = validate_request_column(map_field, db)
        if map_field not in db.locals["metadata_fields"]:  # a column, but no metadata (lemma, word_count...)
            raise BadRequest(f"Collocates can only be counted by a metadata field, not {map_field}")
    filters = collocate_filters(request, db)

    if not request.q:  # collocates are counted around search hits: without a query there are none
        collocation_object.update({"filter_list": [], "results_length": 0, "distance": None})
        if map_field is None:
            collocation_object["collocates"] = []
        return collocation_object

    distance = collocate_distance(request)
    method, method_arg = collocation_search_method(request.q, db.locals.query_patterns, distance)
    hits = db.query(
        request.q,
        method,
        method_arg,
        raw_results=True,
        raw_bytes=True,
        **request.metadata,
    )

    # Build list of search terms to filter out.
    # The .terms file lines ARE the count-vocab keys: "lemma:{value}" when
    # counting lemmas, surface forms otherwise. Pass them through verbatim —
    # don't re-prefix or synthesize case variants.
    query_words = []
    terms_file = f"{hits.filename}.terms"
    # .terms is written before the search produces any hit, so once the hitlist is done without it, it is not
    # coming: either no search ran (the metadata matched nothing) or cleanup removed it from a cached search.
    while not os.path.exists(terms_file):
        hits.update()  # which takes the search over if its producer died
        if hits.done:
            break
        time.sleep(0.1)
    if not os.path.exists(terms_file) and len(hits) > 0:
        rewrite_terms_file(db, request.q, hits.filename)
    word_groups = get_word_groups(terms_file) if os.path.exists(terms_file) else []
    for group in word_groups:
        query_words.extend(group)

    # Bare query_words must be in filter_list so the count-side identity hash
    # matches the (un-suffixed) count vocab keys.
    filter_list, other_list = query_exclusions(query_words, filters, db)
    common, stopwords_missing = common_words(request, config, filters)
    if stopwords_missing:  # the stopwords list was not found: it is left out
        collocation_object["stopwords_missing"] = True
    filter_list.update(common)
    collocation_object["filter_list"] = sorted(filter_list | other_list, key=lambda w: (w.lower(), w))

    hits.finish()
    total_hits = len(hits)

    map_field_info = None
    if map_field is not None:
        field_obj_index, metadata_cache = bulk_load_metadata(db, [map_field], inherit=True)[map_field]
        map_field_info = (metadata_cache, field_obj_index)

    if total_hits == 0:
        # Nothing to count. Don't hand the hits over: when the metadata matched nothing, they are
        # the (empty) metadata corpus, whose rows are object ids rather than word hits.
        if map_field is None:
            result = Counter()
        else:
            result = (
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.int32),
                np.zeros(1, dtype=np.int64),
                [],
                np.empty(0, dtype=np.int32),
            )
    else:
        result = _vectorized_collocation(
            db.path,
            hits,
            filter_list,
            filters.count_lemmas,
            # Within distance words of a phrase is of its first or last word, as for the search
            distance + phrase_words(request.q, db.locals.query_patterns) if distance is not None else None,
            map_field_info=map_field_info,
            properties=filters.properties,
            other_list=other_list,
        )

    collocation_object["results_length"] = total_hits
    collocation_object["distance"] = distance

    if map_field is None:
        all_collocates = result
        if None in all_collocates:
            del all_collocates[None]
        # Cache full Counter to disk; return only top 100 for display
        file_path = create_file_path(request, "", db.hitlist_dir, filters=filters)
        atomic_pickle_dump(all_collocates, file_path)
        collocation_object["collocates"] = all_collocates.most_common(100)
        collocation_object["file_path"] = file_path
    else:
        unique_tids, unique_counts, group_bounds, group_names, group_hits = result
        file_path = create_file_path(request, map_field, db.hitlist_dir, ext=".npz", filters=filters)
        save_map_field_cache(
            file_path,
            unique_tids,
            unique_counts,
            group_bounds,
            group_names,
            filters.count_lemmas,
            group_hits=group_hits,
        )
        collocation_object["file_path"] = file_path

    return collocation_object


def collocation_to_csv(collocates):
    """Convert collocation results (list of (word, count) tuples) to CSV string."""
    import csv
    import io

    if not collocates:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["collocate", "count"])
    writer.writeheader()
    for word, count in collocates:
        writer.writerow({"collocate": word, "count": count})
    return output.getvalue()


def atomic_pickle_dump(data, file_path):
    """Write pickle atomically to prevent truncated reads from concurrent requests."""
    dir_path = os.path.dirname(file_path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(data, f)
        os.replace(tmp_path, file_path)
    except BaseException:
        os.unlink(tmp_path)
        raise


def save_map_field_cache(
    file_path, unique_tids, unique_counts, group_bounds, group_names, count_lemmas, group_hits=None
):
    """Save map_field results as numpy arrays (no string decode, no pickle overhead)."""
    dir_path = os.path.dirname(file_path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            arrays = {
                "tids": unique_tids.astype(np.uint32),
                "counts": unique_counts.astype(np.int32),
                "group_bounds": group_bounds.astype(np.int64),
                "group_names": np.frombuffer("\0".join(group_names).encode("utf-8"), dtype=np.uint8),
                "meta": np.array([int(count_lemmas)], dtype=np.uint8),
            }
            if group_hits is not None:
                arrays["group_hits"] = group_hits.astype(np.int32)
            np.savez(f, **arrays)
        os.replace(tmp_path, file_path)
    except BaseException:
        os.unlink(tmp_path)
        raise


def load_map_field_cache(file_path):
    """Load map_field numpy cache. Returns (tids, counts, group_bounds, group_names, count_lemmas)."""
    data = np.load(file_path, allow_pickle=False)
    tids = data["tids"]
    counts = data["counts"]
    group_bounds = data["group_bounds"]
    # Zero groups and a single empty-named group are both stored as b"": group_bounds tells them apart
    group_names = data["group_names"].tobytes().decode("utf-8").split("\0") if len(group_bounds) > 1 else []
    count_lemmas = bool(data["meta"][0])
    return tids, counts, group_bounds, group_names, count_lemmas


def load_group_hits(file_path):
    """Load per-group query-word hit counts from a map_field cache, or None if absent (legacy cache)."""
    data = np.load(file_path, allow_pickle=False)
    if "group_hits" not in data.files:
        return None
    return data["group_hits"]


def decode_group_collocates(tids, counts, group_bounds, group_index, db_path, count_lemmas):
    """Decode string names for a single group from the numpy cache. Returns Counter."""
    colloc_dir = os.path.join(db_path, "data", "collocations")
    if count_lemmas:
        count_offsets = np.load(os.path.join(colloc_dir, "attr_lemma_vocab_offsets.npy"), mmap_mode="r")
        with open(os.path.join(colloc_dir, "attr_lemma_vocab_strings.bin"), "rb") as f:
            count_data = f.read()
    else:
        count_offsets = np.load(os.path.join(colloc_dir, "vocab_offsets.npy"), mmap_mode="r")
        with open(os.path.join(colloc_dir, "vocab_strings.bin"), "rb") as f:
            count_data = f.read()

    gs, ge = int(group_bounds[group_index]), int(group_bounds[group_index + 1])
    tids_g = tids[gs:ge]
    cnts_g = counts[gs:ge]
    starts_list = count_offsets[tids_g].tolist()
    ends_list = count_offsets[tids_g + 1].tolist()
    nz_names = [count_data[s:e].decode("utf-8") for s, e in zip(starts_list, ends_list)]
    nz_counts = cnts_g.tolist()
    return Counter(dict(zip(nz_names, nz_counts)))


# Per-worker cache of corpus-wide token counts, keyed by (db_path, count_lemmas)
_TOKEN_COUNTS_CACHE: dict = {}


def _corpus_counts(db_path, counts, count_lemmas):
    """The corpus counts (sentence_df: the sentences each word or lemma is in; token_counts: the times it occurs) the
    loader wrote next to the collocation arrays (PostFilters.write_corpus_counts)."""
    path = os.path.join(db_path, "collocations", f"{counts}_{'lemma' if count_lemmas else 'word'}.npy")
    try:
        return np.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No {path}: the database was loaded before its corpus counts were. "
            "Write them with python3 -m philologic.utils.write_corpus_counts <database>"
        ) from None


def _load_token_counts(db_path, count_lemmas):
    """How many times each word (or lemma) occurs in the corpus, cached per worker."""
    key = (db_path, bool(count_lemmas))
    if key not in _TOKEN_COUNTS_CACHE:
        _TOKEN_COUNTS_CACHE[key] = _corpus_counts(db_path, "token_counts", count_lemmas)
    return _TOKEN_COUNTS_CACHE[key]


def _load_sentence_df(db_path, count_lemmas):
    """Return ``(df_array, n_sentences)`` for the corpus, cached per worker: the number of sentences each word (or
    lemma) is in, and of sentences."""
    key = (db_path, bool(count_lemmas))
    if key not in _SENTENCE_DF_CACHE:
        sent_offsets = np.load(os.path.join(db_path, "collocations", "sent_offsets.npy"), mmap_mode="r")
        _SENTENCE_DF_CACHE[key] = (_corpus_counts(db_path, "sentence_df", count_lemmas), len(sent_offsets) - 1)
    return _SENTENCE_DF_CACHE[key]


def _df_threshold_pct(request):
    """Return the document-frequency threshold percent, or None if unset.

    Clamped to [MIN_DF_PCT, MAX_DF_PCT]. Empty/invalid values return None so
    callers can fall back to the legacy top-N frequency filter.
    """
    raw = request["filter_df_pct"]
    if raw in (None, ""):
        return None
    try:
        pct = float(raw)
    except (ValueError, TypeError):
        return None
    return max(MIN_DF_PCT, min(MAX_DF_PCT, pct))


def _df_filter_list(db_path, count_lemmas, pct):
    """Words whose sentence document-frequency exceeds ``pct`` percent.

    The decoded vocab strings are exactly the count-vocab keys (lemma entries
    already carry the ``lemma:`` prefix), so they can be returned verbatim.
    """
    df, n_sentences = _load_sentence_df(db_path, count_lemmas)
    if n_sentences == 0:
        return []
    threshold = n_sentences * (pct / 100.0)
    ids = np.nonzero(df > threshold)[0]
    if len(ids) == 0:
        return []

    colloc_dir = os.path.join(db_path, "collocations")
    if count_lemmas:
        voff_file = "attr_lemma_vocab_offsets.npy"
        vblob_file = "attr_lemma_vocab_strings.bin"
    else:
        voff_file = "vocab_offsets.npy"
        vblob_file = "vocab_strings.bin"
    v_offsets = np.load(os.path.join(colloc_dir, voff_file), mmap_mode="r")
    with open(os.path.join(colloc_dir, vblob_file), "rb") as f:
        v_blob = f.read()

    out = []
    for i in ids.tolist():
        s = int(v_offsets[i])
        e = int(v_offsets[i + 1])
        out.append(v_blob[s:e].decode("utf-8"))
    return out


def build_filter_list(request, config, count_lemmas, choice=None):
    """The words to filter out: the database's stopwords (choice "stopwords"), or its most frequent words (or lemmas).
    None if the stopwords list is not found, or the database has none. choice is the request's colloc_filter_choice
    unless given."""
    if choice is None:
        choice = request.colloc_filter_choice
    if choice == "stopwords":
        if not config.stopwords:
            return None
        if "/" not in config.stopwords:
            filter_file = os.path.join(config.db_path, "data", config.stopwords)
        elif os.path.isabs(config.stopwords):
            filter_file = config.stopwords
        else:
            return None
        if not os.path.exists(filter_file):
            return None
        filter_num = float("inf")
    else:
        df_pct = _df_threshold_pct(request)
        # Legacy top-N frequency filter is preserved for backward compatibility:
        # only taken when an explicit filter_frequency is supplied without a
        # df threshold. Otherwise the document-frequency filter is the default.
        if df_pct is None and request.filter_frequency:
            if count_lemmas is True:
                filter_file = config.db_path + "/data/frequencies/lemmas"
            else:
                filter_file = config.db_path + "/data/frequencies/word_frequencies"
            try:
                filter_num = int(request.filter_frequency)
            except ValueError:
                raise BadRequest(
                    f"The number of most frequent words to filter must be a number: {request.filter_frequency}"
                )
        else:
            if df_pct is None:
                df_pct = DEFAULT_DF_PCT
            return _df_filter_list(os.path.join(config.db_path, "data"), count_lemmas, df_pct)
    filter_list = []
    with open(filter_file, encoding="utf8") as filehandle:
        for line_count, line in enumerate(filehandle):
            if line_count == filter_num:
                break
            try:
                word = line.split()[0]
            except IndexError:
                continue
            if count_lemmas is True and "lemma:" not in word:
                filter_list.append(f"lemma:{word}")
            else:
                filter_list.append(word)
    return filter_list


def create_file_path(request, field, hitlist_dir, ext=".pickle", filters=None):
    hash = hashlib.sha1()
    if filters is not None:
        hash.update(filters.key().encode("utf-8"))
    hash.update(request["q"].encode("utf-8"))
    hash.update(request["method"].encode("utf-8"))
    hash.update(str(request["method_arg"]).encode("utf-8"))
    hash.update(request.colloc_filter_choice.encode("utf-8"))
    hash.update(request.q_attribute.encode("utf-8"))
    hash.update(request.q_attribute_value.encode("utf-8"))
    hash.update(str(request.colloc_within).encode("utf-8"))
    hash.update(str(request.filter_frequency).encode("utf-8"))
    hash.update(str(request["filter_df_pct"]).encode("utf-8"))
    hash.update(field.encode("utf-8"))
    for k, v in sorted(request.metadata.items()):
        if v:
            hash.update(f"{k}={v}".encode("utf-8"))
    return os.path.join(hitlist_dir, f"{hash.hexdigest()}{ext}")


if __name__ == "__main__":
    import sys

    db_path = sys.argv[1]
    q = sys.argv[2]

    class Request:
        def __init__(self, query):
            self.q = query
            self.method = "proxy"
            self.arg = ""
            self.colloc_filter_choice = "frequency"
            self.colloc_lemmas = ""
            self.q_attribute = ""
            self.q_attribute_value = ""
            self.colloc_within = "sent"
            self.filter_frequency = 100
            self.filter_df_pct = ""
            self.start = 0
            self.max_time = 10
            self.map_field = ""
            self.metadata = {}
            self.arg_proxy = ""
            self.method_arg = ""
            self.first = "true"

        def __getitem__(self, key):
            return getattr(self, key)

        def __iter__(self):
            return iter(self.__dict__.items())

    class Config:
        def __init__(self, db_path):
            self.db_path = db_path
            self.stopwords = ""

    request = Request(q)
    config = Config(db_path)

    t0 = timeit.default_timer()
    result = collocation_results(request, config)
    elapsed = timeit.default_timer() - t0
    print(f"Total time: {elapsed:.2f}s")
    print(f"Hits: {result['results_length']:,}")
    print("Top 20 collocates:")
    for word, count in result["collocates"][:20]:
        print(f"  {word}: {count:,}")
