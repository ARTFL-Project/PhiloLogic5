#!/var/lib/philologic5/philologic_env/bin/python3
"""Collocation results"""

import hashlib
import os
import pickle
import struct
import tempfile
import time
import timeit
from collections import Counter
from typing import Any

import numba
import numpy as np

from philologic.runtime.DB import DB
from philologic.runtime.MetadataQuery import bulk_load_metadata
from philologic.runtime.Query import get_word_groups
from philologic.runtime.sql_validation import validate_column


@numba.njit(cache=True)
def _nb_compare_keys(a, b):
    """Lexicographic comparison of two 6-element uint32 key rows."""
    for k in range(6):
        if a[k] < b[k]:
            return -1
        elif a[k] > b[k]:
            return 1
    return 0


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
def _nb_searchsorted(sent_keys, hit_keys, n_sents):
    """Parallel binary search of hit keys against sorted sentence keys."""
    n_hits = hit_keys.shape[0]
    result = np.empty(n_hits, dtype=np.int64)
    match = np.empty(n_hits, dtype=numba.boolean)
    for i in numba.prange(n_hits):
        lo, hi = np.int64(0), np.int64(n_sents)
        while lo < hi:
            mid = (lo + hi) // 2
            if _nb_compare_keys(sent_keys[mid], hit_keys[i]) < 0:
                lo = mid + 1
            else:
                hi = mid
        result[i] = lo
        match[i] = lo < n_sents and _nb_compare_keys(sent_keys[lo], hit_keys[i]) == 0
    return result, match


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
def _nb_fused_count_distance(sent_indices, hit_q_pos, sent_offsets, token_ids, is_filtered, vocab_size, max_dist):
    """Fused count with distance constraint."""
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
                diff = np.int32(pos) - np.int32(q_pos)
                if diff < 0:
                    diff = -diff
                if diff <= max_dist:
                    token = token_ids[j]
                    if not is_filtered[token]:
                        local_counts[tid, token] += 1
            pos += 1
    return local_counts


@numba.njit(parallel=True, cache=True)
def _nb_fused_count_attr(sent_indices, hit_q_pos, sent_offsets, count_ids, is_filtered, attr_ids, target_attr_id, vocab_size):
    """Fused count with attribute inclusion filter and identity exclusion."""
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
            if pos != q_pos and attr_ids[j] == target_attr_id:
                token = count_ids[j]
                if not is_filtered[token]:
                    local_counts[tid, token] += 1
            pos += 1
    return local_counts


@numba.njit(parallel=True, cache=True)
def _nb_fused_count_attr_distance(sent_indices, hit_q_pos, sent_offsets, count_ids, is_filtered, attr_ids, target_attr_id, vocab_size, max_dist):
    """Fused count with attribute inclusion filter, identity exclusion, and distance constraint."""
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
                diff = np.int32(pos) - np.int32(q_pos)
                if diff < 0:
                    diff = -diff
                if diff <= max_dist and attr_ids[j] == target_attr_id:
                    token = count_ids[j]
                    if not is_filtered[token]:
                        local_counts[tid, token] += 1
            pos += 1
    return local_counts

@numba.njit(parallel=True, cache=True)
def _nb_gather_grouped(
    sent_indices, hit_q_pos, hit_gids, sent_offsets,
    count_ids, is_filtered, attr_ids, target_attr_id,
    use_attr, max_dist, hit_word_offsets,
    out_tokens, out_gids, out_valid,
):
    """Single-pass gather: annotate every word with token, group, validity.

    One kernel call for ALL hits (all groups combined). Produces flat arrays
    that are then split by group in numpy for per-group bincount.
    """
    n = len(sent_indices)
    for i in numba.prange(n):
        si = sent_indices[i]
        s = numba.int64(sent_offsets[si])
        e = numba.int64(sent_offsets[si + 1])
        base = hit_word_offsets[i]
        qp = hit_q_pos[i]
        gid = hit_gids[i]
        pos = np.uint32(1)
        for j in range(s, e):
            w = base + (j - s)
            t = count_ids[j]
            out_tokens[w] = t
            out_gids[w] = gid
            ok = pos != qp
            if ok and use_attr:
                ok = attr_ids[j] == target_attr_id
            if ok:
                ok = not is_filtered[t]
            if ok and max_dist >= np.int32(0):
                diff = np.int32(pos) - np.int32(qp)
                if diff < 0:
                    diff = -diff
                ok = diff <= max_dist
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

def _vectorized_collocation(
    db_path,
    hits,
    filter_list,
    count_lemmas,
    attribute,
    attribute_value,
    collocate_distance,
    map_field_info=None,
):
    """Fully vectorized collocation counting using columnar numpy arrays.

    Returns either a Counter (when map_field_info is None) or
    a dict {metadata_value: Counter} (when map_field_info is set).
    """
    colloc_dir = os.path.join(db_path, "collocations")

    # Memory-map arrays (zero-copy, OS manages page caching)
    token_ids_mmap = np.load(os.path.join(colloc_dir, "token_ids.npy"), mmap_mode="r")
    sent_offsets = np.load(os.path.join(colloc_dir, "sent_offsets.npy"), mmap_mode="r")
    sent_keys_native = np.load(os.path.join(colloc_dir, "sent_keys_native.npy"), mmap_mode="r")
    sent_offsets_arr = np.asarray(sent_offsets)

    # Bulk read hitlist
    hits.finish()

    with open(hits.filename, "rb") as f:
        raw = f.read()

    all_hits = np.frombuffer(raw, dtype=np.uint32).reshape(-1, hits.length)

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
    filter_hashes = np.array(
        sorted(struct.unpack("<Q", hashlib.md5(w.encode("utf-8")).digest()[:8])[0] for w in filter_list),
        dtype=np.uint64,
    )
    is_filtered = _nb_build_filter(count_hashes, filter_hashes)

    #  Load attribute arrays if needed
    if attribute is not None:
        attr_ids_mmap = np.load(os.path.join(colloc_dir, f"attr_{attribute}_ids.npy"), mmap_mode="r")
        attr_vocab = np.load(os.path.join(colloc_dir, f"attr_{attribute}_vocab.npy"), allow_pickle=True)
        target_attr_id = np.uint32(
            next((i for i, v in enumerate(attr_vocab) if v == attribute_value), len(attr_vocab))
        )

    #  Parallel binary search (all hits at once)
    hit_keys = np.ascontiguousarray(all_hits[:, :6])
    si, match = _nb_searchsorted(sent_keys_native, hit_keys, len(sent_keys_native))
    sent_indices = si[match]
    hit_q_pos = all_hits[match, 7]

    #  Helpers: fused count + decode (shared by all modes)
    def _fused_count(group_si, group_qp):
        if attribute is not None:
            if collocate_distance is not None:
                return _nb_fused_count_attr_distance(
                    group_si, group_qp, sent_offsets_arr,
                    count_ids, is_filtered, attr_ids_mmap, target_attr_id,
                    count_vocab_size, collocate_distance,
                )
            return _nb_fused_count_attr(
                group_si, group_qp, sent_offsets_arr,
                count_ids, is_filtered, attr_ids_mmap, target_attr_id,
                count_vocab_size,
            )
        if collocate_distance is not None:
            return _nb_fused_count_distance(
                group_si, group_qp, sent_offsets_arr,
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
        if attribute is not None:
            suffix = f":{attribute}:{attribute_value}"
            if count_lemmas:
                nz_names = [n + suffix for n in nz_names]
            else:
                nz_names = [n.lower() + suffix for n in nz_names]
        return Counter(dict(zip(nz_names, nz_counts)))

    #  Simple mode: one fused count over all hits
    if map_field_info is None:
        return _decode_counts(_fused_count(sent_indices, hit_q_pos))

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
        if mv is not None:
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
    if attribute is not None:
        _nb_gather_grouped(
            v_si, v_qp, v_gids, sent_offsets_arr,
            count_ids, is_filtered, attr_ids_mmap, target_attr_id,
            np.bool_(True), _max_dist, hit_word_offsets,
            out_tokens, out_gids, out_valid,
        )
    else:
        _nb_gather_grouped(
            v_si, v_qp, v_gids, sent_offsets_arr,
            count_ids, is_filtered, np.empty(1, dtype=np.uint32), np.uint32(0),
            np.bool_(False), _max_dist, hit_word_offsets,
            out_tokens, out_gids, out_valid,
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

    # Return raw numpy arrays — string decode is deferred to downstream scripts
    group_bounds = np.searchsorted(unique_gids, np.arange(n_groups + 1, dtype=np.int32))
    # Ensure group_names are strings (metadata values may be int, e.g. year)
    group_names = [str(n) for n in group_names]
    return unique_tids, unique_counts, group_bounds, group_names

def collocation_results(request, config):
    """Fetch collocation results"""
    collocation_object: dict[str, Any] = {"query": dict([i for i in request])}
    db = DB(config.db_path + "/data/")

    map_field = request.map_field or None
    if map_field is not None:
        map_field = validate_column(map_field, db)

    hits = db.query(
        request.q,
        "single_term",
        request.arg,
        raw_results=True,
        raw_bytes=True,
        **request.metadata,
    )

    try:
        collocate_distance = int(request.method_arg)
    except ValueError:
        collocate_distance = None

    count_lemmas = "lemma:" in request.q

    if request.colloc_filter_choice == "attribute":
        attribute = request.q_attribute
        attribute_value = request.q_attribute_value
    else:
        attribute = None
        attribute_value = None

    # Build list of search terms to filter out.
    # When counting lemmas, vocab entries are "lemma:{value}", so filter
    # strings must be lemma-prefixed to match the vocab hashes.
    query_words = []
    while not os.path.exists(f"{hits.filename}.terms"):
        time.sleep(0.1)
    for group in get_word_groups(f"{hits.filename}.terms"):
        for word in group:
            if count_lemmas:
                query_words.extend([f"lemma:{word}", f"lemma:{word.lower()}"])
            else:
                query_words.extend([word, word.title(), word.upper()])

    if request.colloc_filter_choice == "nofilter":
        filter_list = set(query_words)
    elif request.colloc_filter_choice == "attribute":
        if f"{attribute}:{attribute_value}" not in request.q:
            filter_list = {f"{word}:{attribute}:{attribute_value}" for word in query_words}
        else:
            filter_list = set(query_words)
            filter_list = filter_list.union(set(query_words))
        filter_list.add(f"{request.q}:{attribute}:{attribute_value}")
    else:
        filter_list = set(build_filter_list(request, config, count_lemmas))
        filter_list = filter_list.union(set(query_words))
    collocation_object["filter_list"] = sorted(filter_list, key=str.lower)

    hits.finish()
    total_hits = len(hits)

    map_field_info = None
    if map_field is not None:
        field_obj_index, metadata_cache = bulk_load_metadata(db, [map_field])[map_field]
        map_field_info = (metadata_cache, field_obj_index)

    result = _vectorized_collocation(
        db.path,
        hits,
        filter_list,
        count_lemmas,
        attribute,
        attribute_value,
        collocate_distance,
        map_field_info=map_field_info,
    )

    collocation_object["results_length"] = total_hits
    collocation_object["distance"] = collocate_distance

    if map_field is None:
        all_collocates = result
        if None in all_collocates:
            del all_collocates[None]
        # Cache full Counter to disk; return only top 100 for display
        file_path = create_file_path(request, "", config.db_path)
        atomic_pickle_dump(all_collocates, file_path)
        collocation_object["collocates"] = all_collocates.most_common(100)
        collocation_object["file_path"] = file_path
    else:
        unique_tids, unique_counts, group_bounds, group_names = result
        file_path = create_file_path(request, map_field, config.db_path, ext=".npz")
        save_map_field_cache(
            file_path, unique_tids, unique_counts, group_bounds, group_names,
            count_lemmas, attribute, attribute_value,
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


def save_map_field_cache(file_path, unique_tids, unique_counts, group_bounds, group_names,
                         count_lemmas, attribute, attribute_value):
    """Save map_field results as numpy arrays (no string decode, no pickle overhead)."""
    dir_path = os.path.dirname(file_path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(
                f,
                tids=unique_tids.astype(np.uint32),
                counts=unique_counts.astype(np.int32),
                group_bounds=group_bounds.astype(np.int64),
                group_names=np.frombuffer("\0".join(group_names).encode("utf-8"), dtype=np.uint8),
                meta=np.array([int(count_lemmas)], dtype=np.uint8),
                attr=np.frombuffer(
                    (attribute or "").encode("utf-8"), dtype=np.uint8
                ) if attribute else np.array([], dtype=np.uint8),
                attr_val=np.frombuffer(
                    (attribute_value or "").encode("utf-8"), dtype=np.uint8
                ) if attribute_value else np.array([], dtype=np.uint8),
            )
        os.replace(tmp_path, file_path)
    except BaseException:
        os.unlink(tmp_path)
        raise


def load_map_field_cache(file_path):
    """Load map_field numpy cache. Returns (tids, counts, group_bounds, group_names, count_lemmas, attribute, attribute_value)."""
    data = np.load(file_path, allow_pickle=False)
    tids = data["tids"]
    counts = data["counts"]
    group_bounds = data["group_bounds"]
    group_names = data["group_names"].tobytes().decode("utf-8").split("\0")
    count_lemmas = bool(data["meta"][0])
    attr_raw = data["attr"]
    attribute = attr_raw.tobytes().decode("utf-8") if len(attr_raw) > 0 else None
    attr_val_raw = data["attr_val"]
    attribute_value = attr_val_raw.tobytes().decode("utf-8") if len(attr_val_raw) > 0 else None
    return tids, counts, group_bounds, group_names, count_lemmas, attribute, attribute_value


def decode_group_collocates(tids, counts, group_bounds, group_index, db_path, count_lemmas, attribute, attribute_value):
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
    if attribute is not None:
        suffix = f":{attribute}:{attribute_value}"
        if count_lemmas:
            nz_names = [n + suffix for n in nz_names]
        else:
            nz_names = [n.lower() + suffix for n in nz_names]
    return Counter(dict(zip(nz_names, nz_counts)))


def build_filter_list(request, config, count_lemmas):
    """set up filtering with stopwords or most frequent terms."""
    if config.stopwords and request.colloc_filter_choice == "stopwords":
        if config.stopwords and "/" not in config.stopwords:
            filter_file = os.path.join(config.db_path, "data", config.stopwords)
        elif os.path.isabs(config.stopwords):
            filter_file = config.stopwords
        else:
            return ["stopwords list not found"]
        if not os.path.exists(filter_file):
            return ["stopwords list not found"]
        filter_num = float("inf")
    elif count_lemmas is True:
        filter_file = config.db_path + "/data/frequencies/lemmas"
        if request.filter_frequency:
            filter_num = int(request.filter_frequency)
        else:
            filter_num = 100
    else:
        filter_file = config.db_path + "/data/frequencies/word_frequencies"
        if request.filter_frequency:
            filter_num = int(request.filter_frequency)
        else:
            filter_num = 100
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


def get_metadata_value(sql_cursor, field, sentence_id, index, obj_level):
    """Get metadata value"""
    object_id = " ".join(map(str, struct.unpack(f"{index}I", sentence_id[: index * 4])))
    sql_cursor.execute(f"SELECT {field} FROM toms WHERE philo_{obj_level}_id=?", (object_id,))
    return sql_cursor.fetchone()[0]


def create_file_path(request, field, path, ext=".pickle"):
    hash = hashlib.sha1()
    hash.update(request["q"].encode("utf-8"))
    hash.update(request["method"].encode("utf-8"))
    hash.update(str(request["method_arg"]).encode("utf-8"))
    hash.update(request.colloc_filter_choice.encode("utf-8"))
    hash.update(request.q_attribute.encode("utf-8"))
    hash.update(request.q_attribute_value.encode("utf-8"))
    hash.update(str(request.colloc_within).encode("utf-8"))
    hash.update(str(request.filter_frequency).encode("utf-8"))
    hash.update(field.encode("utf-8"))
    for k, v in sorted(request.metadata.items()):
        if v:
            hash.update(f"{k}={v}".encode("utf-8"))
    return f"{path}/data/hitlists/{hash.hexdigest()}{ext}"


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
            self.q_attribute = ""
            self.q_attribute_value = ""
            self.colloc_within = "sent"
            self.filter_frequency = 100
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
