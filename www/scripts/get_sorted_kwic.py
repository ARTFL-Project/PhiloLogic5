import hashlib
import os
import time
import timeit
import numba
import numpy as np
import orjson
from custom_functions_loader import get_custom
from philologic.runtime import WebConfig, WSGIHandler, kwic_hit_object, page_interval
from philologic.runtime.DB import DB
from philologic.runtime.MetadataQuery import bulk_load_metadata

# Module-level cache for gunicorn persistent workers
_KWIC_CACHE = {}

# Column layout in .kwic.bin records (31 uint32 per record)
_COL_INDEX = 0
_COL_LEFT = slice(1, 11)
_COL_RIGHT = slice(11, 21)
_COL_QUERY = slice(21, 31)

_FIELD_TO_COLS = {
    "left": _COL_LEFT,
    "right": _COL_RIGHT,
    "q": _COL_QUERY,
}

@numba.njit(cache=True)
def _nb_build_ranks(
    token_ids, sent_offsets, has_number,
    hit_sent_idx, hit_first_pos, hit_last_pos, hit_matched,
    vocab_sort_rank, zzz_rank,
    hit_indices, left_ranks, right_ranks, query_ranks,
    start_index, n_total,
):
    """Classify sentence words and output sort rank arrays. Returns (n_written, new_index)."""
    ri = 0
    index = start_index
    for i in range(n_total):
        if not hit_matched[i]:
            index += 1
            continue
        si = hit_sent_idx[i]
        s_start = sent_offsets[si]
        s_end = sent_offsets[si + 1]
        fp = hit_first_pos[i]
        lp = hit_last_pos[i]

        left_buf = np.empty(10, dtype=np.uint32)
        nl = 0
        nr = 0
        nq = 0
        pos = 1
        for k in range(s_start, s_end):
            tid = token_ids[k]
            if has_number[tid]:
                pos += 1
                continue
            rank = vocab_sort_rank[tid]
            if fp > pos:
                left_buf[nl % 10] = rank
                nl += 1
            elif pos > lp:
                if nr < 10:
                    right_ranks[ri, nr] = rank
                nr += 1
            else:
                if nq < 10:
                    query_ranks[ri, nq] = rank
                nq += 1
            pos += 1

        hit_indices[ri] = index

        # Left: last 10 reversed (newest first), pad with sentinel
        if nl == 0:
            for j in range(10):
                left_ranks[ri, j] = zzz_rank
        else:
            n_left = min(nl, 10)
            for j in range(n_left):
                left_ranks[ri, j] = left_buf[(nl - 1 - j) % 10]
            for j in range(n_left, 10):
                left_ranks[ri, j] = zzz_rank

        # Pad right with sentinel
        if nr == 0:
            for j in range(10):
                right_ranks[ri, j] = zzz_rank
        else:
            for j in range(min(nr, 10), 10):
                right_ranks[ri, j] = zzz_rank

        # Pad query with sentinel
        for j in range(min(nq, 10), 10):
            query_ranks[ri, j] = zzz_rank

        ri += 1
        index += 1
    return ri, index

def _get_kwic_arrays(colloc_dir, ascii_conversion):
    """Load and cache all arrays needed for vectorized KWIC sorting."""
    cache_key = colloc_dir
    if cache_key not in _KWIC_CACHE:
        # Evict oldest entry to bound memory (sent_flat can be large)
        if len(_KWIC_CACHE) >= 10:
            _KWIC_CACHE.pop(next(iter(_KWIC_CACHE)))
        token_ids = np.load(os.path.join(colloc_dir, "token_ids.npy"), mmap_mode="r")
        sent_offsets = np.load(os.path.join(colloc_dir, "sent_offsets.npy"), mmap_mode="r")
        has_number = np.load(os.path.join(colloc_dir, "vocab_has_number.npy"))

        s24_path = os.path.join(colloc_dir, "sent_keys_s24.npy")
        if os.path.exists(s24_path):
            sent_flat = np.load(s24_path, mmap_mode="r")
        else:
            sent_keys_be = np.load(os.path.join(colloc_dir, "sent_keys_be.npy"), mmap_mode="r")
            sent_flat = np.ascontiguousarray(sent_keys_be).view("S24").ravel()

        # Load vocab sort rank (precomputed or compute at load time)
        rank_name = "vocab_sort_rank_ascii.npy" if ascii_conversion else "vocab_sort_rank.npy"
        rank_path = os.path.join(colloc_dir, rank_name)
        if not os.path.exists(rank_path):
            rank_path = os.path.join(colloc_dir, "vocab_sort_rank.npy")
        if os.path.exists(rank_path):
            vocab_sort_rank = np.load(rank_path)
        else:
            # Fallback: compute from vocab strings at load time
            if ascii_conversion and os.path.exists(os.path.join(colloc_dir, "vocab_ascii_strings.bin")):
                _offsets = np.load(os.path.join(colloc_dir, "vocab_ascii_offsets.npy"))
                with open(os.path.join(colloc_dir, "vocab_ascii_strings.bin"), "rb") as f:
                    _data = f.read()
            else:
                _offsets = np.load(os.path.join(colloc_dir, "vocab_offsets.npy"))
                with open(os.path.join(colloc_dir, "vocab_strings.bin"), "rb") as f:
                    _data = f.read()
            _vb = np.frombuffer(_data, dtype=np.uint8)
            n_v = len(_offsets) - 1
            _mwl = 48
            _padded = np.zeros((n_v, _mwl), dtype=np.uint8)
            for _i in range(n_v):
                _s = int(_offsets[_i])
                _e = int(_offsets[_i + 1])
                _L = min(_e - _s, _mwl)
                _padded[_i, :_L] = _vb[_s:_s + _L]
            _keys = _padded.view(f"S{_mwl}").ravel()
            _order = np.argsort(_keys, kind="stable")
            _sorted_keys = _keys[_order]
            _differs = np.ones(n_v, dtype=np.bool_)
            if n_v > 1:
                _differs[1:] = _sorted_keys[1:] != _sorted_keys[:-1]
            _ranks_sorted = np.cumsum(_differs).astype(np.uint32) - 1
            vocab_sort_rank = np.empty(n_v, dtype=np.uint32)
            vocab_sort_rank[_order] = _ranks_sorted

        _KWIC_CACHE[cache_key] = (token_ids, sent_offsets, has_number, sent_flat, vocab_sort_rank)
    return _KWIC_CACHE[cache_key]


def _get_cache_path(request, db):
    """Compute deterministic cache path from query parameters."""
    h = hashlib.sha1()
    h.update(request["q"].encode("utf-8"))
    h.update(request["method"].encode("utf-8"))
    h.update(str(request["arg"]).encode("utf-8"))
    h.update(request.first_kwic_sorting_option.encode("utf-8"))
    h.update(request.second_kwic_sorting_option.encode("utf-8"))
    h.update(request.third_kwic_sorting_option.encode("utf-8"))
    for field, metadata in sorted(request.metadata.items(), key=lambda x: x[0]):
        h.update(f"{field}: {metadata}".encode("utf-8"))
    return os.path.join(db.path, "hitlists", f"{h.hexdigest()}.kwic")



def _numpy_sort(bin_path, request, metadata_fields=None):
    """Sort binary rank records using numpy composite key. Returns sorted hit indices."""
    if metadata_fields is None:
        metadata_fields = []
    record_width = 31 + len(metadata_fields)

    with open(bin_path, "rb") as f:
        raw = f.read()
    records = np.frombuffer(raw, dtype=np.uint32).reshape(-1, record_width)
    hit_indices = records[:, _COL_INDEX]

    # Build column index for metadata fields
    meta_col = {field: 31 + i for i, field in enumerate(metadata_fields)}

    # Collect sort key column slices (word fields: 10 cols, metadata: 1 col)
    key_parts = []
    for opt in (request.first_kwic_sorting_option, request.second_kwic_sorting_option, request.third_kwic_sorting_option):
        if opt and opt in _FIELD_TO_COLS:
            key_parts.append(records[:, _FIELD_TO_COLS[opt]])
        elif opt and opt in meta_col:
            key_parts.append(records[:, meta_col[opt] : meta_col[opt] + 1])

    if not key_parts:
        return hit_indices  # no sort requested

    combined = np.concatenate(key_parts, axis=1)
    composite = np.ascontiguousarray(combined).astype(">u4")
    key_len = combined.shape[1] * 4
    composite_key = composite.view(f"S{key_len}").ravel()
    sorted_idx = np.argsort(composite_key, kind="stable")
    return hit_indices[sorted_idx]


def _collect_vectorized(hits, bin_path, colloc_dir, db):
    """Vectorized data collection with streaming progress. Generator yields NDJSON lines."""
    token_ids_mmap, sent_offsets, has_number, sent_flat, vocab_sort_rank = _get_kwic_arrays(
        colloc_dir, db.locals.ascii_conversion
    )
    n_sents = len(sent_flat)
    zzz_rank = np.uint32(len(vocab_sort_rank))

    index = 0
    last_progress = timeit.default_timer()

    while True:
        hits.update()
        total = len(hits)

        # Read available raw hits from hitlist file
        with open(hits.filename, "rb") as f:
            f.seek(index * hits.hitsize)
            raw = f.read()
        n_bytes = len(raw) - (len(raw) % hits.hitsize)
        raw = raw[:n_bytes]

        if raw:
            all_hits = np.frombuffer(raw, dtype=np.uint32).reshape(-1, hits.length)
            n_total = all_hits.shape[0]

            # Vectorized sentence lookup via numpy searchsorted on byte keys
            hit_keys_be = np.ascontiguousarray(all_hits[:, :6].astype(">u4"))
            hit_flat = hit_keys_be.view("S24").ravel()
            indices = np.searchsorted(sent_flat, hit_flat)
            clipped = np.minimum(indices, n_sents - 1)
            matched = (indices < n_sents) & (sent_flat[clipped] == hit_flat)

            # Extract first/last positions for all hits (vectorized)
            pos_cols = list(range(7, hits.length, 2))
            all_positions = all_hits[:n_total, pos_cols]
            if len(pos_cols) == 1:
                first_pos_arr = all_positions[:, 0].astype(np.int64)
                last_pos_arr = first_pos_arr
            else:
                first_pos_arr = np.min(all_positions, axis=1).astype(np.int64)
                last_pos_arr = np.max(all_positions, axis=1).astype(np.int64)

            # Build sort rank arrays via numba
            n_matched = int(matched.sum())
            hit_indices = np.empty(n_matched, dtype=np.uint32)
            left_ranks = np.empty((n_matched, 10), dtype=np.uint32)
            right_ranks = np.empty((n_matched, 10), dtype=np.uint32)
            query_ranks = np.empty((n_matched, 10), dtype=np.uint32)

            n_written, index = _nb_build_ranks(
                token_ids_mmap, sent_offsets, has_number,
                indices.astype(np.int64), first_pos_arr, last_pos_arr, matched,
                vocab_sort_rank, zzz_rank,
                hit_indices, left_ranks, right_ranks, query_ranks,
                index, n_total,
            )

            if n_written > 0:
                records = np.empty((n_written, 31), dtype=np.uint32)
                records[:, 0] = hit_indices[:n_written]
                records[:, 1:11] = left_ranks[:n_written]
                records[:, 11:21] = right_ranks[:n_written]
                records[:, 21:31] = query_ranks[:n_written]
                with open(bin_path, "ab") as f:
                    f.write(records.tobytes())

        # Check exit condition
        if hits.done and index >= total:
            yield orjson.dumps({"progress": {"hits_done": index, "total": total}}) + b"\n"
            break

        # Yield progress (throttled to ~0.5s)
        now = timeit.default_timer()
        if now - last_progress >= 0.5:
            yield orjson.dumps({"progress": {"hits_done": index, "total": total}}) + b"\n"
            last_progress = now

        if not raw:
            time.sleep(0.05)


def _collect_metadata_sort(hits, bin_path, metadata_fields, config, db):
    """Metadata sort collection producing binary records with word + metadata ranks. Yields NDJSON."""
    colloc_dir = os.path.join(db.path, "collocations")
    token_ids, sent_offsets, has_number, sent_flat, vocab_sort_rank = _get_kwic_arrays(
        colloc_dir, db.locals.ascii_conversion
    )
    n_sents = len(sent_flat)
    zzz_rank = int(len(vocab_sort_rank))

    # Bulk-load metadata and build sort rank maps (same ordering as old TSV + LC_ALL=C sort)
    metadata_caches = bulk_load_metadata(db, metadata_fields)
    meta_rank_maps = {}
    for field in metadata_fields:
        prefix_len, cache = metadata_caches.get(field, (1, {}))
        all_normalized = set()
        for val in cache.values():
            all_normalized.add(",".join(f"{val}".lower().split()))
        sorted_vals = sorted(v for v in all_normalized if v)
        norm_to_rank = {v: i for i, v in enumerate(sorted_vals)}
        meta_rank_maps[field] = (prefix_len, cache, norm_to_rank)

    n_meta = len(metadata_fields)
    record_width = 31 + n_meta

    buffer = []
    index = 0
    last_progress = timeit.default_timer()

    for hit in hits:
        # Sentence lookup via big-endian S24 key
        hit_key = np.array(hit.hit[:6], dtype=">u4").tobytes()
        si = int(np.searchsorted(sent_flat, hit_key))

        positions = sorted(word.philo_id[-1] for word in hit.words)

        # Classify sentence words and collect sort ranks
        left_buf = []
        right_buf = []
        query_buf = []
        if si < n_sents and sent_flat[si] == hit_key:
            s_start = int(sent_offsets[si])
            s_end = int(sent_offsets[si + 1])
            pos = 1
            for k in range(s_start, s_end):
                tid = int(token_ids[k])
                if has_number[tid]:
                    pos += 1
                    continue
                rank = int(vocab_sort_rank[tid])
                if positions[0] > pos:
                    left_buf.append(rank)
                elif pos > positions[-1]:
                    right_buf.append(rank)
                else:
                    query_buf.append(rank)
                pos += 1

        # Build record: [index, left(10), right(10), query(10), meta_ranks...]
        record = np.full(record_width, zzz_rank, dtype=np.uint32)
        record[0] = index
        left_slice = left_buf[-10:]
        left_slice.reverse()
        for j, r in enumerate(left_slice):
            record[1 + j] = r
        for j, r in enumerate(right_buf[:10]):
            record[11 + j] = r
        for j, r in enumerate(query_buf[:10]):
            record[21 + j] = r

        # Metadata sort ranks
        hit_tuple = tuple(hit.hit)
        for mi, field in enumerate(metadata_fields):
            prefix_len, cache, norm_to_rank = meta_rank_maps[field]
            val = cache.get(hit_tuple[:prefix_len], "")
            normalized = ",".join(f"{val}".lower().split())
            record[31 + mi] = norm_to_rank.get(normalized, zzz_rank)

        buffer.append(record.tobytes())
        if len(buffer) >= 1000:
            with open(bin_path, "ab") as f:
                f.write(b"".join(buffer))
            buffer.clear()

        index += 1
        now = timeit.default_timer()
        if now - last_progress >= 0.5:
            hits.update()
            yield orjson.dumps({"progress": {"hits_done": index, "total": len(hits)}}) + b"\n"
            last_progress = now

    if buffer:
        with open(bin_path, "ab") as f:
            f.write(b"".join(buffer))

    hits.update()
    yield orjson.dumps({"progress": {"hits_done": index, "total": len(hits)}}) + b"\n"



def _sort_cache(bin_path, sorted_path, request, metadata_fields=None):
    """Sort collected binary cache and write .sorted file."""
    if os.path.exists(bin_path):
        sorted_indices = _numpy_sort(bin_path, request, metadata_fields)
        sorted_indices.tofile(sorted_path)
        os.remove(bin_path)
    else:
        # No data collected (zero results) — write empty sentinel
        np.array([], dtype=np.uint32).tofile(sorted_path)


def _paginate(sorted_path, hits, request, config, db):
    """Read .sorted file and return paginated KWIC result dict."""
    start, end, _ = page_interval(request.results_per_page, hits, request.start, request.end)
    kwic_object = {
        "description": {"start": start, "end": end, "results_per_page": request.results_per_page},
        "query": dict([i for i in request]),
    }

    kwic_results = []
    if os.path.exists(sorted_path):
        sorted_indices = np.fromfile(sorted_path, dtype=np.uint32)
        for pos in range(start - 1, min(end, len(sorted_indices))):
            index = int(sorted_indices[pos])
            hit = hits[index]
            kwic_results.append(kwic_hit_object(hit, config, db))

    kwic_object["results"] = kwic_results
    kwic_object["results_length"] = len(hits)
    kwic_object["query_done"] = hits.done
    return kwic_object

def get_sorted_kwic(environ, start_response):
    """Streaming sorted KWIC: collect sort data, sort, paginate — all over one connection."""
    headers = [
        ("Content-Type", "application/x-ndjson; charset=UTF-8"),
        ("Access-Control-Allow-Origin", "*"),
        ("X-Accel-Buffering", "no"),
        ("Cache-Control", "no-cache"),
    ]
    start_response("200 OK", headers)

    db_path = environ.get("PHILOLOGIC_DBPATH", os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    _WebConfig = get_custom(db_path, "WebConfig", WebConfig)
    _WSGIHandler = get_custom(db_path, "WSGIHandler", WSGIHandler)
    config = _WebConfig(db_path)
    db = DB(config.db_path + "/data/")
    request = _WSGIHandler(environ, config)

    hits = db.query(request["q"], request["method"], request["arg"], **request.metadata)

    cache_path = _get_cache_path(request, db)
    sorted_path = f"{cache_path}.sorted"
    bin_path = cache_path + ".bin"

    # Phase 1: Fast pagination shortcut — .sorted already exists
    if os.path.exists(sorted_path):
        yield orjson.dumps(_paginate(sorted_path, hits, request, config, db)) + b"\n"
        return

    # Phase 2: Clean stale partial caches from interrupted requests
    if os.path.exists(bin_path):
        os.remove(bin_path)
    if os.path.exists(cache_path):
        os.remove(cache_path)

    # Determine sort mode: metadata sort if any sort option is a metadata field
    metadata_search = not (
        request.first_kwic_sorting_option in ("left", "right", "q", "")
        and request.second_kwic_sorting_option in ("left", "right", "q", "")
        and request.third_kwic_sorting_option in ("left", "right", "q", "")
    )
    colloc_dir = os.path.join(db.path, "collocations")

    # Phase 3: Data collection (streams progress lines)
    if metadata_search:
        metadata_fields = list(config.kwic_metadata_sorting_fields)
        yield from _collect_metadata_sort(hits, bin_path, metadata_fields, config, db)
    else:
        metadata_fields = []
        yield from _collect_vectorized(hits, bin_path, colloc_dir, db)

    # Phase 4: Sort
    _sort_cache(bin_path, sorted_path, request, metadata_fields)

    # Phase 5: Paginate and yield final result
    yield orjson.dumps(_paginate(sorted_path, hits, request, config, db)) + b"\n"

