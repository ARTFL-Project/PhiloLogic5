import fcntl
import hashlib
import os
import time
from contextlib import contextmanager
import numba
import numpy as np
from philologic.runtime import kwic_hit_object, page_interval
from philologic.runtime.DB import DB
from philologic.runtime.exceptions import BadRequest
from philologic.runtime.HitList import sort_ranks
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
    """Classify sentence words and output sort rank arrays. Returns (n_written, new_index). The hits in no sentence of
    the collocation index have the ranks of no word."""
    ri = 0
    index = start_index
    for i in range(n_total):
        if not hit_matched[i]:
            hit_indices[ri] = index
            for j in range(10):
                left_ranks[ri, j] = zzz_rank
                right_ranks[ri, j] = zzz_rank
                query_ranks[ri, j] = zzz_rank
            ri += 1
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
    return os.path.join(db.hitlist_dir, f"{h.hexdigest()}.kwic")



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


def _prefix_keys(ids, depth):
    """The first depth columns of rows of object ids, as byte strings that compare like the id tuples."""
    return np.ascontiguousarray(ids[:, :depth], dtype=">u4").view(f"S{4 * depth}").ravel()


def _metadata_rank_tables(db, metadata_fields, zzz_rank):
    """For each metadata field, (prefix_len, the sorted id keys of its objects, the rank of their values). The values
    are ranked in the order the concordance sorts them (regardless of case, and of accents with ascii_conversion):
    values sorting alike share a rank, missing ones (zzz_rank) come last."""
    metadata_caches = bulk_load_metadata(db, metadata_fields, inherit=True)
    tables = []
    for field in metadata_fields:
        prefix_len, cache = metadata_caches.get(field, (1, {}))
        value_rank = sort_ranks(cache.values(), db.locals.ascii_conversion)
        keys = _prefix_keys(np.array(list(cache), dtype=np.uint32).reshape(-1, prefix_len), prefix_len)
        ranks = np.array([value_rank.get(value, zzz_rank) for value in cache.values()], dtype=np.uint32)
        order = np.argsort(keys, kind="stable")
        tables.append((prefix_len, keys[order], ranks[order]))
    return tables


def _metadata_ranks(all_hits, metadata_tables, zzz_rank):
    """The ranks of the metadata values of hits (rows of a hitlist), a column per field: zzz_rank for the hits in no
    object of the field."""
    columns = np.full((len(all_hits), len(metadata_tables)), zzz_rank, dtype=np.uint32)
    for column, (prefix_len, keys, ranks) in enumerate(metadata_tables):
        if len(keys):
            hit_keys = _prefix_keys(all_hits, prefix_len)
            found = np.minimum(np.searchsorted(keys, hit_keys), len(keys) - 1)
            matched = keys[found] == hit_keys
            columns[matched, column] = ranks[found[matched]]
    return columns


def _collect_vectorized(hits, bin_path, colloc_dir, db, metadata_fields=()):
    """Vectorized data collection: a rank record for each hit, appended to bin_path as the search finds them. With
    metadata_fields, the records end with the ranks of the hits' values of these fields."""
    token_ids_mmap, sent_offsets, has_number, sent_flat, vocab_sort_rank = _get_kwic_arrays(
        colloc_dir, db.locals.ascii_conversion
    )
    n_sents = len(sent_flat)
    zzz_rank = np.uint32(len(vocab_sort_rank))
    metadata_tables = _metadata_rank_tables(db, metadata_fields, zzz_rank)

    index = 0

    while True:
        hits.update()
        total = len(hits)

        # Read available raw hits from hitlist file
        with hits.open_raw() as f:
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
            hit_indices = np.empty(n_total, dtype=np.uint32)
            left_ranks = np.empty((n_total, 10), dtype=np.uint32)
            right_ranks = np.empty((n_total, 10), dtype=np.uint32)
            query_ranks = np.empty((n_total, 10), dtype=np.uint32)

            n_written, index = _nb_build_ranks(
                token_ids_mmap, sent_offsets, has_number,
                indices.astype(np.int64), first_pos_arr, last_pos_arr, matched,
                vocab_sort_rank, zzz_rank,
                hit_indices, left_ranks, right_ranks, query_ranks,
                index, n_total,
            )

            if n_written > 0:
                records = np.empty((n_written, 31 + len(metadata_tables)), dtype=np.uint32)
                records[:, 0] = hit_indices[:n_written]
                records[:, 1:11] = left_ranks[:n_written]
                records[:, 11:21] = right_ranks[:n_written]
                records[:, 21:31] = query_ranks[:n_written]
                if metadata_tables:
                    records[:, 31:] = _metadata_ranks(all_hits, metadata_tables, zzz_rank)
                with open(bin_path, "ab") as f:
                    f.write(records.tobytes())

        if hits.done and index >= total:
            break

        if not raw:
            time.sleep(0.05)


@contextmanager
def _sort_lock(cache_path):
    """Hold the exclusive flock of a KWIC sort, so that one request at a time collects and sorts it: the others for the
    same sort wait for its .sorted, instead of collecting into the same .bin. The kernel releases the flock of a worker
    that dies, and the next request sorts again."""
    lock_path = cache_path + ".lock"
    while True:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            if os.path.samestat(os.fstat(fd), os.stat(lock_path)):
                break
        except FileNotFoundError:  # removed by the hitlist cleanup while we waited: lock the new one
            pass
        os.close(fd)
    try:
        os.utime(lock_path)  # recent, so that the hitlist cleanup leaves it while it is held
        yield
    finally:
        os.close(fd)


def _sort_cache(bin_path, sorted_path, request, metadata_fields=None):
    """Sort collected binary cache and write .sorted file, whole: requests page through it without the lock."""
    if os.path.exists(bin_path):
        sorted_indices = _numpy_sort(bin_path, request, metadata_fields)
    else:
        # No data collected (zero results) — write empty sentinel
        sorted_indices = np.array([], dtype=np.uint32)
    sorted_indices.tofile(sorted_path + ".tmp")
    os.replace(sorted_path + ".tmp", sorted_path)
    if os.path.exists(bin_path):
        os.remove(bin_path)


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

def get_sorted_kwic(request, config):
    """A page of the KWIC sorted by the words around its hits or by metadata. The sort is cached: the other pages
    come from the cache."""
    if not request.q:
        raise BadRequest("A search term (q) is required to sort a KWIC")
    db = DB(config.db_path + "/data/")
    hits = db.query(request["q"], request["method"], request["arg"], **request.metadata)

    cache_path = _get_cache_path(request, db)
    sorted_path = f"{cache_path}.sorted"
    bin_path = cache_path + ".bin"

    # Fast pagination shortcut: .sorted already exists
    if os.path.exists(sorted_path):
        return _paginate(sorted_path, hits, request, config, db)

    with _sort_lock(cache_path):
        if not os.path.exists(sorted_path):  # else sorted by the request that held the lock before
            # Clean stale partial caches from interrupted requests
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
            metadata_fields = list(config.kwic_metadata_sorting_fields) if metadata_search else []

            _collect_vectorized(hits, bin_path, colloc_dir, db, metadata_fields)

            _sort_cache(bin_path, sorted_path, request, metadata_fields)
    return _paginate(sorted_path, hits, request, config, db)
