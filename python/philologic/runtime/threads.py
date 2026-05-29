"""Global-first thread detection for collocation evolution.

Identifies persistent "threads" — coherent sets of collocate words representing
distinct uses or senses of the query word over time.

Decomposition is **HyperLex** (Véronis 2004), a white-box word-sense-induction
method: the query's collocates form a weighted graph (words linked by NPMI), and
each distinct sense is anchored by a *hub* — a tightly connected collocate. The
rule, statable in one sentence: "each sense is a hub collocate; every other word
joins the hub it co-occurs with most strongly; words tied to no hub are left
unplaced." It replaced an HDBSCAN density-clustering stack (persistence floor,
EOM/LEAF switch, min_cluster_size sweep) whose parameters were opaque and which
treated a graph as a point cloud — it could not decompose abstract words
(raison, vérité, esprit) at all, where HyperLex decomposes them cleanly.

Algorithm (one pass on all hits, then project to time bins):
  1. Build per-hit collocate bags from the query's hitlist.
  2. Filter candidate vocabulary by:
       - df_floor: word must appear in ≥ 0.5% of hits (min 3),
       - max_df:   word in ≤ 30% of hits (too generic to anchor a sense),
       - length ≥ 2 and the user-configured stopword list (build_filter_list —
         the single exclusion authority, shared with the frequency view).
  3. Build an NPMI distance matrix on the kept vocabulary.
  4. HyperLex (see _hyperlex): hub detection + direct-affinity assignment over a
     single NPMI edge floor, auto-tuned by _auto_floor (a principled base that
     rises only when one hub over-grabs).
  5. Project each sense to per-year-bin intensity = share of hits in the bin
     whose bag intersects the sense's vocabulary.
  6. Label each sense by within-sense ANCHOR strength: rank its words by their
     summed NPMI to the other members (graph centrality inside the sense). This
     is sense-specific (unlike a global corpus-IDF, which can't tell one sense
     of the query from another) and surfaces the tight core, so the hub leads
     and the peripheral members added by step 4's growth don't muddy the label.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Dict, List, Optional, Tuple

import numba
import numpy as np
from scipy.sparse import csr_matrix

from philologic.runtime.DB import DB
from philologic.runtime.MetadataQuery import bulk_load_metadata
from philologic.runtime.reports.collocation import get_word_groups


# ----------------------------- Hit bagging ------------------------------

def _resolve_attr_target(colloc_dir: str, attribute: str, attribute_value: str) -> int:
    attr_vocab = np.load(
        os.path.join(colloc_dir, f"attr_{attribute}_vocab.npy"), allow_pickle=True
    )
    for i, v in enumerate(attr_vocab):
        if v == attribute_value:
            return i
    return -1


def _build_excluded_mask(
    v_blob: bytes, v_offsets: np.ndarray, filter_set: set, n_vocab: int,
) -> np.ndarray:
    """Mark vocab ids whose name appears in ``filter_set``.

    Naive iteration over all ``n_vocab`` entries dominates on surface-form
    corpora where the vocab can hit millions of entries; the filter set is
    tiny (the query word + a few attribute variants). Bin candidate tids by
    byte length once, then for each filter word only scan tids of matching
    length and break on the first content match.
    """
    excluded = np.zeros(n_vocab, dtype=bool)
    if not filter_set:
        return excluded
    filter_bytes = [s.encode("utf-8") for s in filter_set]
    lengths = (v_offsets[1:] - v_offsets[:-1]).astype(np.int64)
    unique_lens = {len(fb) for fb in filter_bytes}
    tids_by_len = {L: np.where(lengths == L)[0] for L in unique_lens}
    for fb in filter_bytes:
        L = len(fb)
        for tid in tids_by_len[L]:
            s = int(v_offsets[tid])
            if v_blob[s:s + L] == fb:
                excluded[tid] = True
                break  # vocab entries are unique — at most one match per filter word
    return excluded


def _build_hit_bags(
    db: DB,
    db_path: str,
    q: str,
    count_lemmas: bool,
    attribute: Optional[str],
    attribute_value: Optional[str],
    metadata: Dict[str, str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bytes, np.ndarray]:
    """Run the query, return (flat_ids, indptr, years, vocab_blob, vocab_offsets).

    The bags are returned in CSR-style flat form: ``flat_ids`` is a 1-D array
    of count-vocab IDs concatenated across all hits, ``indptr`` gives the
    bag boundaries (``indptr[i]:indptr[i+1]`` is bag ``i``). Unmatched hits
    get zero-width bags; the caller is expected to drop them via the year
    filter (year == -1 for unmatched). Identity-filtered words and, if an
    attribute filter is set, off-attribute tokens are excluded.
    """
    hits = db.query(q, "single_term", "", raw_results=True, raw_bytes=True, **metadata)
    while not os.path.exists(f"{hits.filename}.terms"):
        time.sleep(0.05)
    hits.finish()
    if len(hits) == 0:
        return (
            np.array([], dtype=np.uint32),
            np.array([0], dtype=np.int64),
            np.array([], dtype=np.int32),
            b"",
            np.array([0], dtype=np.uint64),
        )

    # Identity filter from .terms file
    query_words: List[str] = []
    for group in get_word_groups(f"{hits.filename}.terms"):
        query_words.extend(group)
    filter_set = set(query_words)
    if attribute is not None:
        filter_set.update(f"{w}:{attribute}:{attribute_value}" for w in query_words)
        filter_set.add(f"{q}:{attribute}:{attribute_value}")

    with open(hits.filename, "rb") as f:
        raw = f.read()
    all_hits = np.frombuffer(raw, dtype=np.uint32).reshape(-1, hits.length)

    colloc_dir = os.path.join(db_path, "collocations")
    sent_keys_s24 = np.load(os.path.join(colloc_dir, "sent_keys_s24.npy"), mmap_mode="r")
    sent_offsets = np.load(os.path.join(colloc_dir, "sent_offsets.npy"), mmap_mode="r")
    if count_lemmas:
        count_ids = np.load(os.path.join(colloc_dir, "attr_lemma_ids.npy"), mmap_mode="r")
        v_offsets = np.load(os.path.join(colloc_dir, "attr_lemma_vocab_offsets.npy"), mmap_mode="r")
        with open(os.path.join(colloc_dir, "attr_lemma_vocab_strings.bin"), "rb") as f:
            v_blob = f.read()
    else:
        count_ids = np.load(os.path.join(colloc_dir, "token_ids.npy"), mmap_mode="r")
        v_offsets = np.load(os.path.join(colloc_dir, "vocab_offsets.npy"), mmap_mode="r")
        with open(os.path.join(colloc_dir, "vocab_strings.bin"), "rb") as f:
            v_blob = f.read()

    n_vocab = len(v_offsets) - 1
    excluded = _build_excluded_mask(v_blob, v_offsets, filter_set, n_vocab)

    hit_keys_be = np.ascontiguousarray(all_hits[:, :6].astype(">u4"))
    hit_arr = hit_keys_be.view("S24").ravel()
    n_sents = len(sent_keys_s24)
    sent_idx = np.searchsorted(sent_keys_s24, hit_arr)
    matched = (sent_idx < n_sents) & (sent_keys_s24[np.clip(sent_idx, 0, n_sents - 1)] == hit_arr)

    field_obj_index, metadata_cache = bulk_load_metadata(db, ["year"])["year"]
    all_years = _extract_years(all_hits, field_obj_index, metadata_cache).copy()
    # Force unmatched hits to year=-1 so the downstream year filter naturally
    # drops them alongside the year-missing ones.
    all_years[~matched] = -1

    # Upper bound on flat token count: sum of matched sentence sizes, before
    # the exclusion filter (which can only remove tokens, never add). Cheap
    # to compute via vectorized numpy.
    si_safe = np.minimum(sent_idx, len(sent_offsets) - 2)
    sizes = (sent_offsets[si_safe + 1] - sent_offsets[si_safe]).astype(np.int64)
    sizes = np.where(matched, sizes, 0)
    max_total = int(sizes.sum())

    if attribute is not None:
        attr_ids_mmap = np.load(
            os.path.join(colloc_dir, f"attr_{attribute}_ids.npy"), mmap_mode="r"
        )
        target_attr_id = _resolve_attr_target(colloc_dir, attribute, attribute_value or "")
        flat_ids, indptr = _bags_kernel_attr(
            sent_idx, sent_offsets, count_ids,
            attr_ids_mmap, target_attr_id,
            excluded, matched, max_total,
        )
    else:
        flat_ids, indptr = _bags_kernel(
            sent_idx, sent_offsets, count_ids,
            excluded, matched, max_total,
        )

    return flat_ids, indptr, all_years, v_blob, np.asarray(v_offsets)


def _extract_years(
    all_hits: np.ndarray, field_obj_index: int, metadata_cache: Dict,
) -> np.ndarray:
    """Vectorized year lookup per hit.

    Pre-parses the metadata cache once (the slow ``str→int`` conversion) and
    uses fancy indexing into a flat ``year_by_doc`` array when the year field
    is doc-level (``field_obj_index == 1``, the common case). Falls back to a
    dict-based per-hit lookup for deeper-scoped year fields.
    """
    n = len(all_hits)
    if n == 0:
        return np.array([], dtype=np.int32)
    parsed: Dict[tuple, int] = {}
    for k, v in metadata_cache.items():
        try:
            parsed[k] = int(str(v).strip()[:4]) if v is not None else -1
        except (ValueError, TypeError):
            parsed[k] = -1

    if field_obj_index == 1:
        max_doc = 0
        for (k,) in parsed.keys():
            if k > max_doc:
                max_doc = k
        hit_max = int(all_hits[:, 0].max())
        size = max(max_doc, hit_max) + 1
        year_by_doc = np.full(size, -1, dtype=np.int32)
        for (k,), v in parsed.items():
            if 0 <= k < size:
                year_by_doc[k] = v
        return year_by_doc[all_hits[:, 0].astype(np.int64)]

    years = np.empty(n, dtype=np.int32)
    for i in range(n):
        prefix = tuple(int(x) for x in all_hits[i, :field_obj_index])
        years[i] = parsed.get(prefix, -1)
    return years


# ----------------------------- Helpers ------------------------------

def _smooth(arr: np.ndarray, win: int = 5) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float64)
    for i in range(len(arr)):
        lo = max(0, i - win // 2)
        hi = min(len(arr), i + win // 2 + 1)
        out[i] = arr[lo:hi].mean()
    return out


def _vocab_name(tid: int, v_blob: bytes, v_offsets: np.ndarray, count_lemmas: bool) -> str:
    s = v_blob[int(v_offsets[tid]):int(v_offsets[tid + 1])].decode("utf-8", errors="replace")
    if count_lemmas and s.startswith("lemma:"):
        s = s[6:]
    return s


@numba.jit(nopython=True, nogil=True, cache=True)
def _bags_kernel(
    sent_idx: np.ndarray, sent_offsets: np.ndarray, count_ids: np.ndarray,
    excluded: np.ndarray, matched: np.ndarray, max_total: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Walk matched hits, write each hit's filtered sentence tokens into a
    pre-allocated flat buffer, and record bag boundaries in ``indptr``.

    Replaces the per-hit Python loop in ``_build_hit_bags`` that allocated
    one numpy array per bag. Unmatched hits get zero-width bags so indices
    stay aligned with the hit/year arrays — caller drops them in the year
    filter below.
    """
    n = len(matched)
    flat = np.empty(max_total, dtype=count_ids.dtype)
    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    pos = 0
    for i in range(n):
        if matched[i]:
            si = sent_idx[i]
            s_start = sent_offsets[si]
            s_end = sent_offsets[si + 1]
            for k in range(s_start, s_end):
                tid = count_ids[k]
                if not excluded[tid]:
                    flat[pos] = tid
                    pos += 1
        indptr[i + 1] = pos
    return flat[:pos], indptr


@numba.jit(nopython=True, nogil=True, cache=True)
def _reorder_bags_kernel(
    flat_ids: np.ndarray, indptr: np.ndarray, order: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Permute and shrink bags in flat form.

    ``order`` is a list of source bag indices to keep, in the desired output
    order. Returns new (flat_ids, indptr) with bags in that order.
    """
    n_out = len(order)
    sizes = np.empty(n_out, dtype=np.int64)
    for new_i in range(n_out):
        old_i = order[new_i]
        sizes[new_i] = indptr[old_i + 1] - indptr[old_i]
    new_indptr = np.empty(n_out + 1, dtype=np.int64)
    new_indptr[0] = 0
    for i in range(n_out):
        new_indptr[i + 1] = new_indptr[i] + sizes[i]
    total = new_indptr[n_out]
    new_flat = np.empty(total, dtype=flat_ids.dtype)
    for new_i in range(n_out):
        old_i = order[new_i]
        src = indptr[old_i]
        dst = new_indptr[new_i]
        sz = sizes[new_i]
        for k in range(sz):
            new_flat[dst + k] = flat_ids[src + k]
    return new_flat, new_indptr


def _reorder_bags(
    flat_ids: np.ndarray, indptr: np.ndarray, order: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Permute+shrink bags by an index array, returning new flat form.

    Used after building bags for ALL hits to drop year-invalid bags and
    reorder the survivors by year.
    """
    if len(order) == 0:
        return np.empty(0, dtype=flat_ids.dtype), np.array([0], dtype=np.int64)
    return _reorder_bags_kernel(flat_ids, indptr, order.astype(np.int64))


@numba.jit(nopython=True, nogil=True, cache=True)
def _bags_kernel_attr(
    sent_idx: np.ndarray, sent_offsets: np.ndarray, count_ids: np.ndarray,
    attr_ids: np.ndarray, target_attr_id: int,
    excluded: np.ndarray, matched: np.ndarray, max_total: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Same as ``_bags_kernel`` but additionally filters tokens by attribute
    (used when the user has e.g. a POS filter on the query)."""
    n = len(matched)
    flat = np.empty(max_total, dtype=count_ids.dtype)
    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    pos = 0
    for i in range(n):
        if matched[i]:
            si = sent_idx[i]
            s_start = sent_offsets[si]
            s_end = sent_offsets[si + 1]
            for k in range(s_start, s_end):
                if attr_ids[k] != target_attr_id:
                    continue
                tid = count_ids[k]
                if not excluded[tid]:
                    flat[pos] = tid
                    pos += 1
        indptr[i + 1] = pos
    return flat[:pos], indptr


@numba.jit(nopython=True, nogil=True, cache=True)
def _df_kernel(flat_ids: np.ndarray, indptr: np.ndarray, n_vocab: int) -> np.ndarray:
    """Document-frequency count, dedup'd within each bag.

    Reusable ``seen`` buffer is reset only for the ids we actually touched
    in each bag — much cheaper than allocating a fresh boolean array per
    iteration.
    """
    df = np.zeros(n_vocab, dtype=np.int64)
    seen = np.zeros(n_vocab, dtype=np.bool_)
    n_bags = len(indptr) - 1
    for hi in range(n_bags):
        start = indptr[hi]
        end = indptr[hi + 1]
        for k in range(start, end):
            tid = flat_ids[k]
            if not seen[tid]:
                seen[tid] = True
                df[tid] += 1
        for k in range(start, end):
            seen[flat_ids[k]] = False
    return df


def _accumulate_df(flat_ids: np.ndarray, indptr: np.ndarray, n_vocab: int) -> np.ndarray:
    """Compute per-bag deduplicated document frequency for every vocab id."""
    if flat_ids.size == 0:
        return np.zeros(n_vocab, dtype=np.int64)
    return _df_kernel(flat_ids, indptr, n_vocab)


@numba.jit(nopython=True, nogil=True, cache=True)
def _flat_to_sparse_cols(
    flat_ids: np.ndarray, indptr: np.ndarray, vocab_to_col: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map flat bag tokens to candidate-column indices, deduplicating per bag.

    Produces ``(cols, csr_indptr)`` directly suitable for ``scipy.sparse.csr_matrix``
    construction. Non-candidate tokens (``vocab_to_col == -1``) and duplicates
    within a bag are skipped — the per-bag ``seen[]`` buffer is reset only for
    the columns we touched (same trick as ``_df_kernel``).
    """
    n_bags = len(indptr) - 1
    cols = np.empty(len(flat_ids), dtype=np.int32)
    out_indptr = np.empty(n_bags + 1, dtype=np.int64)
    out_indptr[0] = 0
    K = int(vocab_to_col.max()) + 1 if len(vocab_to_col) > 0 else 0
    seen = np.zeros(max(K, 1), dtype=np.bool_)
    touched = np.empty(max(len(flat_ids), 1), dtype=np.int32)
    pos = 0
    for bi in range(n_bags):
        n_touched = 0
        for k in range(indptr[bi], indptr[bi + 1]):
            col = vocab_to_col[flat_ids[k]]
            if col >= 0 and not seen[col]:
                seen[col] = True
                touched[n_touched] = col
                n_touched += 1
                cols[pos] = col
                pos += 1
        for j in range(n_touched):
            seen[touched[j]] = False
        out_indptr[bi + 1] = pos
    return cols[:pos], out_indptr


@numba.jit(nopython=True, nogil=True, cache=True)
def _intensity_kernel(
    flat_ids: np.ndarray, indptr: np.ndarray, bin_of_hit: np.ndarray,
    thread_mask: np.ndarray, n_bins: int,
) -> np.ndarray:
    """Per-bin count of bags containing *any* id in ``thread_mask``.

    A bag contributes 1 to its year-bin's count if it intersects the thread's
    word set — multiplicity within a bag doesn't compound. Breaks on the
    first match to avoid scanning the rest of the bag.
    """
    intensity = np.zeros(n_bins, dtype=np.float64)
    n_bags = len(indptr) - 1
    for hi in range(n_bags):
        start = indptr[hi]
        end = indptr[hi + 1]
        bi = bin_of_hit[hi]
        for k in range(start, end):
            if thread_mask[flat_ids[k]]:
                intensity[bi] += 1.0
                break
    return intensity


def _select_candidates(
    flat_ids: np.ndarray, indptr: np.ndarray,
    v_blob: bytes, v_offsets: np.ndarray,
    count_lemmas: bool, stopwords: set, df_floor_pct: float = 0.005,
    max_df_pct: float = 0.30, safety_cap: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (candidate vocab ids, per-id counts, raw_K before cap).

    ``counts`` here is per-bag document frequency (each id counts once per
    bag regardless of in-bag multiplicity) — that matches what the c-TF-IDF
    ranking later wants as the term-frequency component.
    """
    n_hits = len(indptr) - 1
    n_vocab = len(v_offsets) - 1
    df_int = _accumulate_df(flat_ids, indptr, n_vocab)
    df = df_int.astype(np.float64)
    counts = df  # df *is* the count under per-bag dedup semantics
    df_floor = max(3, int(n_hits * df_floor_pct))
    df_ceiling = int(n_hits * max_df_pct)
    raw_candidates = np.where((df >= df_floor) & (df <= df_ceiling))[0]
    # Token filter: length ≥ 2 and not stopword. The stopword set is the single
    # exclusion authority — the user's UI/config-driven filter list
    # (build_filter_list in the collocation report), shared with the frequency
    # and comparison views, so per-corpus customization Just Works and all views
    # exclude consistently. (Numerals/OCR fragments belong there, not here.)
    filtered = []
    for cid in raw_candidates:
        name = _vocab_name(int(cid), v_blob, v_offsets, count_lemmas)
        if len(name) < 2:
            continue
        if name in stopwords:
            continue
        filtered.append(int(cid))
    cids = np.array(filtered, dtype=np.int64)
    raw_K = len(cids)
    if raw_K > safety_cap:
        # Keep the most-frequent ones — only fires for very large corpora.
        cids = cids[np.argsort(counts[cids])[::-1][:safety_cap]]
    return cids, counts, raw_K


def _build_distance(
    flat_ids: np.ndarray, indptr: np.ndarray,
    candidate_ids: np.ndarray, n_vocab: int,
) -> Optional[np.ndarray]:
    """NPMI distance matrix over the candidate words.

    Distance = 1 − NPMI for co-occurring pairs, 1.0 (no edge) otherwise. NPMI
    (normalized pointwise mutual information) is bounded to [−1, 1] and already
    encodes co-occurrence strength normalized for word frequency — its
    magnitude alone separates real associations from chance. An earlier
    version gated edges through a chi²/Benjamini-Hochberg-FDR significance
    test; an A/B over many queries showed it produced byte-identical
    clusterings, so it was removed as redundant machinery.
    """
    n_hits = len(indptr) - 1
    K = len(candidate_ids)
    if K < 10:
        return None
    # Sparse CSR cooc matrix: skip the dense hits_mat allocation entirely.
    # hits_mat would be ~1% dense at our typical K and bag sizes; sparse
    # matmul exploits that and also avoids the n_hits × K × 4-byte buffer.
    # The K × K cooc result is small (sub-MB), safe to densify.
    vocab_to_col = np.full(n_vocab, -1, dtype=np.int32)
    vocab_to_col[candidate_ids] = np.arange(K, dtype=np.int32)
    cols, csr_indptr = _flat_to_sparse_cols(flat_ids, indptr, vocab_to_col)
    mat = csr_matrix(
        (np.ones(len(cols), dtype=np.float32), cols, csr_indptr),
        shape=(n_hits, K),
    )
    cooc = (mat.T @ mat).toarray().astype(np.float64)
    # df_local = number of bags containing each column. Since each col appears
    # at most once per bag (dedup in _flat_to_sparse_cols), it equals cooc's
    # diagonal — no need for a separate sum.
    df_local = np.diag(cooc).copy()
    N = float(n_hits)

    a_mat = cooc.copy()
    np.fill_diagonal(a_mat, 0.0)
    expected = (df_local[:, None] * df_local[None, :]) / N
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((N * a_mat) / np.where(expected > 0, df_local[:, None] * df_local[None, :], 1.0))
        log_p_joint = np.log(np.where(a_mat > 0, a_mat / N, 1.0))
        npmi = np.where(a_mat > 0, pmi / -log_p_joint, 0.0)
    npmi = np.clip(npmi, -1.0, 1.0)

    dist = np.where(npmi > 0, 1.0 - npmi, 1.0)
    np.fill_diagonal(dist, 0.0)
    return dist


def _build_graph(
    thread_records: List[Dict], kept_clusters: Dict[int, List[int]],
    candidate_ids: np.ndarray, dist: np.ndarray, counts: Dict[int, float],
    v_blob: bytes, v_offsets: np.ndarray, count_lemmas: bool,
    knn_intra: int = 6, knn_inter: int = 2, max_nodes: int = 500,
) -> Dict:
    """Spatial twin of the streamgraph: the thread communities as a network.

    Two node tiers, both colored by thread id (so the network matches the
    streamgraph):
      - members: words that landed in a surviving thread (``member: True``).
      - context: every other candidate word, attached to its *highest-NPMI-
        affinity* thread — even though it never formally clustered, it still
        co-occurs most with one community, so it can be placed there
        ("member: False"). Words with zero affinity to any thread are dropped
        (genuinely unplaceable). Context words are ordered by affinity so a
        UI slider can expand the view outward from the cluster cores.

    ``n_members`` (the count of tier-1 nodes) is returned as the slider floor.

    Edges are a per-node k-nearest-neighbour selection over NPMI strength with
    intra-/cross-cluster budgets kept separate: up to ``knn_intra`` same-thread
    links (these make a community cohere) and only ``knn_inter`` cross-thread
    bridges. Without that split a single query word's collocate graph has
    enough weak cross-links to collapse every community into one hairball.
    Each edge carries an ``intra`` flag so the layout can spring same-thread
    links harder than bridges.
    """
    col_of = {int(t): i for i, t in enumerate(candidate_ids)}
    K = len(candidate_ids)
    # npmi[i, j] = co-occurrence strength (0 where there is no edge).
    npmi = 1.0 - dist
    np.fill_diagonal(npmi, 0.0)

    nodes: List[Dict] = []
    node_vocab: List[int] = []      # node index → vocab id
    node_index: Dict[int, int] = {}  # vocab id → node index
    member_set: set = set()
    cluster_cols: Dict[int, List[int]] = {}  # thread id → member col indices

    # Tier 1 — thread members.
    for t in thread_records:
        cid = t["_cid"]
        tid = t["id"]
        cluster_cols.setdefault(tid, [])
        for w in kept_clusters[cid]:
            if w in node_index:
                continue
            node_index[w] = len(nodes)
            node_vocab.append(w)
            nodes.append({
                "word": _vocab_name(w, v_blob, v_offsets, count_lemmas),
                "cluster": tid,
                "weight": int(counts[w]),
                "member": True,
            })
            member_set.add(w)
            cluster_cols[tid].append(col_of[w])
    n_members = len(nodes)

    # Tier 2 — context words, each attached to its strongest-affinity thread.
    cluster_col_arr = {tid: np.array(cols, dtype=np.int64)
                       for tid, cols in cluster_cols.items()}
    context: List[Tuple[float, int, int]] = []  # (affinity, vocab_id, thread_id)
    for ci in range(K):
        w = int(candidate_ids[ci])
        if w in member_set:
            continue
        row = npmi[ci]
        best_tid, best_aff = None, 0.0
        for tid, mcols in cluster_col_arr.items():
            aff = float(row[mcols].sum())
            if aff > best_aff:
                best_aff, best_tid = aff, tid
        if best_tid is not None and best_aff > 0.0:
            context.append((best_aff, w, best_tid))
    context.sort(key=lambda x: -x[0])
    for aff, w, tid in context[:max(0, max_nodes - n_members)]:
        node_index[w] = len(nodes)
        node_vocab.append(w)
        nodes.append({
            "word": _vocab_name(w, v_blob, v_offsets, count_lemmas),
            "cluster": tid,
            "weight": int(counts[w]),
            "member": False,
        })

    # Anchor strength per node: summed NPMI to its cluster's MEMBER nodes only
    # (excluding tier-2 context). Restricting to members makes this identical to
    # the per-sense centrality used for the legend's word ranking, so the map's
    # node sizes and labels match the legend's anchor words exactly. The UI sizes
    # nodes by this (how core a word is to its sense, not raw frequency);
    # ``weight`` (frequency) is kept for tooltips. npmi's diagonal is 0.
    cluster_of = [n["cluster"] for n in nodes]
    member_of = np.array([n["member"] for n in nodes])
    cols = np.array([col_of[w] for w in node_vocab], dtype=np.int64)
    cluster_arr = np.array(cluster_of)
    for i, n in enumerate(nodes):
        same = cols[(cluster_arr == n["cluster"]) & member_of]  # cluster members
        n["anchor"] = round(float(npmi[cols[i], same].sum()), 4)

    # Edges — KNN over the full node set, intra/cross budgeted separately.
    seen: set = set()
    edges: List[Dict] = []
    for ni in range(len(node_vocab)):
        d = dist[cols[ni], cols]  # distances to every other node word
        intra = inter = 0
        for nj in np.argsort(d):
            nj = int(nj)
            if nj == ni:
                continue
            if d[nj] >= 1.0:
                break  # sorted ascending — the rest are non-edges
            same = cluster_of[ni] == cluster_of[nj]
            if same and intra >= knn_intra:
                continue
            if not same and inter >= knn_inter:
                continue
            a, b = (ni, nj) if ni < nj else (nj, ni)
            if (a, b) not in seen:
                seen.add((a, b))
                edges.append({"source": a, "target": b,
                              "weight": round(1.0 - float(d[nj]), 4),
                              "intra": same})
            if same:
                intra += 1
            else:
                inter += 1
            if intra >= knn_intra and inter >= knn_inter:
                break
    return {"nodes": nodes, "edges": edges, "n_members": n_members}


# ---------- Per-query intermediates cache ----------
# Disk-backed cache for the expensive query-dependent intermediates
# (bags + NPMI distance matrix). Lets the min-words-per-thread slider rerun
# in ~300 ms instead of ~1.6 s on a big query. All gunicorn workers share
# the same cache files, so cross-worker reruns also hit cache.
#
# Cache files live in the existing ``hitlists/`` directory (already mode 777
# in every corpus install — gunicorn runs as www-data and needs to write
# hitlists there) with a ``.thread.npz`` suffix to keep them visually
# distinguishable from hitlist binaries.
#
# Cache key includes everything that affects the cached arrays:
#   - q, count_lemmas, attribute, attribute_value, metadata: feed _build_hit_bags
#   - stopwords: feed _select_candidates → candidate_ids → dist
# It does NOT include HDBSCAN params (min_cluster_size*, persistence floor,
# etc.) — those only affect the downstream clustering, which runs fresh
# every call. That's the whole point: the slider changes only those.


def _thread_cache_key(
    q: str, count_lemmas: bool, attribute: Optional[str],
    attribute_value: Optional[str], metadata: Dict[str, str],
    stopwords: Optional[set],
) -> str:
    h = hashlib.sha1()
    h.update(b"hlex2\0")  # schema tag: bump when the cached array set changes
    h.update(q.encode("utf-8"))
    h.update(b"\0L1" if count_lemmas else b"\0L0")
    h.update(b"\0a")
    if attribute:
        h.update(attribute.encode("utf-8"))
    h.update(b"\0v")
    if attribute_value:
        h.update(attribute_value.encode("utf-8"))
    h.update(b"\0m")
    for k in sorted(metadata.keys()):
        h.update(k.encode("utf-8"))
        h.update(b"=")
        h.update(str(metadata[k]).encode("utf-8"))
        h.update(b"\0")
    h.update(b"\0s")
    if stopwords:
        for w in sorted(stopwords):
            h.update(w.encode("utf-8"))
            h.update(b"\0")
    return h.hexdigest()


def _thread_cache_path(db_path: str, key: str) -> str:
    # Sit alongside hitlist files (already mode 777, world-writable so
    # gunicorn-as-www-data can write to it). The .thread.npz suffix keeps
    # our cache files visually distinguishable from hitlist binaries.
    return os.path.join(db_path, "hitlists", f"{key}.thread.npz")


def _load_intermediates(path: str) -> Optional[Dict[str, np.ndarray]]:
    """Load cached intermediates from disk. Returns None on miss or corruption."""
    if not os.path.exists(path):
        return None
    try:
        f = np.load(path, allow_pickle=False)
        out = {k: f[k] for k in f.files}
        f.close()
        return out
    except Exception:
        # Partial write, format mismatch, etc. — caller will recompute and
        # overwrite. Don't crash the request on a stale cache file.
        return None


def _save_intermediates(path: str, **arrays: np.ndarray) -> None:
    """Atomically write intermediates to disk (write tmp + rename).

    Errors are swallowed: a failed cache write must not break detection.
    The target directory (``hitlists/``) already exists with permissive
    perms in every install, so no dir-creation logic is needed.
    """
    tmp_path = None
    try:
        base = path[:-4] if path.endswith(".npz") else path
        tmp_base = f"{base}.tmp.{os.getpid()}"
        tmp_path = tmp_base + ".npz"
        np.savez(tmp_base, **arrays)
        os.rename(tmp_path, path)
    except Exception:
        # Cache failure should never break a query — silently skip.
        try:
            if tmp_path is not None and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


def _load_vocab(db_path: str, count_lemmas: bool) -> Tuple[bytes, np.ndarray]:
    """Read the vocab blob + offsets from disk (mmap'd, near-free)."""
    colloc_dir = os.path.join(db_path, "collocations")
    if count_lemmas:
        v_offsets = np.load(os.path.join(colloc_dir, "attr_lemma_vocab_offsets.npy"), mmap_mode="r")
        with open(os.path.join(colloc_dir, "attr_lemma_vocab_strings.bin"), "rb") as f:
            v_blob = f.read()
    else:
        v_offsets = np.load(os.path.join(colloc_dir, "vocab_offsets.npy"), mmap_mode="r")
        with open(os.path.join(colloc_dir, "vocab_strings.bin"), "rb") as f:
            v_blob = f.read()
    return v_blob, np.asarray(v_offsets)


# ----------------------------- HyperLex sense induction ------------------------------
# The collocate graph (NPMI affinities) is decomposed by Véronis-style hub
# detection: each sense is anchored by a hub collocate, every other word joins
# the hub it co-occurs with most, and words tied to no hub are left unplaced
# (the honest tail). Pure NumPy — no graph/clustering library needed.

_HYPERLEX_BASE_FLOOR = 0.22   # standard NPMI "real association" cutoff (not fitted)
_HYPERLEX_MAX_HUBS = 8        # internal cap on number of senses


def _hyperlex(
    dist: np.ndarray, edge_floor: float, min_neighbors: int, max_hubs: int,
    grow: bool = False,
) -> Tuple[List[int], np.ndarray]:
    """Decompose the NPMI graph into hub-anchored senses.

    Returns ``(hub_cols, labels)`` where ``labels[i]`` is the sense index of
    candidate column ``i`` (or -1 if unplaced). Steps:
      1. NPMI affinity; an edge exists iff ``npmi >= edge_floor``.
      2. Hubs: repeatedly take the highest-degree still-available node, then
         remove it and its strong neighbours from the pool. This is the
         percolation guard — a hub absorbs its neighbourhood, so no two hubs
         grow out of one dense region (unlike connected-components, which a
         single hub word fuses into a giant blob). Stop at ``max_hubs``.
      3. Each non-hub word joins the hub it has the strongest direct edge to
         (``>= edge_floor``); otherwise it stays unplaced (the hub-core).
      4. If ``grow``: attach each still-unplaced word to the sense it has its
         strongest real link with — over that sense's *members*, not just its
         hub (a word can belong via members other than the hub). Same floor
         gate, so no new parameter and the honest tail survives. ``_auto_floor``
         calls with grow=False (it sizes the cores); detection grows.
    """
    K = dist.shape[0]
    npmi = np.where(dist < 1.0, 1.0 - dist, 0.0)
    np.fill_diagonal(npmi, 0.0)
    adj = npmi >= edge_floor            # boolean adjacency over "real" edges
    degree = adj.sum(axis=1)

    available = np.ones(K, dtype=bool)
    hubs: List[int] = []
    for node in np.argsort(-degree):    # highest-degree first
        if len(hubs) >= max_hubs:
            break
        node = int(node)
        if not available[node] or degree[node] < min_neighbors:
            continue
        hubs.append(node)
        available[node] = False
        available[adj[node]] = False    # claim the hub's strong neighbourhood

    labels = np.full(K, -1, dtype=np.int64)
    if not hubs:
        return hubs, labels
    hub_arr = np.array(hubs, dtype=np.int64)
    for hi, h in enumerate(hubs):
        labels[h] = hi
    # Hub-core: argmax NPMI to any hub, thresholded at floor. (Véronis used a
    # minimum-spanning tree, but its transitive chains re-percolate on dense
    # graphs; direct affinity keeps the honest tail.)
    aff = npmi[:, hub_arr]
    best = np.argmax(aff, axis=1)
    best_aff = aff[np.arange(K), best]
    place = (labels < 0) & (best_aff >= edge_floor)
    labels[place] = best[place]

    if grow:
        # Whole-sense growth: a word joins the sense containing the single member
        # it co-occurs with most strongly (>= edge_floor). One quantity — the
        # strongest member-link — is both the gate and the which-sense tie-break,
        # so there's no new threshold; the hub is just one special member.
        members: Dict[int, List[int]] = {}
        for col in range(K):
            if labels[col] >= 0:
                members.setdefault(int(labels[col]), []).append(col)
        member_cols = {s: np.asarray(c, dtype=np.int64) for s, c in members.items()}
        for w in range(K):
            if labels[w] >= 0:
                continue
            best_s, best_link = -1, edge_floor
            for s, mcols in member_cols.items():
                link = float(npmi[w, mcols].max())
                if link >= best_link:
                    best_link, best_s = link, s
            if best_s >= 0:
                labels[w] = best_s
    return hubs, labels


def _auto_floor(
    dist: np.ndarray, min_neighbors: int, max_hubs: int,
    base: float = _HYPERLEX_BASE_FLOOR,
) -> float:
    """Self-tuning NPMI edge floor: keep the principled base, raise ONLY on over-grab.

    Internal tightness metrics (modularity / cohesion / silhouette) all reward
    over-tightening monotonically, so maximizing one collapses rich
    decompositions into a few 5-word fragments. Instead: scan upward from the
    base and take the lowest floor whose biggest sense fits a readable cap
    (= the over-grab guard), as long as ≥ ceil(max_hubs/2) senses survive. Most
    queries stay at the base; only a word dominated by one generic hub tightens.
    """
    K = dist.shape[0]
    band_cap = max(40.0, 0.08 * K)
    min_senses = max(2, (max_hubs + 1) // 2)
    floors = [round(base + 0.02 * i, 2) for i in range(7)]  # base .. base+0.12
    fallback, fallback_big = base, None
    for f in floors:
        _, labels = _hyperlex(dist, f, min_neighbors, max_hubs)
        placed = labels[labels >= 0]
        if placed.size == 0:
            continue
        sizes = np.bincount(placed)
        n_senses = int((sizes > 0).sum())
        biggest = int(sizes.max())
        if n_senses < min_senses:
            continue
        if biggest <= band_cap:
            return f
        if fallback_big is None or biggest < fallback_big:
            fallback, fallback_big = f, biggest
    return fallback


def _sense_cohesion(member_cols: List[int], dist: np.ndarray) -> float:
    """Mean within-sense NPMI similarity (non-edges count as 0). Transparent
    per-sense quality score that replaces HDBSCAN's opaque persistence."""
    m = len(member_cols)
    if m < 2:
        return 0.0
    cols = np.array(member_cols, dtype=np.int64)
    sub = dist[np.ix_(cols, cols)]
    sim = np.where(sub < 1.0, 1.0 - sub, 0.0)
    iu = np.triu_indices(m, k=1)
    return float(sim[iu].mean()) if iu[0].size else 0.0


def detect_threads(
    db: DB,
    db_path: str,
    q: str,
    count_lemmas: bool,
    attribute: Optional[str],
    attribute_value: Optional[str],
    metadata: Dict[str, str],
    *,
    stopwords: Optional[set] = None,
    top_n_threads: Optional[int] = None,
    # User-facing "number of themes" knob: the number of hubs (= senses) to
    # induce at the fixed floor — more themes = finer senses (hubs are nested, so
    # raising the count appends the next-strongest hub region and a broad sense
    # splits as members re-flow to it); fewer = coarser. None → a sensible
    # default. The UI dropdown is populated from available_theme_counts (the
    # achievable range at this floor, capped at max_senses).
    n_clusters_override: Optional[int] = None,
    # HyperLex knobs. edge_floor=None auto-tunes (recommended); hub_min_neighbors
    # is how many strong neighbours a word needs to anchor a sense; max_senses
    # caps the granularity (the most themes the dropdown will offer).
    edge_floor: Optional[float] = None,
    hub_min_neighbors: int = 4,
    max_senses: int = 12,
    include_graph: bool = False,
) -> Dict:
    """End-to-end global-first thread detection (HyperLex)."""
    stop_set = stopwords or set()

    # Try cache first: bags + candidates + NPMI dist are invariant across
    # HDBSCAN-parameter changes (the only thing that varies in a min-words
    # rerun). Cache hit skips ~1.3 s of work on a big query like `woman`.
    cache_key = _thread_cache_key(
        q, count_lemmas, attribute, attribute_value, metadata, stop_set,
    )
    cache_path = _thread_cache_path(db_path, cache_key)
    cached = _load_intermediates(cache_path)

    # The cache key carries a schema tag (see _thread_cache_key), so any hit is
    # already the current schema; "counts_at_cand" is a cheap presence check.
    if cached is not None and "counts_at_cand" in cached:
        flat_ids = cached["flat_ids"]
        indptr = cached["indptr"]
        years = cached["years"]
        candidate_ids = cached["candidate_ids"]
        counts_at_cand = cached["counts_at_cand"]
        raw_K = int(cached["raw_K"])
        dist = cached["dist"]
        v_blob, v_offsets = _load_vocab(db_path, count_lemmas)
    else:
        flat_ids, indptr, years, v_blob, v_offsets = _build_hit_bags(
            db, db_path, q, count_lemmas, attribute, attribute_value, metadata
        )
        if len(indptr) <= 1:
            return {"n_total_hits": 0, "threads": []}

        # Drop bags whose year is -1 (either unmatched-to-sentence or
        # metadata-missing) and reorder by year. Both operations on flat form
        # require gathering bag slices — _reorder_bags handles it.
        yr_mask = years >= 0
        keep = np.where(yr_mask)[0]
        order = keep[np.argsort(years[keep], kind="stable")]
        flat_ids, indptr = _reorder_bags(flat_ids, indptr, order)
        years = years[order]
        if len(indptr) - 1 < 30:
            return {"n_total_hits": len(indptr) - 1, "threads": []}

        candidate_ids, counts_full, raw_K = _select_candidates(
            flat_ids, indptr, v_blob, v_offsets, count_lemmas, stop_set,
        )
        # Store only the K counts we'll ever look up downstream. The full
        # n_vocab array would inflate the cache file by ~38 MB on big corpora
        # for zero functional gain — every consumer does counts[w] only for
        # w in candidate_ids.
        counts_at_cand = counts_full[candidate_ids].astype(np.float64)

        n_vocab = len(v_offsets) - 1
        dist = _build_distance(flat_ids, indptr, candidate_ids, n_vocab=n_vocab)
        if dist is not None:
            # Persist only successful intermediates; degenerate cases stay
            # uncached so they're cheaply recomputed if data later changes.
            _save_intermediates(
                cache_path,
                flat_ids=flat_ids, indptr=indptr, years=years,
                candidate_ids=candidate_ids, counts_at_cand=counts_at_cand,
                raw_K=np.array(raw_K, dtype=np.int64), dist=dist,
            )

    # Per-candidate counts as a vocab-id → value dict for the downstream "n"
    # field (frequency of each word in the result set).
    counts = {int(candidate_ids[i]): float(counts_at_cand[i])
              for i in range(len(candidate_ids))}

    # Derived values — same in both cache hit / miss paths.
    n_hits = len(indptr) - 1
    K = len(candidate_ids)
    yr_min = int(years.min())
    yr_max = int(years.max())
    year_bin = 1  # always yearly resolution; smoothing handles display jitter
    n_bins = (yr_max - yr_min) // year_bin + 1
    # Smoothing window adapts to the corpus span — gentle (no more than 20-year
    # windows). max(3, span // 25) so e.g. a 100-year corpus uses a 4-year
    # window, 200-year uses 8, capped at 20.
    smooth_win = max(3, min(20, (yr_max - yr_min) // 25))
    bin_of_hit = ((years.astype(np.int64) - yr_min) // year_bin).astype(np.int64)
    n_hits_yr = np.bincount(bin_of_hit, minlength=n_bins).astype(np.int64)
    frequency = [[yr_min + i * year_bin, int(n_hits_yr[i])] for i in range(n_bins)]

    empty_result = {
        "n_total_hits": n_hits,
        "year_range": [yr_min, yr_max],
        "year_bin": year_bin,
        "n_bins": n_bins,
        "n_threads": 0,
        "n_detected_threads": 0,
        "vocab_size": K,
        "available_theme_counts": [],
        "threads": [],
        "frequency": frequency,
    }

    if dist is None or K < 10:
        return empty_result

    # ---- HyperLex sense induction (white-box; replaces HDBSCAN) -------------
    # Treat the collocate graph as the graph it is: each sense is anchored by a
    # hub collocate, every other word joins the hub it co-occurs with most, and
    # words tied to no hub are left unplaced (the honest tail). One interpretable
    # knob — the NPMI edge floor — auto-tuned (a principled base of 0.22 that
    # rises only when a single hub over-grabs). No persistence, no EOM/LEAF, no
    # min_cluster_size sweep: a humanist can state the rule in one sentence and
    # inspect why any given word landed where it did.
    floor = (
        float(edge_floor) if edge_floor is not None
        else _auto_floor(dist, hub_min_neighbors, _HYPERLEX_MAX_HUBS)
    )
    # "Number of themes" = how many hubs to induce at this fixed floor. The hubs
    # are deterministic and nested, so raising the count only appends the
    # next-strongest hub region; membership re-flows to the nearest hub, so a
    # broad sense splits into finer ones rather than the senses reshuffling
    # arbitrarily. Find the achievable range once (for the dropdown), then induce
    # at the requested count. (The floor is held fixed across the knob, so only
    # granularity changes — not which associations count.)
    hubs_avail, _ = _hyperlex(dist, floor, hub_min_neighbors, max_senses)
    n_avail = max(2, len(hubs_avail))
    available_counts = list(range(2, n_avail + 1))
    requested = n_clusters_override if n_clusters_override is not None else min(4, n_avail)
    n_req = max(2, min(int(requested), n_avail))
    _, labels = _hyperlex(dist, floor, hub_min_neighbors, n_req, grow=True)

    # Group placed candidate columns into senses; keep both vocab ids (for
    # counts/labels) and column indices (for cohesion + the graph projection).
    kept_clusters: Dict[int, List[int]] = {}
    sense_cols: Dict[int, List[int]] = {}
    for col in range(K):
        s = int(labels[col])
        if s < 0:
            continue
        kept_clusters.setdefault(s, []).append(int(candidate_ids[col]))
        sense_cols.setdefault(s, []).append(col)
    if not kept_clusters:
        return empty_result
    sense_cohesion = {s: _sense_cohesion(cols, dist) for s, cols in sense_cols.items()}

    # Per-thread word counts (also feeds the "n" field in the output).
    thread_word_counts: Dict[int, Dict[int, int]] = {}
    union_words: List[int] = []
    for cid, words in kept_clusters.items():
        thread_word_counts[cid] = {w: int(counts[w]) for w in words}
        union_words.extend(words)
    union_words = sorted(set(union_words))
    cid_order = list(kept_clusters.keys())

    # Bags are already in flat (flat_ids, indptr) form from _build_hit_bags;
    # the intensity kernel can use them directly per thread.
    n_vocab_total = len(v_offsets) - 1
    safe_totals = np.where(n_hits_yr > 0, n_hits_yr, 1).astype(np.float64)
    smooth_bins = max(1, smooth_win // year_bin)

    thread_records: List[Dict] = []
    for cid in cid_order:
        words = kept_clusters[cid]
        # Thread intensity = share of bin hits whose bag intersects thread words.
        # Build a vocab-wide boolean mask and let the kernel scan all bags in
        # one tight loop (early-exit on first hit per bag).
        thread_mask = np.zeros(n_vocab_total, dtype=np.bool_)
        thread_mask[np.asarray(words, dtype=np.int64)] = True
        intensity = _intensity_kernel(flat_ids, indptr, bin_of_hit, thread_mask, n_bins)
        intensity = intensity / safe_totals
        intensity_smooth = _smooth(intensity, win=smooth_bins) if smooth_bins > 1 else intensity
        max_v = float(intensity_smooth.max())
        if max_v <= 0:
            continue

        # Rank thread words by within-sense ANCHOR strength: each word's summed
        # NPMI to the other members of this sense (its graph centrality inside
        # the sense). Pure NPMI — no counts, no corpus-IDF — so it's sense-
        # discriminating (unlike global IDF) and robust to the whole-sense
        # growth (it surfaces the tight core, not the peripheral members the
        # growth adds). The hub lands first naturally, being the most central.
        cols = np.asarray(sense_cols[cid], dtype=np.int64)
        sub = dist[np.ix_(cols, cols)]
        sub_npmi = np.where(sub < 1.0, 1.0 - sub, 0.0)
        np.fill_diagonal(sub_npmi, 0.0)
        centrality = sub_npmi.sum(axis=1)
        order = np.argsort(-centrality)
        name_of = {w: _vocab_name(w, v_blob, v_offsets, count_lemmas) for w in words}
        words_out = [
            {
                "word": name_of[words[int(i)]],
                "n": thread_word_counts[cid][words[int(i)]],
                "score": round(float(centrality[int(i)]), 4),
            }
            for i in order
        ]

        thread_records.append({
            "_cid": cid,  # internal: maps the record back to its sense
            "n_words": len(words),
            "cohesion": round(sense_cohesion[cid], 4),
            "words": words_out,
            "max_intensity": round(max_v, 6),
            "intensity": [round(float(intensity_smooth[bi]), 6) for bi in range(n_bins)],
        })

    # Sort by total share of mass (intensity area), assign ids, build labels.
    thread_records.sort(key=lambda t: -sum(t["intensity"]))
    n_detected = len(thread_records)
    # available_counts was computed at induction time (the achievable hub range
    # at this floor). The theme COUNT is controlled by the hub count above, not
    # by truncation; top_n_threads remains a separate hard cap (rarely set).
    if top_n_threads is not None and top_n_threads > 0:
        thread_records = thread_records[:top_n_threads]
    for i, t in enumerate(thread_records):
        t["id"] = i + 1
        t["label"] = ", ".join(w["word"] for w in t["words"][:5])

    # Optional network projection of the same communities (built before _cid
    # is stripped, so node colors match the streamgraph's thread ids).
    graph = None
    if include_graph:
        graph = _build_graph(
            thread_records, kept_clusters, candidate_ids, dist, counts,
            v_blob, v_offsets, count_lemmas,
        )
    for t in thread_records:
        t.pop("_cid", None)

    result = {
        "n_total_hits": n_hits,
        "year_range": [yr_min, yr_max],
        "year_bin": year_bin,
        "n_bins": n_bins,
        "n_threads": len(thread_records),
        "n_detected_threads": n_detected,
        "vocab_size": K,
        # Theme counts the UI offers in the "Number of themes" dropdown
        # (2 .. senses found); picking fewer truncates to the top senses by mass.
        "available_theme_counts": available_counts,
        "threads": thread_records,
        "frequency": frequency,
    }
    if graph is not None:
        result["graph"] = graph
    return result
