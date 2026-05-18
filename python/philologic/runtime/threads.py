"""Global-first thread detection for collocation evolution.

Identifies persistent "threads" — coherent sets of collocate words representing
distinct uses or contexts of the query word over time.

Algorithm (one HDBSCAN call on all hits, then project to time bins):
  1. Build per-hit collocate bags from the query's hitlist.
  2. Filter candidate vocabulary by:
       - df_floor: word must appear in ≥ 0.5% of hits (min 3),
       - max_df:   word in ≤ 30% of hits (too generic to anchor a thread),
       - length ≥ 2 and not in the user-configured stopword list.
  3. Build an NPMI distance matrix on the kept vocabulary.
  4. HDBSCAN over a min_cluster_size sweep; pick the resolution at the widest
     stable plateau in the cluster-count curve (see _pick).
  5. Thread-validity test — each raw cluster must be BOTH:
       - stable:  HDBSCAN persistence ≥ 0.02 (it actually cohered), and
       - focused: size ≤ 25% of K (it isn't an "everything-else" catch-all).
     These catch disjoint failure modes (see _kept_at).
  6. Project each surviving thread to per-year-bin intensity = share of hits
     in the bin whose bag intersects the thread's vocabulary.
  7. Corpus-IDF c-TF-IDF for per-thread label ranking: each word is scored by
     its result-set count × its rarity in the whole corpus, so a word that is
     frequent in these hits *and* uncommon corpus-wide rises to the top.
     (Fightin' words was tried but, with disjoint clusters, it degenerates to
     plain count-ranking — it can't tell a distinctive word from a globally
     common one.)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from hdbscan import HDBSCAN

from philologic.runtime.DB import DB
from philologic.runtime.phase_shifts import _build_hit_bags


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


def _select_candidates(
    bags: Sequence[np.ndarray], v_blob: bytes, v_offsets: np.ndarray,
    count_lemmas: bool, stopwords: set, df_floor_pct: float = 0.005,
    max_df_pct: float = 0.30, safety_cap: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (candidate vocab ids, per-id counts, raw_K before cap)."""
    n_hits = len(bags)
    n_vocab = len(v_offsets) - 1
    counts = np.zeros(n_vocab, dtype=np.float64)
    df = np.zeros(n_vocab, dtype=np.float64)
    for bag in bags:
        if len(bag) == 0:
            continue
        ids = bag.astype(np.int64)
        counts[ids] += 1
        df[np.unique(ids)] += 1
    df_floor = max(3, int(n_hits * df_floor_pct))
    df_ceiling = int(n_hits * max_df_pct)
    raw_candidates = np.where((df >= df_floor) & (df <= df_ceiling))[0]
    # Token filter: length ≥ 2 and not stopword. The stopword set comes from
    # the user's UI/config-driven filter list (build_filter_list in the
    # collocation report), so per-corpus customization Just Works.
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
    bags: Sequence[np.ndarray], candidate_ids: np.ndarray,
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
    n_hits = len(bags)
    K = len(candidate_ids)
    if K < 10:
        return None
    id_to_col = {int(t): ci for ci, t in enumerate(candidate_ids)}
    hits_mat = np.zeros((n_hits, K), dtype=np.uint8)
    for hi, bag in enumerate(bags):
        for t in bag:
            ci = id_to_col.get(int(t))
            if ci is not None:
                hits_mat[hi, ci] = 1
    cooc = hits_mat.astype(np.float64).T @ hits_mat.astype(np.float64)
    df_local = hits_mat.sum(axis=0).astype(np.float64)
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
    candidate_ids: np.ndarray, dist: np.ndarray, counts: np.ndarray,
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

    # Edges — KNN over the full node set, intra/cross budgeted separately.
    cluster_of = [n["cluster"] for n in nodes]
    cols = np.array([col_of[w] for w in node_vocab], dtype=np.int64)
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
    corpus_idf: Optional[Dict[str, float]] = None,
    corpus_idf_default: float = 1.0,
    top_n_threads: Optional[int] = None,
    min_cluster_size: int = 10,
    min_cluster_size_floor: int = 5,
    min_cluster_size_override: Optional[int] = None,
    min_samples: int = 1,
    abs_persistence_floor: float = 0.02,
    max_cluster_share: float = 0.25,
    include_graph: bool = False,
) -> Dict:
    """End-to-end global-first thread detection."""
    bags, years, v_blob, v_offsets = _build_hit_bags(
        db, db_path, q, count_lemmas, attribute, attribute_value, metadata
    )
    if not bags:
        return {"n_total_hits": 0, "threads": []}

    yr_mask = years >= 0
    bags = [bags[i] for i in range(len(bags)) if yr_mask[i]]
    years = years[yr_mask]
    order = np.argsort(years, kind="stable")
    bags = [bags[i] for i in order]
    years = years[order]
    n_hits = len(bags)
    if n_hits < 30:
        return {"n_total_hits": n_hits, "threads": []}

    yr_min = int(years.min())
    yr_max = int(years.max())
    year_bin = 1  # always yearly resolution; smoothing handles display jitter
    n_bins = (yr_max - yr_min) // year_bin + 1
    # Smoothing window adapts to the corpus span — gentle (no more than 20-year
    # windows). max(3, span // 25) so e.g. a 100-year corpus uses a 4-year
    # window, 200-year uses 8, capped at 20.
    smooth_win = max(3, min(20, (yr_max - yr_min) // 25))

    stop_set = stopwords or set()

    candidate_ids, counts, raw_K = _select_candidates(
        bags, v_blob, v_offsets, count_lemmas, stop_set,
    )
    K = len(candidate_ids)

    bin_of_hit = [(int(yr) - yr_min) // year_bin for yr in years]
    n_hits_yr = np.zeros(n_bins, dtype=np.int64)
    for bi in bin_of_hit:
        n_hits_yr[bi] += 1
    frequency = [[yr_min + i * year_bin, int(n_hits_yr[i])] for i in range(n_bins)]

    empty_result = {
        "n_total_hits": n_hits,
        "year_range": [yr_min, yr_max],
        "year_bin": year_bin,
        "n_bins": n_bins,
        "n_threads": 0,
        "n_detected_threads": 0,
        "vocab_size": K,
        "threads": [],
        "frequency": frequency,
    }

    if K < min_cluster_size * 2:
        return empty_result

    dist = _build_distance(bags, candidate_ids)
    if dist is None:
        return empty_result

    # Adaptive HDBSCAN via a min_cluster_size sweep. Lower mcs → more, finer
    # clusters; higher mcs → fewer, coarser. There is no query-independent
    # "right" mcs, so instead of picking one we sweep the whole range and read
    # the *cluster-count curve*:
    #
    #   - The widest plateau (run of consecutive mcs sharing the same count,
    #     count ≥ 2) is the most *stable* resolution — the count the data
    #     keeps returning across many mcs values. That is the pick.
    #   - Ties in plateau width break toward the run reaching the lower mcs
    #     (the richer decomposition — surfaces a real extra thread rather than
    #     leaving it merged).
    #   - No plateau at all means the count never stabilizes (e.g. it explodes
    #     8→13→18→23); chasing it would over-fragment, so we fall back to the
    #     first mcs (highest, coarsest) that yields ≥ 2 clusters.
    #   - No mcs yields ≥ 2: keep the first 1-cluster result (some queries
    #     genuinely have a single dominant usage).
    #
    # Two-pass cluster-selection: EOM first (Excess of Mass — stable/compact
    # clusters, the right default); if EOM yields < 2, retry with LEAF (the
    # cluster-tree leaves — finer-grained; rescues dense graphs EOM collapses
    # into one huge blob, e.g. surface queries without a POS filter).
    # Thread-validity test: a raw HDBSCAN cluster is a real thread only if it is
    # BOTH stable AND focused. These are two facets of one question ("is this a
    # real thread?"), not an accidental filter stack — an A/B over 727 raw
    # clusters showed they catch near-perfectly disjoint failure modes (3/727
    # overlap):
    #   - stable  (persistence ≥ abs_persistence_floor): kills clusters that
    #     never cohered — HDBSCAN persistence near 0, normal size. ~37% of
    #     raw clusters. This is the dominant filter.
    #   - focused (size ≤ max_cluster_share × K): kills clusters that *did*
    #     clear the persistence floor but cohered into an "everything-else"
    #     catch-all — 30–80% of the vocabulary in one cluster. ~4% of raw
    #     clusters. Real threads top out near 19% of K and these blobs start
    #     near 30%, so the cap sits in a genuine empty gap, not on a knife-edge.
    # You cannot replace one with the other: the blobs reach persistence ~0.06
    # (above many legitimate threads), so no persistence floor catches them
    # without also killing real threads — the two axes are orthogonal.
    size_threshold = max(20, int(K * max_cluster_share))
    floor = max(2, int(min_cluster_size_floor))
    mcs_start = max(floor, int(min_cluster_size))
    mcs_desc = list(range(mcs_start, floor - 1, -1))

    def _is_valid_thread(persistence: float, n_words: int) -> bool:
        """A raw cluster is a real thread iff it is both stable and focused."""
        stable = persistence >= abs_persistence_floor
        focused = n_words <= size_threshold
        return stable and focused

    def _kept_at(mcs_local: int, method: str):
        """Run HDBSCAN at one mcs; return (labels, pers, kept, kept_pers).

        `kept` holds only clusters passing the thread-validity test above.
        """
        try:
            clusterer = HDBSCAN(
                min_cluster_size=mcs_local,
                min_samples=min_samples,
                metric="precomputed",
                cluster_selection_method=method,
            )
            attempt_labels = clusterer.fit_predict(dist)
        except Exception:
            return None, None, {}, {}
        attempt_pers = clusterer.cluster_persistence_
        cluster_ids = sorted(int(x) for x in np.unique(attempt_labels) if x >= 0)
        cluster_words: Dict[int, List[int]] = {}
        for ci, lbl in enumerate(attempt_labels):
            if lbl < 0:
                continue
            cluster_words.setdefault(int(lbl), []).append(int(candidate_ids[ci]))
        attempt_kept: Dict[int, List[int]] = {}
        attempt_kept_pers: Dict[int, float] = {}
        for cid in cluster_ids:
            if _is_valid_thread(attempt_pers[cid], len(cluster_words[cid])):
                attempt_kept[cid] = cluster_words[cid]
                attempt_kept_pers[cid] = float(attempt_pers[cid])
        return attempt_labels, attempt_pers, attempt_kept, attempt_kept_pers

    def _pick(method: str):
        # Sweep the full mcs range, then read the count curve (see above).
        sweep = {mcs: _kept_at(mcs, method) for mcs in mcs_desc}
        counts = {mcs: len(sweep[mcs][2]) for mcs in mcs_desc}
        # Runs of consecutive mcs with equal count.
        runs: List[Tuple[int, List[int]]] = []  # (count, [mcs,...])
        i = 0
        while i < len(mcs_desc):
            j = i
            while j + 1 < len(mcs_desc) and counts[mcs_desc[j + 1]] == counts[mcs_desc[i]]:
                j += 1
            runs.append((counts[mcs_desc[i]], mcs_desc[i:j + 1]))
            i = j + 1
        plateaus = [(c, run) for c, run in runs if c >= 2 and len(run) >= 2]
        if plateaus:
            # widest; tie → run reaching the lower mcs (richer decomposition)
            best = max(plateaus, key=lambda cr: (len(cr[1]), -min(cr[1])))
            rep_mcs = max(best[1])  # highest mcs in the plateau (most stable)
            return sweep[rep_mcs]
        for mcs in mcs_desc:  # no plateau → first (coarsest) with ≥ 2
            if counts[mcs] >= 2:
                return sweep[mcs]
        for mcs in mcs_desc:  # no ≥ 2 anywhere → first 1-cluster result
            if counts[mcs] == 1:
                return sweep[mcs]
        return None, None, {}, {}

    if min_cluster_size_override is not None:
        # Explicit user override ("minimum words per thread"): skip the
        # plateau auto-selection and cluster directly at the requested mcs.
        ov = max(2, int(min_cluster_size_override))
        labels, persistence, kept_clusters, kept_pers = _kept_at(ov, "eom")
        if len(kept_clusters) < 2:
            l = _kept_at(ov, "leaf")
            if len(l[2]) > len(kept_clusters):
                labels, persistence, kept_clusters, kept_pers = l
    else:
        labels, persistence, kept_clusters, kept_pers = _pick("eom")
        if len(kept_clusters) < 2:
            # EOM gave 0 or 1 usable clusters — LEAF fragments more
            # aggressively and may surface real sub-structure EOM collapsed.
            # Use it if it finds strictly more clusters than EOM did.
            l_labels, l_pers, l_kept, l_kept_pers = _pick("leaf")
            if len(l_kept) > len(kept_clusters):
                labels, persistence, kept_clusters, kept_pers = (
                    l_labels, l_pers, l_kept, l_kept_pers,
                )

    if not kept_clusters:
        return empty_result

    # Per-thread word counts (also feeds the "n" field in the output).
    thread_word_counts: Dict[int, Dict[int, int]] = {}
    union_words: List[int] = []
    for cid, words in kept_clusters.items():
        thread_word_counts[cid] = {w: int(counts[w]) for w in words}
        union_words.extend(words)
    union_words = sorted(set(union_words))
    cid_order = list(kept_clusters.keys())
    cidf = corpus_idf or {}

    # Per-bin word-presence counts for intensity projection
    word_year_counts = np.zeros((max(union_words) + 1 if union_words else 1, n_bins), dtype=np.float64)
    bag_sets = [set(int(t) for t in b) for b in bags]
    for hi, bag in enumerate(bag_sets):
        bi = bin_of_hit[hi]
        for w in bag:
            if w < word_year_counts.shape[0]:
                word_year_counts[w, bi] += 1
    safe_totals = np.where(n_hits_yr > 0, n_hits_yr, 1).astype(np.float64)
    smooth_bins = max(1, smooth_win // year_bin)

    thread_records: List[Dict] = []
    for cid in cid_order:
        words = kept_clusters[cid]
        # Thread intensity = share of bin hits whose bag intersects thread words
        thread_set = set(words)
        intensity = np.zeros(n_bins, dtype=np.float64)
        for hi, bag in enumerate(bag_sets):
            if bag & thread_set:
                intensity[bin_of_hit[hi]] += 1
        intensity = intensity / safe_totals
        intensity_smooth = _smooth(intensity, win=smooth_bins) if smooth_bins > 1 else intensity
        max_v = float(intensity_smooth.max())
        if max_v <= 0:
            continue

        # Rank thread words by corpus-IDF c-TF-IDF: count × whole-corpus rarity
        name_of = {w: _vocab_name(w, v_blob, v_offsets, count_lemmas) for w in words}
        ctfidf = {
            w: thread_word_counts[cid][w] * cidf.get(name_of[w], corpus_idf_default)
            for w in words
        }
        ranked_words = sorted(words, key=lambda w: -ctfidf[w])
        words_out = [
            {
                "word": name_of[w],
                "n": thread_word_counts[cid][w],
                "score": round(ctfidf[w], 3),
            }
            for w in ranked_words
        ]

        # Peaks (local maxima ≥ 25% of thread's own max)
        peaks_raw: List[Tuple[int, float]] = []
        for bi in range(n_bins):
            v = intensity_smooth[bi]
            if v < 0.25 * max_v:
                continue
            left = intensity_smooth[bi - 1] if bi > 0 else 0
            right = intensity_smooth[bi + 1] if bi < n_bins - 1 else 0
            if v >= left and v >= right:
                peaks_raw.append((yr_min + bi * year_bin, float(v)))
        if not peaks_raw:
            peaks_raw = [(yr_min + int(np.argmax(intensity_smooth)) * year_bin, max_v)]
        dedup_window = max(year_bin, 5)
        deduped: List[Tuple[int, float]] = []
        for p, v in peaks_raw:
            if deduped and p - deduped[-1][0] <= dedup_window:
                if v > deduped[-1][1]:
                    deduped[-1] = (p, v)
            else:
                deduped.append((p, v))
        peaks = [p for p, _ in deduped]

        thread_records.append({
            "_cid": cid,  # internal: maps the record back to its HDBSCAN cluster
            "n_words": len(words),
            "persistence": round(kept_pers[cid], 4),
            "words": words_out,
            "peaks": peaks,
            "max_intensity": round(max_v, 6),
            "intensity": [round(float(intensity_smooth[bi]), 6) for bi in range(n_bins)],
        })

    # Sort by total share of mass (intensity area), assign ids, build labels
    thread_records.sort(key=lambda t: -sum(t["intensity"]))
    n_detected = len(thread_records)
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
        "threads": thread_records,
        "frequency": frequency,
    }
    if graph is not None:
        result["graph"] = graph
    return result
