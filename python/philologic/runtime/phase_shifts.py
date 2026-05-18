"""Phase-shift detection for collocation reports.

Given a query (e.g. "lemma:citoyen" with optional POS filter) over a corpus
window, identifies "phases" — contiguous spans of time within which the
query word's collocate distribution is internally coherent and distinct from
neighbors.

Algorithm:
  1. Run the query, sort hits by document year, and build a per-hit bag of
     collocate vocabulary IDs (filtered by attribute and excluding identity).
  2. Compute a sliding-window cosine-similarity curve over the hit stream:
     similarity[i] = cos(left_window_counts, right_window_counts).
  3. Local minima of the curve are candidate phase boundaries.
  4. Each candidate is assigned a "drop depth" = local-window max - similarity
     at the boundary (sharper shift = deeper drop).
  5. The kneedle algorithm on sorted depths picks a default K (number of
     boundaries to keep). The user can override via `n_phases`.
  6. For each phase, distinctive collocates are identified via log-odds with
     a Dirichlet prior against the rest of the query window.

Statistical grounding: drop depth is a non-parametric measure of how much
the local vocabulary distribution changes at a candidate point. The kneedle
default selects the K at which adding more boundaries gives diminishing
returns in drop magnitude — i.e. where the curve transitions from sharp
shifts to incremental noise. Users can slide along the drop-rank ordering
to see finer or coarser segmentations.
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

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


def _build_hit_bags(
    db: DB,
    db_path: str,
    q: str,
    count_lemmas: bool,
    attribute: Optional[str],
    attribute_value: Optional[str],
    metadata: Dict[str, str],
) -> Tuple[List[np.ndarray], np.ndarray, bytes, np.ndarray]:
    """Run the query, return (bags, years, vocab_blob, vocab_offsets).

    Each bag is a 1-D uint32 array of count-vocab IDs in that hit's sentence,
    excluding identity-filtered words and (if attribute is set) tokens whose
    attribute does not match attribute_value.
    """
    hits = db.query(q, "single_term", "", raw_results=True, raw_bytes=True, **metadata)
    while not os.path.exists(f"{hits.filename}.terms"):
        time.sleep(0.05)
    hits.finish()
    if len(hits) == 0:
        return [], np.array([], dtype=np.int32), b"", np.array([0], dtype=np.uint64)

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
    excluded = np.zeros(n_vocab, dtype=bool)
    filter_bytes = {s.encode("utf-8") for s in filter_set}
    for tid in range(n_vocab):
        b = v_blob[int(v_offsets[tid]):int(v_offsets[tid + 1])]
        if b in filter_bytes:
            excluded[tid] = True

    if attribute is not None:
        attr_ids_mmap = np.load(os.path.join(colloc_dir, f"attr_{attribute}_ids.npy"), mmap_mode="r")
        target_attr_id = _resolve_attr_target(colloc_dir, attribute, attribute_value)
    else:
        attr_ids_mmap = None
        target_attr_id = -1

    hit_keys_be = np.ascontiguousarray(all_hits[:, :6].astype(">u4"))
    hit_arr = hit_keys_be.view("S24").ravel()
    n_sents = len(sent_keys_s24)
    sent_idx = np.searchsorted(sent_keys_s24, hit_arr)
    matched = (sent_idx < n_sents) & (sent_keys_s24[np.clip(sent_idx, 0, n_sents - 1)] == hit_arr)

    field_obj_index, metadata_cache = bulk_load_metadata(db, ["year"])["year"]

    bags: List[np.ndarray] = []
    years_list: List[int] = []
    for i in range(len(all_hits)):
        if not matched[i]:
            continue
        si = int(sent_idx[i])
        s_start = int(sent_offsets[si])
        s_end = int(sent_offsets[si + 1])
        if attribute is not None:
            mask = (attr_ids_mmap[s_start:s_end] == target_attr_id)
            tids = count_ids[s_start:s_end][mask]
        else:
            tids = np.asarray(count_ids[s_start:s_end])
        if len(tids) > 0:
            tids = tids[~excluded[tids]]
        bags.append(np.asarray(tids))
        prefix = tuple(int(x) for x in all_hits[i, :field_obj_index])
        mv = metadata_cache.get(prefix)
        try:
            year = int(str(mv).strip()[:4]) if mv is not None else -1
        except (ValueError, TypeError):
            year = -1
        years_list.append(year)

    years = np.array(years_list, dtype=np.int32)
    return bags, years, v_blob, np.asarray(v_offsets)


# ------------------------ Sliding-window similarity ----------------------

def _vec_from_bags(bags: Sequence[np.ndarray], vocab_size: int) -> np.ndarray:
    v = np.zeros(vocab_size, dtype=np.float64)
    for b in bags:
        if len(b) > 0:
            np.add.at(v, b, 1.0)
    return v


def _compute_idf(bags: Sequence[np.ndarray], vocab_size: int, min_df: int = 2) -> np.ndarray:
    """IDF per vocab id, treating each hit's sentence-bag as one 'document'.
    Returns idf[w] = log(N / (1 + df[w])) + 1 — strictly positive, downweights
    backbone vocabulary that appears in many hits, upweights distinctive words.
    Hapaxes (df < min_df) are zeroed out: they're noise (proper nouns, typos,
    single-occurrence idiosyncrasies) and would otherwise get the highest IDF.
    """
    df = np.zeros(vocab_size, dtype=np.float64)
    for b in bags:
        if len(b) == 0:
            continue
        df[np.unique(b)] += 1
    N = len(bags)
    idf = np.log(N / (1.0 + df)) + 1.0
    idf[df < min_df] = 0.0
    return idf


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return 1.0
    return float(np.dot(u, v) / (nu * nv))


def _sliding_similarity(
    bags: Sequence[np.ndarray], vocab_size: int, window: int, idf: np.ndarray
) -> np.ndarray:
    """TF-IDF cosine between left and right windows around each hit-index.
    `idf` is applied multiplicatively to each window's count vector before cosine,
    so the curve responds to shifts in distinctive vocabulary rather than to
    fluctuations in the high-frequency backbone (loi, état, nation, ...).
    """
    n = len(bags)
    sim = np.full(n, np.nan, dtype=np.float64)
    if n < 2 * window:
        return sim
    for i in range(window, n - window):
        left = _vec_from_bags(bags[i - window:i], vocab_size) * idf
        right = _vec_from_bags(bags[i:i + window], vocab_size) * idf
        sim[i] = _cosine(left, right)
    return sim


def _local_minima(curve: np.ndarray, neighborhood: int) -> List[int]:
    n = len(curve)
    out: List[int] = []
    for i in range(n):
        if math.isnan(curve[i]):
            continue
        lo = max(0, i - neighborhood)
        hi = min(n, i + neighborhood + 1)
        seg = curve[lo:hi]
        seg = seg[~np.isnan(seg)]
        if len(seg) == 0:
            continue
        if curve[i] == seg.min() and (seg < curve[i] + 1e-12).sum() == 1:
            out.append(i)
    return out


# --------------------------- K selection -------------------------------

def _kneedle_K(sorted_drops: np.ndarray) -> int:
    """Return the number of points before the elbow of a descending sorted
    series. Elbow = farthest-below the diagonal (0,1)-(1,0) in unit square."""
    n = len(sorted_drops)
    if n < 2:
        return n
    x = np.arange(n, dtype=np.float64)
    x_n = x / x[-1] if x[-1] > 0 else x
    span = sorted_drops[0] - sorted_drops[-1]
    y_n = (sorted_drops - sorted_drops[-1]) / span if span > 0 else sorted_drops - sorted_drops[-1]
    dist = (x_n + y_n - 1) / math.sqrt(2)
    return int(np.argmin(dist)) + 1


# ----------------------- Per-phase distinctive vocab --------------------

def _phase_top_words(
    phase_vec: np.ndarray, corpus_vec: np.ndarray, top_k: int = 10, min_count: int = 3
) -> List[Tuple[int, float]]:
    rest = corpus_vec - phase_vec
    n_a = float(phase_vec.sum())
    n_b = float(rest.sum())
    if n_a <= 0 or n_b <= 0:
        return []
    alpha = 0.01
    V = len(phase_vec)
    p_a = (phase_vec + alpha) / (n_a + alpha * V)
    p_b = (rest + alpha) / (n_b + alpha * V)
    log_odds = np.log(p_a / (1 - p_a)) - np.log(p_b / (1 - p_b))
    var = 1 / (phase_vec + alpha) + 1 / (rest + alpha)
    z = log_odds / np.sqrt(var)
    z[phase_vec < min_count] = -np.inf
    top = np.argsort(z)[::-1][:top_k]
    return [(int(t), float(z[t])) for t in top if not math.isinf(z[t])]


def _vocab_name(tid: int, v_blob: bytes, v_offsets: np.ndarray, count_lemmas: bool) -> str:
    s = v_blob[int(v_offsets[tid]):int(v_offsets[tid + 1])].decode("utf-8", errors="replace")
    if count_lemmas and s.startswith("lemma:"):
        s = s[6:]
    return s


# ------------------------------- Top-level ------------------------------

def detect_phases(
    db: DB,
    db_path: str,
    q: str,
    count_lemmas: bool,
    attribute: Optional[str],
    attribute_value: Optional[str],
    metadata: Dict[str, str],
    *,
    n_phases: Optional[int] = None,
    min_phase_hits: int = 40,
    top_k_words: int = 10,
) -> Dict:
    """Run end-to-end phase detection.

    `n_phases` overrides the kneedle default; if None, use kneedle's K+1 phases.
    """
    bags, years, v_blob, v_offsets = _build_hit_bags(
        db, db_path, q, count_lemmas, attribute, attribute_value, metadata
    )
    if not bags:
        return {"n_total_hits": 0, "phases": [], "candidates": []}

    # Year-filter & sort
    yr_mask = years >= 0
    bags = [bags[i] for i in range(len(bags)) if yr_mask[i]]
    years = years[yr_mask]
    order = np.argsort(years, kind="stable")
    bags = [bags[i] for i in order]
    years = years[order]
    n = len(bags)

    # Compact-remap vocab to only IDs in this corpus subset
    seen = set()
    for b in bags:
        seen.update(b.tolist())
    seen_sorted = sorted(seen)
    n_vocab = len(v_offsets) - 1
    remap = np.full(n_vocab, -1, dtype=np.int32)
    for ci, tid in enumerate(seen_sorted):
        remap[tid] = ci
    bags = [remap[b] for b in bags]
    inverse_remap = np.array(seen_sorted, dtype=np.int64)
    compact_size = len(seen_sorted)

    # Sliding-window similarity & candidate detection (TF-IDF cosine)
    window = max(20, min(150, n // 8))
    idf = _compute_idf(bags, compact_size)
    sim = _sliding_similarity(bags, compact_size, window, idf)
    minima = _local_minima(sim, neighborhood=window // 2)
    minima_sorted = sorted(minima)

    # Enforce min phase size
    candidates: List[int] = []
    last = 0
    for b in minima_sorted:
        if b - last < min_phase_hits or n - b < min_phase_hits:
            continue
        candidates.append(b)
        last = b

    # Compute drop depth per candidate
    depths: List[Tuple[int, float, float]] = []
    for b in candidates:
        lo = max(0, b - window)
        hi = min(len(sim), b + window + 1)
        seg = sim[lo:hi]
        seg = seg[~np.isnan(seg)]
        local_max = float(seg.max()) if len(seg) > 0 else 1.0
        depths.append((b, local_max - float(sim[b]), float(sim[b])))
    depths_sorted = sorted(depths, key=lambda x: -x[1])

    # Default K from kneedle
    drops_arr = np.array([d for _, d, _ in depths_sorted])
    default_K = _kneedle_K(drops_arr) if len(drops_arr) > 0 else 0
    if n_phases is None:
        K_used = default_K
    else:
        K_used = max(0, min(int(n_phases) - 1, len(depths_sorted)))

    # Take top-K candidates, sort by position
    chosen = sorted(b for b, _, _ in depths_sorted[:K_used])
    cuts = [0] + chosen + [n]

    # Build phase summaries
    corpus_vec = _vec_from_bags(bags, compact_size)
    phases = []
    for ph in range(len(cuts) - 1):
        a, b = cuts[ph], cuts[ph + 1]
        phase_vec = _vec_from_bags(bags[a:b], compact_size)
        top = _phase_top_words(phase_vec, corpus_vec, top_k=top_k_words)
        words = [
            {"word": _vocab_name(int(inverse_remap[t]), v_blob, v_offsets, count_lemmas),
             "z": round(z, 3)}
            for t, z in top
        ]
        phases.append({
            "start_year": int(years[a]),
            "end_year": int(years[b - 1]),
            "n_hits": int(b - a),
            "top_words": words,
        })

    candidates_out = [
        {"hit_idx": int(b), "year": int(years[b]), "drop": round(d, 3), "rank": rank + 1}
        for rank, (b, d, _) in enumerate(depths_sorted)
    ]

    # Per-year hit counts (for the temporal anchor line)
    yr_min, yr_max = int(years.min()), int(years.max())
    counts = np.bincount(years - yr_min, minlength=yr_max - yr_min + 1)
    frequency = [[yr_min + i, int(c)] for i, c in enumerate(counts)]

    return {
        "n_total_hits": n,
        "year_range": [yr_min, yr_max],
        "min_phases": 1,
        "max_phases": len(depths_sorted) + 1,
        "default_n_phases": default_K + 1,
        "n_phases": K_used + 1,
        "candidates": candidates_out,
        "phases": phases,
        "frequency": frequency,
    }
