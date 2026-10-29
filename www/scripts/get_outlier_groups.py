"""Identify outlier groups (e.g. titles, authors) in a map_field collocation cache.

Each group's distinctiveness is the sum of its top-k positive fightin' words z-scores against the rest of the corpus
-- captures both breadth (how many distinctively over-used collocates) and strength -- counting only the collocates
found in more than one work: the others are names and words used once, which every group of a work has. As z grows with
the evidence, bigger groups have more of it for the same lean: the outlier score is how far a group's distinctiveness
rises above that typical of groups of its size (see size_neutral_scores), as a ratio (1 is typical). Each outlier
is [name, score, explainers, their z-scores]: the z-scores weigh the explainers
in the group's representative passages (get_representative_passages.py).
"""

import os

import numpy as np

from philologic.runtime.reports.collocation import (
    cache_file_path,
    fightin_words_zscores_vs_rest,
    load_collocate_works,
    load_group_hits,
    load_map_field_cache,
    score_with_top_k_positive,
)
from philologic.runtime.WSGIHandler import whole_number


def get_outlier_groups(request, config):
    """Score each group's collocate distribution against the rest of the corpus."""
    file_path = cache_file_path(config.db_path, request.file_path, ".npz")
    tids, counts, group_bounds, group_names, count_lemmas = load_map_field_cache(file_path)
    group_hits = load_group_hits(file_path)  # None for legacy caches
    works = load_collocate_works(file_path)  # None for legacy caches: every collocate counts

    min_hits = whole_number("min_hits", request.min_hits, 10)
    top_n = whole_number("top_n", request.top_n, 50)
    top_k = whole_number("top_k", request.top_k, 20)

    n_groups = len(group_names)
    if n_groups == 0 or len(tids) == 0:
        return {"outliers": []}

    counts_f = counts.astype(np.float64)
    tids_i = tids.astype(np.int64)
    max_tid = int(tids_i.max()) + 1

    # Aggregate corpus vector (sum over all groups) and total
    corpus_vec = np.zeros(max_tid, dtype=np.float64)
    np.add.at(corpus_vec, tids_i, counts_f)
    corpus_total = float(counts_f.sum())

    # Per-group collocate-observation totals (for fightin' words n_a / n_b)
    group_totals = np.zeros(n_groups, dtype=np.float64)
    for g in range(n_groups):
        gs, ge = int(group_bounds[g]), int(group_bounds[g + 1])
        group_totals[g] = counts_f[gs:ge].sum()

    # Per-group fightin' words z-scores, restricted to tokens the group uses.
    # Tokens the group does not use only contribute negative z (under-represented),
    # which we discard, so the restriction is exact for the positive direction.
    # alpha_0 = n_a + n_b = corpus_total, and alpha[i] simplifies to y_a + y_b,
    # giving the closed form below.
    scores = np.full(n_groups, -np.inf, dtype=np.float64)
    explainer_tids = [None] * n_groups
    explainer_z = [None] * n_groups

    for g in range(n_groups):
        # Representativeness floor: minimum query-word hits in this group.
        # Legacy caches lack group_hits -- fall back to a collocate-observation floor.
        if group_hits is not None:
            if int(group_hits[g]) < min_hits:
                continue
        elif group_totals[g] < min_hits * 2:
            continue
        n_a = group_totals[g]
        n_b = corpus_total - n_a
        if n_a <= 0 or n_b <= 0:
            continue

        gs, ge = int(group_bounds[g]), int(group_bounds[g + 1])
        g_tids = tids_i[gs:ge]
        y_a = counts_f[gs:ge]
        y_b = corpus_vec[g_tids] - y_a  # rest-of-corpus on the group's tids

        z = fightin_words_zscores_vs_rest(y_a, y_b, n_a, n_b)
        if works is not None:
            z = np.where(works[gs:ge] >= 2, z, -np.inf)  # a collocate of one work is no use of the group's
        score, exp_tids, exp_z = score_with_top_k_positive(z, g_tids, top_k, min_distinctive=1)
        if score == -np.inf:
            continue
        scores[g] = score
        explainer_tids[g] = exp_tids
        explainer_z[g] = exp_z

    # Pick top-N eligible groups, by how much more distinctive they are than their size makes typical (then by
    # distinctiveness, when that can't be told)
    scored = np.flatnonzero(scores != -np.inf)
    if len(scored) == 0:
        return {"outliers": []}
    above = size_neutral_scores(scores[scored], group_totals[scored])
    ranked = scored[np.lexsort((-scores[scored], -above))][:top_n].tolist()
    ratio = dict(zip(scored.tolist(), np.exp(above).tolist()))

    # Decode vocab strings only for the displayed explainers
    colloc_dir = os.path.join(config.db_path, "data", "collocations")
    if count_lemmas:
        offsets = np.load(os.path.join(colloc_dir, "attr_lemma_vocab_offsets.npy"), mmap_mode="r")
        with open(os.path.join(colloc_dir, "attr_lemma_vocab_strings.bin"), "rb") as f:
            blob = f.read()
    else:
        offsets = np.load(os.path.join(colloc_dir, "vocab_offsets.npy"), mmap_mode="r")
        with open(os.path.join(colloc_dir, "vocab_strings.bin"), "rb") as f:
            blob = f.read()

    def display_word(tid):
        """Return the bare word for explainer display.
        When counting lemmas, this is the lemma (normalized) form; otherwise the
        surface form. The 'lemma:' prefix and any ':attr:value' suffix used by
        clickable collocates are stripped -- explainers are descriptive only."""
        s = int(offsets[tid])
        e = int(offsets[tid + 1])
        name = blob[s:e].decode("utf-8")
        if name.startswith("lemma:"):
            name = name[6:]
        return name

    outliers = [
        (
            group_names[g],
            round(ratio[g], 3),
            [display_word(int(t)) for t in explainer_tids[g]],
            [round(float(z), 3) for z in explainer_z[g]],
        )
        for g in ranked
    ]
    return {"outliers": outliers}


def size_neutral_scores(scores, sizes):
    """How far each group's score rises above the scores typical of groups of its size (sizes, its collocates), in log:
    its distance to the least-squares line of the log scores on the log sizes, pulled toward the line as far as noise
    could explain it (empirical Bayes). On average the squared distances are tau2 + c / size (least squares), tau2 the
    spread of the groups' own leans and c / size the noise of a group of that size, and each distance is weighed by
    tau2 / (tau2 + c / size): a small group's lead counts less. With fewer than 3 groups, or no spread beyond the
    noise, no group can be told from its size's typical score: all are 0."""
    x, y = np.log(sizes), np.log(scores)
    if len(x) < 3:
        return np.zeros(len(x))
    slope, intercept = np.polyfit(x, y, 1) if np.ptp(x) > 0 else (0.0, float(y.mean()))
    distance = y - (intercept + slope * x)
    (tau2, c), *_ = np.linalg.lstsq(np.c_[np.ones_like(x), 1 / sizes], distance**2, rcond=None)
    if tau2 <= 0:
        return np.zeros(len(x))
    return tau2 / (tau2 + max(c, 0.0) / sizes) * distance
