"""Identify outlier groups (e.g. titles, authors) in a map_field collocation cache.

Each group is scored by fightin' words z-scores against the rest of the corpus.
The outlier score is the sum of the top-k positive z-scores -- captures both
breadth (how many distinctively over-used collocates) and strength.
"""

import os

import numpy as np

from philologic.runtime.reports.collocation import load_group_hits, load_map_field_cache


def get_outlier_groups(request, config):
    """Score each group's collocate distribution against the rest of the corpus."""
    tids, counts, group_bounds, group_names, count_lemmas, attribute, attribute_value = (
        load_map_field_cache(request.file_path)
    )
    group_hits = load_group_hits(request.file_path)  # None for legacy caches

    min_hits = int(request.min_hits or 10)
    top_n = int(request.top_n or 50)
    top_k = int(request.top_k or 20)

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

        delta = (
            np.log((2.0 * y_a + y_b) / (2.0 * n_a + n_b))
            - np.log((y_a + 2.0 * y_b) / (n_a + 2.0 * n_b))
        )
        sigma_sq = 1.0 / (2.0 * y_a + y_b) + 1.0 / (y_a + 2.0 * y_b)
        z = delta / np.sqrt(sigma_sq)

        pos_mask = z > 0
        if not pos_mask.any():
            continue
        z_pos = z[pos_mask]
        t_pos = g_tids[pos_mask]
        order = np.argsort(z_pos)[::-1][:top_k]
        scores[g] = float(z_pos[order].sum())
        explainer_tids[g] = t_pos[order]

    # Pick top-N eligible groups
    ranked = [int(g) for g in np.argsort(scores)[::-1] if scores[g] != -np.inf][:top_n]
    if not ranked:
        return {"outliers": []}

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
            round(float(scores[g]), 3),
            [display_word(int(t)) for t in explainer_tids[g]],
        )
        for g in ranked
    ]
    return {"outliers": outliers}
