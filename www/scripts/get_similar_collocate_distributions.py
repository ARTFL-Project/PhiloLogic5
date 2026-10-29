"""The groups (authors, titles...) whose use of the query is most like that of the part of the corpus a search's
metadata selects (the reference), with the collocates that make them alike as explanations.

Each use is profiled by what it over-uses: the positive part of the fightin' words z-scores (Monroe et al. 2008) of its
collocates against the rest of the corpus, as the distinctive groups have them. What every text does around the query
cancels out, and the similarity of two uses is the cosine of their profiles: they are alike when they over-use the same
collocates. Each group is taken outside the reference: its collocates in the whole corpus (file_path) less those in the
reference (inside_file_path, counted with the search's metadata). So the reference is compared with what the groups do
elsewhere, and the groups all inside it, its own, are left out."""

import os

import numpy as np

from philologic.runtime.exceptions import BadRequest
from philologic.runtime.reports.collocation import (
    cache_file_path,
    fightin_words_zscores_vs_rest,
    load_map_field_cache,
    safe_pickle_load,
)


def get_similar_collocate_distributions(request, config):
    """Get similar collocate distributions"""
    if not request.primary_file_path or not request.inside_file_path:
        raise BadRequest("Similar usage needs the search's collocates and those of the groups inside and outside it")
    tids, counts, group_bounds, group_names, count_lemmas = load_map_field_cache(
        cache_file_path(config.db_path, request.file_path, ".npz")
    )
    inside = load_map_field_cache(cache_file_path(config.db_path, request.inside_file_path, ".npz"))
    reference_collocates = safe_pickle_load(cache_file_path(config.db_path, request.primary_file_path, ".pickle"))
    names, bounds, group_tids, group_counts = _outside_counts(tids, counts, group_bounds, group_names, inside)
    if not names or not reference_collocates:
        return {"similar": []}

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
    name_to_tid = {}
    tid_to_name = {}
    for tid in np.unique(tids).tolist():
        s = int(count_offsets[tid])
        e = int(count_offsets[tid + 1])
        name = vocab_data[s:e].decode("utf-8")
        name_to_tid[name] = tid
        tid_to_name[tid] = name

    # The reference's collocates: those of the groups' vocabulary (by tid), and the others, which only its hits without
    # a value of the field have
    ref_counts = np.zeros(int(tids.max()) + 1, dtype=np.float64)
    only_ref = []
    for word, count in reference_collocates.items():
        tid = name_to_tid.get(word)
        if tid is None:
            only_ref.append(count)
        else:
            ref_counts[tid] = count
    only_ref = np.array(only_ref, dtype=np.float64)

    # The corpus: the reference and every group outside it. A profile is against the rest of it.
    outside = np.bincount(group_tids, weights=group_counts, minlength=len(ref_counts))
    corpus = ref_counts + outside
    n_ref = float(ref_counts.sum() + only_ref.sum())
    n_rest = float(outside.sum())
    if n_rest == 0:  # the reference is the whole corpus: there is no rest to compare it with
        return {"similar": []}

    # Profiles: max(0, z) of each collocate against the rest of the corpus
    ref_profile = np.zeros(len(ref_counts), dtype=np.float64)
    support = np.flatnonzero(ref_counts)
    ref_profile[support] = _over_use(ref_counts[support], outside[support], n_ref, n_rest)
    ref_norm = np.sqrt(np.square(ref_profile).sum() + np.square(_over_use(only_ref, 0.0, n_ref, n_rest)).sum())
    entry_group = np.repeat(np.arange(len(names)), np.diff(bounds))
    n_group = np.bincount(entry_group, weights=group_counts, minlength=len(names))[entry_group]
    profiles = _over_use(group_counts, corpus[group_tids] - group_counts, n_group, n_ref + n_rest - n_group)

    # Cosine of each group's profile with the reference's; the parts of their shared collocates explain it
    parts = profiles * ref_profile[group_tids]
    dots = np.bincount(entry_group, weights=parts, minlength=len(names))
    norms = np.sqrt(np.bincount(entry_group, weights=np.square(profiles), minlength=len(names)))
    similarities = np.zeros(len(names), dtype=np.float64)
    alike = (dots > 0) & (norms > 0) & (ref_norm > 0)
    similarities[alike] = dots[alike] / (norms[alike] * ref_norm)

    similar = []
    for g in [g for g in np.argsort(-similarities, kind="stable").tolist() if similarities[g] > 0][:50]:
        gs, ge = int(bounds[g]), int(bounds[g + 1])
        shared = [gs + i for i in np.argsort(-parts[gs:ge], kind="stable")[:20].tolist() if parts[gs + i] > 0]
        similar.append((names[g], round(float(similarities[g]), 3), [_display(tid_to_name[group_tids[i]]) for i in shared]))
    return {"similar": similar}


def _outside_counts(tids, counts, group_bounds, group_names, inside):
    """Each group's collocate counts outside the reference: its counts in the whole corpus less those inside the
    reference (inside, a map_field cache of the same groups counted with the reference's metadata). The groups with no
    collocate left, all inside the reference, are left out. Returns (names, bounds, tids, counts), with the bounds of
    each group in tids and counts (floats)."""
    in_tids, in_counts, in_bounds, in_names, _ = inside
    index = {name: g for g, name in enumerate(group_names)}
    in_groups = np.array([index.get(name, -1) for name in in_names], dtype=np.int64)
    if (in_groups < 0).any():
        raise BadRequest("The groups inside the search are not those of the whole corpus: count them the same way")
    n_tids = int(max(tids.max(initial=0), in_tids.max(initial=0))) + 1
    keys = np.repeat(np.arange(len(group_names), dtype=np.int64), np.diff(group_bounds)) * n_tids + tids
    in_keys = np.repeat(in_groups, np.diff(in_bounds)) * n_tids + in_tids
    unique_keys, inverse = np.unique(np.concatenate([keys, in_keys]), return_inverse=True)
    left = np.bincount(
        inverse.ravel(),
        weights=np.concatenate([counts.astype(np.float64), -in_counts.astype(np.float64)]),
        minlength=len(unique_keys),
    )
    kept = left > 0
    unique_keys, left = unique_keys[kept], left[kept]
    kept_groups, starts = np.unique(unique_keys // n_tids, return_index=True)
    names = [group_names[g] for g in kept_groups.tolist()]
    return names, np.append(starts, len(unique_keys)), unique_keys % n_tids, left


def _over_use(y_a, y_b, n_a, n_b):
    """How much more a profile uses its collocates (y_a of its n_a) than the rest of the corpus (y_b of n_b): max(0, z)"""
    if len(y_a) == 0:
        return np.zeros(0, dtype=np.float64)
    return np.maximum(0.0, fightin_words_zscores_vs_rest(y_a, y_b, n_a, n_b))


def _display(name):
    """A collocate as the explanations show it: a lemma without its prefix"""
    return name[6:] if name.startswith("lemma:") else name
