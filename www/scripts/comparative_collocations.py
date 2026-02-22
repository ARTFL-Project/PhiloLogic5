"""Compare collocations between two corpora."""

import numpy as np
from philologic.runtime.reports.collocation import fightin_words_zscores, safe_pickle_load

def comparative_collocations(request, config):
    """Calculate relative proportion of each collocate."""
    all_collocates = safe_pickle_load(request.primary_file_path)
    other_collocates = safe_pickle_load(request.other_file_path)
    whole_corpus = request.whole_corpus.lower() == "true" if request.whole_corpus else False

    top_relative_proportions, low_relative_proportions = get_relative_proportions(
        all_collocates, other_collocates, whole_corpus
    )

    return {
        "top": top_relative_proportions,
        "bottom": low_relative_proportions,
    }


def get_relative_proportions(all_collocates, other_collocates, whole_corpus):
    """Compare two Counter dicts using Fightin' Words z-scores."""
    words = sorted(set(all_collocates) | set(other_collocates))
    y_sub = np.array([all_collocates.get(w, 0) for w in words], dtype=np.float64)
    y_other = np.array([other_collocates.get(w, 0) for w in words], dtype=np.float64)

    if whole_corpus:
        y_other = np.maximum(y_other - y_sub, 0.0)

    zscores = fightin_words_zscores(y_sub, y_other)

    order = np.argsort(zscores)[::-1]

    top = []
    bottom = []
    for idx in order:
        z = round(float(zscores[idx]), 4)
        if z > 0:
            top.append((words[idx], z))
        else:
            bottom.append((words[idx], abs(z)))
    bottom.reverse()

    return top[:100], bottom[:100]
