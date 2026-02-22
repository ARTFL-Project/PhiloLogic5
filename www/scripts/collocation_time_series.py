"""Time series of collocations: each period is compared to the previous to get a sense of the shift between each period."""

import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from philologic.runtime.reports.collocation import fightin_words_zscores, load_map_field_cache

def collocation_time_series(request, config):
    """Reads a numpy cache containing collocations for each year."""
    cache_tids, cache_counts, group_bounds, group_names, count_lemmas, attribute, attribute_value = load_map_field_cache(
        request.file_path
    )

    # Decode vocab strings for column names
    colloc_dir = os.path.join(config.db_path, "data", "collocations")
    if count_lemmas:
        count_offsets = np.load(os.path.join(colloc_dir, "attr_lemma_vocab_offsets.npy"), mmap_mode="r")
        with open(os.path.join(colloc_dir, "attr_lemma_vocab_strings.bin"), "rb") as f:
            vocab_data = f.read()
    else:
        count_offsets = np.load(os.path.join(colloc_dir, "vocab_offsets.npy"), mmap_mode="r")
        with open(os.path.join(colloc_dir, "vocab_strings.bin"), "rb") as f:
            vocab_data = f.read()

    # Build sparse matrix: rows = years, cols = token IDs
    n_groups = len(group_names)
    max_tid = int(cache_tids.max()) + 1 if len(cache_tids) > 0 else 1

    # Build COO-format row indices from group_bounds
    row_indices = np.empty(len(cache_tids), dtype=np.int32)
    for g in range(n_groups):
        gs, ge = int(group_bounds[g]), int(group_bounds[g + 1])
        row_indices[gs:ge] = g

    sparse_matrix = csr_matrix(
        (cache_counts.astype(np.int64), (row_indices, cache_tids.astype(np.int64))),
        shape=(n_groups, max_tid),
    )

    # Decode column names for the unique tids that appear
    unique_tids = np.unique(cache_tids)

    # Build a dense matrix only over used columns
    used_sparse = sparse_matrix[:, unique_tids]
    col_names = []
    for tid in unique_tids.tolist():
        s = int(count_offsets[tid])
        e = int(count_offsets[tid + 1])
        name = vocab_data[s:e].decode("utf-8")
        if attribute is not None:
            suffix = f":{attribute}:{attribute_value}"
            name = (name + suffix) if count_lemmas else (name.lower() + suffix)
        col_names.append(name)

    # Convert year group_names to integers for period grouping
    year_indices = []
    valid_rows = []
    for i, name in enumerate(group_names):
        try:
            year_indices.append(int(name))
            valid_rows.append(i)
        except (ValueError, TypeError):
            continue

    if not valid_rows:
        return {"period": None, "done": True}

    dense_data = used_sparse[valid_rows].toarray()
    collocates_per_year_df = pd.DataFrame(dense_data, index=year_indices, columns=col_names)

    # Group by period ranges
    period = int(request.year_interval)
    collocates_per_year_df["period_group"] = (collocates_per_year_df.index // period) * period
    collocates_per_period = collocates_per_year_df.groupby("period_group").sum()
    collocates_per_period.sort_index(inplace=True)

    period_number = int(request.period_number)
    current_year = int(collocates_per_period.index[period_number])
    current_period = collocates_per_period.iloc[period_number].to_numpy()

    # Get frequent collocates for current period (ranked by raw count)
    frequent_collocates = pd.Series(current_period, index=collocates_per_period.columns).sort_values(ascending=False)
    frequent_collocates = [(word, int(score)) for word, score in frequent_collocates.head(100).items()]

    # Get neighboring periods
    prev_period = collocates_per_period.iloc[period_number - 1].to_numpy() if period_number > 0 else None
    next_period = (
        collocates_per_period.iloc[period_number + 1].to_numpy()
        if period_number < len(collocates_per_period) - 1
        else None
    )

    # Calculate distinctive collocates compared to both neighbors
    distinctive = calculate_distinctive_collocates(current_period, prev_period, next_period, collocates_per_period)

    done = period_number == len(collocates_per_period) - 1

    return {
        "period": {
            "year": current_year,
            "collocates": {"frequent": frequent_collocates, "distinctive": distinctive},
        },
        "done": done,
    }


def calculate_distinctive_collocates(current_period, prev_period, next_period, collocates_per_period):
    """Distinctive collocates using log-proportion ratio with Dirichlet prior (Fightin' Words)."""
    if prev_period is None and next_period is None:
        return []

    if prev_period is None:
        neighbor_period = next_period
    elif next_period is None:
        neighbor_period = prev_period
    else:
        neighbor_period = prev_period + next_period

    zscores = fightin_words_zscores(current_period, neighbor_period)

    # Return only over-represented collocates (positive z-scores), sorted by magnitude
    mask = zscores > 0
    positive_z = pd.Series(zscores[mask], index=collocates_per_period.columns[mask])
    positive_z = positive_z.sort_values(ascending=False).round(4)

    return [(word, float(score)) for word, score in positive_z.head(100).items()]
