#!/usr/bin env python3
"""Frequency results for facets"""

import numpy as np
from urllib.parse import quote_plus

from philologic.runtime.DB import DB
from philologic.runtime.MetadataQuery import bulk_load_metadata
from philologic.runtime.link import make_absolute_query_link
from philologic.runtime.sql_validation import validate_column

OBJ_DICT = {"doc": 1, "div1": 2, "div2": 3, "div3": 4, "para": 5, "sent": 6, "word": 7}


def frequency_results(request, config):
    """reads through a hitlist. looks up request.frequency_field in each hit, and builds up a list of
    unique values and their frequencies."""
    db = DB(config.db_path + "/data/")
    frequency_field = validate_column(request.frequency_field, db)
    biblio_search = False
    if request.q == "" and request.no_q:
        biblio_search = True
        if request.no_metadata:
            hits = db.get_all(
                db.locals["default_object_level"],
                sort_order=["rowid"],
                raw_results=True,
            )
        else:
            hits = db.query(sort_order=["rowid"], raw_results=True, **request.metadata)
    else:
        hits = db.query(
            request["q"],
            request["method"],
            request["arg"],
            raw_results=True,
            **request.metadata,
        )

    metadata_type = db.locals["metadata_types"][frequency_field]
    has_metadata_filter = any(v for v in request.metadata.values())

    # Build metadata_dict and word_counts via bulk_load_metadata.
    # When no metadata filters are active, load word_count in the same scan.
    # With filters, word_counts need a separate filtered query.
    metadata_dict = {}
    word_counts_by_field_name = {}
    load_word_count = (metadata_type == "div") or (not has_metadata_filter and not biblio_search)
    if load_word_count:
        _, cache = bulk_load_metadata(db, [frequency_field], extra_columns=["word_count"])[frequency_field]
        for prefix, (field_name, word_count) in cache.items():
            if not field_name:
                continue
            metadata_dict[prefix] = field_name
            if not biblio_search:
                wc = int(word_count) if word_count else 0
                word_counts_by_field_name[f"{field_name}"] = word_counts_by_field_name.get(f"{field_name}", 0) + wc
    else:
        _, cache = bulk_load_metadata(db, [frequency_field])[frequency_field]
        for prefix, field_name in cache.items():
            if field_name:
                metadata_dict[prefix] = field_name
        if not biblio_search:
            word_counts_by_field_name = db.query(get_word_count_field=frequency_field, **request.metadata)

    base_url = make_absolute_query_link(
        config,
        request,
        frequency_field="",
        start="0",
        end="0",
        report=request.report,
        script="",
    )

    hits.finish()

    # Use numpy to count hits per object-level ID
    if metadata_type != "div":
        object_level = OBJ_DICT[metadata_type]
    else:
        object_level = OBJ_DICT["div3"]  # Extract at finest div level
    id_counts, total_hits = __count_hits_by_level(hits, object_level)

    # Build frequency counts from distinct IDs with pre-computed hit counts
    counts = {}
    for philo_id, hit_count in id_counts.items():
        if metadata_type == "div":
            key = ""
            for div in ["div3", "div2", "div1"]:
                prefix = philo_id[: OBJ_DICT[div]]
                if prefix in metadata_dict:
                    key = metadata_dict[prefix]
                    break
            if not key:
                continue
        else:
            try:
                key = metadata_dict[philo_id]
            except KeyError:
                continue
        key = f"{key}"  # convert potential integers to strings
        if key not in counts:
            counts[key] = {"count": 0, "metadata": {frequency_field: key}}
            counts[key]["url"] = f'{base_url}&{frequency_field}="{quote_plus(key)}"'
            if not biblio_search:
                try:
                    counts[key]["total_word_count"] = word_counts_by_field_name[key]
                except KeyError:
                    # Worst case when there are different values for the field in div1, div2, and div3
                    query_metadata = {k: v for k, v in request.metadata.items() if v}
                    query_metadata[frequency_field] = f'"{key}"'
                    local_hits = db.query(**query_metadata)
                    counts[key]["total_word_count"] = local_hits.get_total_word_count()
        counts[key]["count"] += hit_count

    # Handle NULL values
    new_metadata = {k: v for k, v in request.metadata.items() if v}
    new_metadata[frequency_field] = '"NULL"'
    if request.q == "" and request.no_q:
        new_hits = db.query(sort_order=["rowid"], raw_results=True, **new_metadata)
    else:
        new_hits = db.query(
            request["q"],
            request["method"],
            request["arg"],
            raw_results=True,
            **new_metadata,
        )
    new_hits.finish()
    if len(new_hits):
        null_url = f'{base_url}&{frequency_field}="NULL"'
        local_hits = db.query(**new_metadata, raw_results=True)
        if not biblio_search:
            counts["NULL"] = {
                "count": len(new_hits),
                "url": null_url,
                "metadata": {frequency_field: '"NULL"'},
                "total_word_count": local_hits.get_total_word_count(),
            }
        else:
            counts["NULL"] = {
                "count": len(new_hits),
                "url": null_url,
                "metadata": {frequency_field: '"NULL"'},
            }

    # Build sorted results list — top 100 by absolute count
    results_list = []
    for label, data in sorted(counts.items(), key=lambda x: x[1]["count"], reverse=True)[:100]:
        entry = dict(data)
        entry["label"] = label
        results_list.append(entry)

    result = {
        "results": results_list,
        "results_length": len(hits),
        "query": dict([i for i in request]),
    }

    # Compute relative frequency (per 10,000 words) — top 100 by relative frequency
    if not biblio_search:
        relative_list = []
        for label, data in counts.items():
            total_wc = data.get("total_word_count", 0)
            if total_wc > 0:
                relative_list.append({
                    "label": label,
                    "count": round((data["count"] / total_wc) * 10000, 2),
                    "absolute_count": data["count"],
                    "total_word_count": total_wc,
                    "metadata": data["metadata"],
                    "url": data["url"],
                })
        relative_list.sort(key=lambda x: x["count"], reverse=True)
        result["relative_results"] = relative_list[:100]

    return result


def __count_hits_by_level(hits, object_level):
    """Stream sorted hitlist with numpy, return per-ID hit counts.

    Exploits sorted hitlist order: uses vectorized diff to find boundaries
    and computes run lengths without converting every hit to Python.

    Returns:
        (id_counts, total_results) where id_counts is {philo_id_tuple: hit_count}
    """
    CHUNK_SIZE = 100_000
    id_counts = {}
    total_results = 0
    prev_id = None
    prev_count = 0

    with open(hits.filename, "rb") as f:
        while True:
            chunk = f.read(hits.length * 4 * CHUNK_SIZE)
            if not chunk:
                break
            arr = np.frombuffer(chunk, dtype="u4").reshape(-1, hits.length)
            total_results += arr.shape[0]

            if object_level == 1:
                col = arr[:, 0]
                change_indices = np.where(col[1:] != col[:-1])[0] + 1
                boundaries = np.concatenate([[0], change_indices, [len(col)]])
                run_lengths = np.diff(boundaries)
                unique_vals = col[boundaries[:-1]]

                for val, rlen in zip(unique_vals.tolist(), run_lengths.tolist()):
                    key = (val,)
                    if key == prev_id:
                        prev_count += rlen
                    else:
                        if prev_id is not None:
                            id_counts[prev_id] = prev_count
                        prev_id = key
                        prev_count = rlen
            else:
                cols = np.ascontiguousarray(arr[:, :object_level])
                void_col = cols.view(np.dtype((np.void, object_level * 4))).ravel()
                change_indices = np.where(void_col[1:] != void_col[:-1])[0] + 1
                boundaries = np.concatenate([[0], change_indices, [len(void_col)]])
                run_lengths = np.diff(boundaries)

                for idx, rlen in zip(boundaries[:-1].tolist(), run_lengths.tolist()):
                    key = tuple(cols[idx].tolist())
                    if key == prev_id:
                        prev_count += rlen
                    else:
                        if prev_id is not None:
                            id_counts[prev_id] = prev_count
                        prev_id = key
                        prev_count = rlen

    if prev_id is not None:
        id_counts[prev_id] = prev_count

    return id_counts, total_results


if __name__ == "__main__":
    import sys

    from philologic.runtime import WebConfig

    class Request:
        def __init__(self, q, field, metadata):
            self.q = q
            self.frequency_field = field
            self.no_metadata = False
            self.no_q = False
            self.metadata = metadata
            self.method = "proxy"
            self.report = "frequency"
            self.arg = ""
            self.start = 0

        def __getitem__(self, item):
            return getattr(self, item)

        def __iter__(self):
            for item in ["q", "frequency_field", "report"]:
                yield item, self[item]

    query_term, field, db_path = sys.argv[1:]
    config = WebConfig(db_path)
    request = Request(query_term, field, {})
    frequency_results(request, config)
