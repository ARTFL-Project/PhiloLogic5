#!/usr/bin env python3
"""Frequency results for facets"""

import numpy as np
from urllib.parse import quote_plus

from philologic.runtime.DB import DB
from philologic.runtime.link import make_absolute_query_link
from philologic.runtime.sql_validation import validate_column

OBJ_DICT = {"doc": 1, "div1": 2, "div2": 3, "div3": 4, "para": 5, "sent": 6, "word": 7}


def frequency_results(request, config, sorted_results=False):
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

    # Build metadata_dict and word_counts in a single table scan when no metadata
    # filters are active. With filters, word_counts need a filtered query.
    cursor = db.dbh.cursor()
    metadata_dict = {}
    word_counts_by_field_name = {}
    if metadata_type != "div":
        if not has_metadata_filter and not biblio_search:
            cursor.execute(
                f"SELECT philo_id, {frequency_field}, word_count FROM toms WHERE philo_type=? AND {frequency_field} IS NOT NULL",
                (metadata_type,),
            )
            for philo_id, field_name, word_count in cursor:
                philo_id = tuple(map(int, philo_id[: philo_id.index(" 0")].split()))
                metadata_dict[philo_id] = field_name
                key = f"{field_name}"
                wc = int(word_count) if word_count else 0
                if key in word_counts_by_field_name:
                    word_counts_by_field_name[key] += wc
                else:
                    word_counts_by_field_name[key] = wc
        else:
            cursor.execute(
                f"SELECT philo_id, {frequency_field} FROM toms WHERE philo_type=? AND {frequency_field} IS NOT NULL",
                (metadata_type,),
            )
            for philo_id, field_name in cursor:
                philo_id = tuple(map(int, philo_id[: philo_id.index(" 0")].split()))
                metadata_dict[philo_id] = field_name
            if not biblio_search:
                word_counts_by_field_name = db.query(get_word_count_field=frequency_field, **request.metadata)
    else:
        cursor.execute(
            f"SELECT philo_id, {frequency_field}, word_count FROM toms WHERE philo_type IN (?, ?, ?) AND {frequency_field} IS NOT NULL",
            ("div1", "div2", "div3"),
        )
        for philo_id, field_name, word_count in cursor:
            philo_id = tuple(map(int, philo_id[: philo_id.index(" 0")].split()))
            metadata_dict[philo_id] = field_name
            if not biblio_search:
                key = f"{field_name}"
                wc = int(word_count) if word_count else 0
                if key in word_counts_by_field_name:
                    word_counts_by_field_name[key] += wc
                else:
                    word_counts_by_field_name[key] = wc

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

    frequency_object = {}
    frequency_object["results"] = counts
    frequency_object["hits_done"] = total_hits

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
            frequency_object["results"]["NULL"] = {
                "count": len(new_hits),
                "url": null_url,
                "metadata": {frequency_field: '"NULL"'},
                "total_word_count": local_hits.get_total_word_count(),
            }
        else:
            frequency_object["results"]["NULL"] = {
                "count": len(new_hits),
                "url": null_url,
                "metadata": {frequency_field: '"NULL"'},
            }

    frequency_object["more_results"] = False
    frequency_object["results_length"] = len(hits)
    frequency_object["query"] = dict([i for i in request])

    if sorted_results is True:
        frequency_object["results"] = sorted(
            frequency_object["results"].items(),
            key=lambda x: x[1]["count"],
            reverse=True,
        )

    return frequency_object


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
