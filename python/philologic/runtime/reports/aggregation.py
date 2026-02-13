# /usr/bin/env python3
"""Report designed to group results by metadata with additional breakdown optional"""

import numpy as np

from philologic.runtime.DB import DB
from philologic.runtime.sql_validation import validate_column, validate_object_level

OBJ_DICT = {"doc": 1, "div1": 2, "div2": 3, "div3": 4, "para": 5, "sent": 6, "word": 7}
OBJ_ZEROS = {"doc": 6, "div1": 5, "div2": 4, "div3": 3, "para": 2, "sent": 1, "word": 0}


def aggregation_by_field(request, config):
    """Group hitlist by metadata field"""
    db = DB(config.db_path + "/data/")
    if request.q == "" and request.no_q:
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

    group_by = validate_column(request.group_by, db)
    field_obj = __get_field_config(group_by, config)
    metadata_type = validate_object_level(field_obj["object_level"])

    metadata_fields_needed = {group_by, "philo_id", f"philo_{metadata_type}_id"}
    for citation in field_obj["field_citation"]:
        if citation["field"] in db.locals["metadata_fields"]:
            metadata_fields_needed.add(validate_column(citation["field"], db))
    if field_obj["break_up_field_citation"] is not None:
        for citation in field_obj["break_up_field_citation"]:
            if citation["field"] in db.locals["metadata_fields"]:
                metadata_fields_needed.add(validate_column(citation["field"], db))

    hits.finish()
    id_counts, total_results = __expand_hits_counted(hits, metadata_type)

    # Batch queries to avoid SQLite's ~999 variable limit
    cursor = db.dbh.cursor()
    distinct_philo_ids = [" ".join(map(str, pid)) for pid in id_counts]
    BATCH_SIZE = 900
    metadata_dict = {}
    fields_select = ', '.join(metadata_fields_needed)
    for i in range(0, len(distinct_philo_ids), BATCH_SIZE):
        batch = distinct_philo_ids[i:i + BATCH_SIZE]
        placeholders = ', '.join('?' for _ in batch)
        cursor.execute(
            f"select {fields_select} from toms where philo_{metadata_type}_id IN ({placeholders}) ORDER BY philo_{metadata_type}_id",
            batch,
        )
        for row in cursor:
            if group_by == "title":
                uniq_name = row[f"philo_{metadata_type}_id"]
            else:
                uniq_name = row[group_by]
            metadata_dict[tuple(map(int, row[f"philo_{metadata_type}_id"].split()))] = {
                **{field: row[field] or "" for field in metadata_fields_needed if row[field] or field == group_by},
                "field_name": uniq_name,
            }

    # Aggregate counts per metadata field value using pre-computed hit counts
    counts_by_field = {}
    break_up_field_name = field_obj["break_up_field"]
    if break_up_field_name is not None:
        for philo_id, hit_count in id_counts.items():
            try:
                field_name = metadata_dict[philo_id]["field_name"]
            except KeyError:
                continue
            try:
                break_up_field = f"{metadata_dict[philo_id][break_up_field_name]} {' '.join(map(str, philo_id[:OBJ_DICT[metadata_type]]))}"
            except KeyError:
                break_up_field = f"NO {break_up_field_name} {philo_id}"
            if field_name not in counts_by_field:
                counts_by_field[field_name] = {
                    "count": hit_count,
                    "metadata_fields": metadata_dict[philo_id],
                    "break_up_field": {break_up_field: {"count": hit_count, "philo_id": philo_id}},
                }
            else:
                counts_by_field[field_name]["count"] += hit_count
                if break_up_field not in counts_by_field[field_name]["break_up_field"]:
                    counts_by_field[field_name]["break_up_field"][break_up_field] = {"count": hit_count, "philo_id": philo_id}
                else:
                    counts_by_field[field_name]["break_up_field"][break_up_field]["count"] += hit_count
    else:
        for philo_id, hit_count in id_counts.items():
            try:
                field_name = metadata_dict[philo_id]["field_name"]
            except KeyError:
                continue
            if field_name not in counts_by_field:
                counts_by_field[field_name] = {
                    "count": hit_count,
                    "metadata_fields": metadata_dict[philo_id],
                    "break_up_field": {},
                }
            else:
                counts_by_field[field_name]["count"] += hit_count

    if break_up_field_name is not None:
        results = []
        for _, values in sorted(counts_by_field.items(), key=lambda x: x[1]["count"], reverse=True):
            results.append(
                {
                    "metadata_fields": values["metadata_fields"],
                    "count": values["count"],
                    "break_up_field": [
                        {"count": v["count"], "metadata_fields": metadata_dict[v["philo_id"]]}
                        for v in sorted(values["break_up_field"].values(), key=lambda item: item["count"], reverse=True)
                    ],
                }
            )
    else:
        results = [
            {"metadata_fields": values["metadata_fields"], "count": values["count"], "break_up_field": []}
            for values in sorted(counts_by_field.values(), key=lambda x: x["count"], reverse=True)
        ]

    return {
        "results": results,
        "break_up_field": break_up_field_name or "",
        "query": {k: v for k, v in request},
        "total_results": total_results,
    }


def __expand_hits_counted(hits, metadata_type):
    """Stream sorted hitlist with numpy, return per-ID hit counts.

    Exploits sorted hitlist order: uses vectorized diff to find object-level
    boundaries and computes run lengths without converting every hit to Python.

    Returns:
        (id_counts, total_results) where id_counts is {philo_id_tuple: hit_count}
    """
    object_level = OBJ_DICT[metadata_type]
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
                # Find where doc IDs change
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
                # Multi-column: view as void dtype for element-wise comparison
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

    # Flush last run
    if prev_id is not None:
        id_counts[prev_id] = prev_count

    return id_counts, total_results


def __get_field_config(group_by, config):
    field_to_return = {}
    for field_obj in config["aggregation_config"]:
        if field_obj["field"] == group_by:
            field_to_return = field_obj
    return field_to_return


if __name__ == "__main__":
    import sys

    from philologic.runtime import WebConfig

    class Request:
        def __init__(self, q, field, metadata):
            self.q = q
            self.group_by = field
            self.no_metadata = False
            self.no_q = False
            self.metadata = metadata
            self.method = "proxy"
            self.report = "aggregation"
            self.arg = ""
            self.start = 0

        def __getitem__(self, item):
            if item == "group_by":
                return self.group_by
            return getattr(self, item)

        def __iter__(self):
            for item in ["q", "group_by", "report"]:
                yield item, self[item]

    query_term, field, db_path = sys.argv[1:]
    config = WebConfig(db_path)
    request = Request(query_term, field, {})
    aggregation_by_field(request, config)
