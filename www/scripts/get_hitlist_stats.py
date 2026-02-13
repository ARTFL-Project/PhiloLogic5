#!/var/lib/philologic5/philologic_env/bin/python3

import os
from wsgiref.handlers import CGIHandler

import numpy as np
import orjson
from philologic.runtime.DB import DB
from philologic.runtime.sql_validation import validate_column, validate_philo_type

from philologic.runtime import WebConfig, WSGIHandler

from custom_functions_loader import get_custom


OBJECT_LEVEL = {"doc": 6, "div1": 5, "div2": 4, "div3": 3, "para": 2, "sent": 1}
OBJ_DICT = {"doc": 1, "div1": 2, "div2": 3, "div3": 4, "para": 5, "sent": 6, "word": 7}


def get_hitlist_stats(environ, start_response):
    """Count hit occurences per metadata field"""
    status = "200 OK"
    headers = [
        ("Content-type", "application/json; charset=UTF-8"),
        ("Access-Control-Allow-Origin", "*"),
    ]
    start_response(status, headers)
    db_path = environ.get("PHILOLOGIC_DBPATH", os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    _WebConfig = get_custom(db_path, "WebConfig", WebConfig)
    _WSGIHandler = get_custom(db_path, "WSGIHandler", WSGIHandler)
    config = _WebConfig(db_path)
    db = DB(config.db_path + "/data/")
    request = _WSGIHandler(environ, config)
    if request.no_q:
        if request.no_metadata:
            hits = db.get_all(
                db.locals["default_object_level"],
                request["sort_order"],
                raw_results=True,
            )

        else:
            hits = db.query(sort_order=request["sort_order"], raw_results=True, **request.metadata)
    else:
        hits = db.query(
            request["q"],
            request["method"],
            request["arg"],
            raw_results=True,
            **request.metadata,
        )

    hits.finish()

    # Collect unique philo_ids per object level by streaming chunks.
    # Exploits sorted hitlist order: only converts values at document boundaries.
    CHUNK_SIZE = 100_000  # hits per chunk
    obj_levels = {}
    unique_ids_per_level = {}
    prev_per_level = {}
    for field_obj in config["results_summary"]:
        obj_level = OBJ_DICT[field_obj["object_level"]]
        obj_levels[field_obj["field"]] = obj_level
        unique_ids_per_level.setdefault(obj_level, [])
        prev_per_level.setdefault(obj_level, None)

    total_results = 0
    with open(hits.filename, "rb") as f:
        while True:
            chunk = f.read(hits.length * 4 * CHUNK_SIZE)
            if not chunk:
                break
            arr = np.frombuffer(chunk, dtype="u4").reshape(-1, hits.length)
            total_results += arr.shape[0]
            for obj_level in unique_ids_per_level:
                if obj_level == 1:
                    col = arr[:, 0]
                    prev = prev_per_level[obj_level]
                    if prev is None or col[0] != prev:
                        unique_ids_per_level[obj_level].append(int(col[0]))
                    if len(col) > 1:
                        mask = col[1:] != col[:-1]
                        unique_ids_per_level[obj_level].extend(col[1:][mask].tolist())
                    prev_per_level[obj_level] = col[-1]
                else:
                    cols = np.ascontiguousarray(arr[:, :obj_level])
                    void_col = cols.view(np.dtype((np.void, obj_level * 4))).ravel()
                    prev = prev_per_level[obj_level]
                    if prev is None or void_col[0] != prev:
                        unique_ids_per_level[obj_level].append(void_col[0])
                    if len(void_col) > 1:
                        mask = void_col[1:] != void_col[:-1]
                        unique_ids_per_level[obj_level].extend(void_col[1:][mask].tolist())
                    prev_per_level[obj_level] = void_col[-1]

    # Convert lists to sets for dedup (handles chunk boundary duplicates)
    for obj_level in unique_ids_per_level:
        unique_ids_per_level[obj_level] = set(unique_ids_per_level[obj_level])

    stats = []
    cursor = db.dbh.cursor()
    for field_obj in config["results_summary"]:
        obj_level = obj_levels[field_obj["field"]]
        id_set = unique_ids_per_level[obj_level]

        if field_obj["field"] == "title":
            count = len(id_set)
        else:
            try:
                philo_type = db.locals["metadata_types"][field_obj["field"]]
            except KeyError:
                continue

            # Convert unique integer IDs to philo_id strings for SQL
            if obj_level == 1:
                doc_values = [(f"{doc_id} 0 0 0 0 0 0",) for doc_id in id_set]
            else:
                # For multi-column obj_levels, decode the void bytes back to ints
                doc_values = []
                for void_val in id_set:
                    ids = np.frombuffer(void_val, dtype="u4")
                    doc_values.append((tuple_to_str(ids, obj_level).strip("'"),))

            # Validate field name to prevent SQL injection
            field = validate_column(field_obj["field"], db)

            # Use temp table + JOIN instead of batched IN queries
            cursor.execute("CREATE TEMP TABLE _hit_ids (philo_id TEXT)")
            cursor.executemany("INSERT INTO _hit_ids VALUES (?)", doc_values)
            cursor.execute("CREATE INDEX _hit_idx ON _hit_ids(philo_id)")

            if philo_type != "div":
                validated_philo_type = validate_philo_type(philo_type)
                cursor.execute(
                    f"SELECT DISTINCT toms.{field} FROM toms INNER JOIN _hit_ids ON toms.philo_id = _hit_ids.philo_id WHERE toms.philo_type=?",
                    (validated_philo_type,)
                )
            else:
                cursor.execute(
                    f"SELECT DISTINCT toms.{field} FROM toms INNER JOIN _hit_ids ON toms.philo_id = _hit_ids.philo_id WHERE toms.philo_type IN ('div1', 'div2', 'div3')"
                )

            distinct_values = set()
            null_count = 0
            for row in cursor:
                if row[0] is None:
                    null_count = 1  # Count NULL as one distinct value
                else:
                    distinct_values.add(row[0])

            cursor.execute("DROP TABLE _hit_ids")
            count = len(distinct_values) + null_count
        link_field = False
        for agg_config in config.aggregation_config:
            if agg_config["field"] == field_obj["field"]:
                link_field = True
                break
        stats.append({"field": field_obj["field"], "count": count, "link_field": link_field})
    yield orjson.dumps({"stats": stats})


def tuple_to_str(philo_id, obj_level):
    """Fast philo_id to str conversion:
    This is actually about 40-50% faster than a ' '.join(map(str, philo_id["obj_level]))"""
    if obj_level == 1:
        return f"'{philo_id[0]} 0 0 0 0 0 0'"
    elif obj_level == 2:
        return f"'{philo_id[0]} {philo_id[1]} 0 0 0 0 0'"
    elif obj_level == 3:
        return f"'{philo_id[0]} {philo_id[1]} {philo_id[2]} 0 0 0 0'"
    elif obj_level == 4:
        return f"'{philo_id[0]} {philo_id[1]} {philo_id[2]} {philo_id[3]} {philo_id[4]} 0 0 0'"


if __name__ == "__main__":
    CGIHandler().run(get_hitlist_stats)
