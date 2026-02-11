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
    total_results = 0
    docs = [set() for _ in range(len(config["results_summary"]))]

    with open(hits.filename, "rb") as buffer:
        current_hits = buffer.read(hits.length * 4 * 1000)  # read 1000 hits initially
        while current_hits:
            hits_array = np.frombuffer(current_hits, dtype="u4").reshape(-1, hits.length)
            for pos, field in enumerate(config["results_summary"]):
                obj_level = OBJ_DICT[field["object_level"]]
                # Convert each hit to a tuple and add to the respective set
                docs[pos].update(map(lambda hit: tuple_to_str(hit, obj_level), hits_array))
            total_results += hits_array.shape[0]
            current_hits = buffer.read(hits.length * 4 * 10000)

    stats = []
    cursor = db.dbh.cursor()
    BATCH_SIZE = 900  # SQLite limit is ~999 variables
    for pos, field_obj in enumerate(config["results_summary"]):
        if field_obj["field"] == "title":
            count = len(docs[pos])
        else:
            try:
                philo_type = db.locals["metadata_types"][field_obj["field"]]
            except KeyError:
                continue
            # Validate field name to prevent SQL injection
            field = validate_column(field_obj["field"], db)
            # Convert doc set to list for parameterized query
            doc_list = list(docs[pos])
            # Strip quotes from philo_ids (they were added for the old string interpolation)
            doc_values = [d.strip("'") for d in doc_list]

            # Collect distinct values across batches using a set
            distinct_values = set()
            null_count = 0
            for i in range(0, len(doc_values), BATCH_SIZE):
                batch = doc_values[i:i + BATCH_SIZE]
                placeholders = ", ".join("?" for _ in batch)

                if philo_type != "div":
                    validated_philo_type = validate_philo_type(philo_type)
                    cursor.execute(
                        f"SELECT DISTINCT {field} FROM toms WHERE philo_type=? AND philo_id IN ({placeholders})",
                        (validated_philo_type, *batch)
                    )
                else:
                    cursor.execute(
                        f"SELECT DISTINCT {field} FROM toms WHERE philo_type IN ('div1', 'div2', 'div3') AND philo_id IN ({placeholders})",
                        batch
                    )
                for row in cursor:
                    if row[0] is None:
                        null_count = 1  # Count NULL as one distinct value
                    else:
                        distinct_values.add(row[0])

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
