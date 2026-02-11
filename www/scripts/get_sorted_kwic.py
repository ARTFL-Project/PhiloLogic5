#!/var/lib/philologic5/philologic_env/bin/python3

import os
import subprocess
from pathlib import Path
from wsgiref.handlers import CGIHandler

import orjson
from philologic.runtime import kwic_hit_object, page_interval
from philologic.runtime.DB import DB

from philologic.runtime import WebConfig, WSGIHandler

from custom_functions_loader import get_custom


def validate_cache_path(cache_path, db_path):
    """Validate that cache_path is within the hitlists directory."""
    hitlists_dir = Path(db_path, "data", "hitlists").resolve()
    try:
        requested_path = Path(cache_path).resolve()
        if not str(requested_path).startswith(str(hitlists_dir) + os.sep):
            raise ValueError("Invalid cache path: must be within hitlists directory")
        return str(requested_path)
    except (ValueError, OSError) as e:
        raise ValueError(f"Invalid cache path: {e}")


def get_sorted_kwic(environ, start_response):
    """Get sorted KWIC"""
    status = "200 OK"
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response(status, headers)
    db_path = environ.get("PHILOLOGIC_DBPATH", os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    _WebConfig = get_custom(db_path, "WebConfig", WebConfig)
    _WSGIHandler = get_custom(db_path, "WSGIHandler", WSGIHandler)
    config = _WebConfig(db_path)
    db = DB(config.db_path + "/data/")
    request = _WSGIHandler(environ, config)
    sorted_hits = get_sorted_hits(request, config, db)
    yield orjson.dumps(sorted_hits)


def get_sorted_hits(request, config, db):
    """Get sorted hits"""
    # Validate cache_path to prevent path traversal
    cache_path = validate_cache_path(request.cache_path, config.db_path)

    hits = db.query(request["q"], request["method"], request["arg"], **request.metadata)
    start, end, _ = page_interval(request.results_per_page, hits, request.start, request.end)
    kwic_object = {
        "description": {"start": start, "end": end, "results_per_page": request.results_per_page},
        "query": dict([i for i in request]),
    }
    if not os.path.exists(f"{cache_path}.sorted"):
        with open(cache_path) as cache:
            fields = cache.readline().strip().split("\t")
        sort_keys = []
        if request.first_kwic_sorting_option:
            key = fields.index(request.first_kwic_sorting_option) + 1
            sort_keys.extend(["-k", f"{key},{key}"])
        if request.second_kwic_sorting_option:
            key = fields.index(request.second_kwic_sorting_option) + 1
            sort_keys.extend(["-k", f"{key},{key}"])
        if request.third_kwic_sorting_option:
            key = fields.index(request.third_kwic_sorting_option) + 1
            sort_keys.extend(["-k", f"{key},{key}"])

        # Use subprocess instead of os.system for safety
        with open(cache_path, "r") as infile:
            # Skip header line
            next(infile)
            content = infile.read()

        sort_result = subprocess.run(
            ["sort"] + sort_keys,
            input=content,
            capture_output=True,
            text=True,
            check=True,
        )

        with open(f"{cache_path}.sorted", "w") as outfile:
            outfile.write(sort_result.stdout)

        os.remove(cache_path)
    kwic_results = []
    with open(f"{cache_path}.sorted") as sorted_results:
        for line_number, line in enumerate(sorted_results, 1):
            if line_number < start:
                continue
            if line_number > end:
                break
            index = int(line.split("\t")[0])
            hit = hits[index]
            kwic_result = kwic_hit_object(hit, config, db)
            kwic_results.append(kwic_result)

    kwic_object["results"] = kwic_results
    kwic_object["results_length"] = len(hits)
    kwic_object["query_done"] = hits.done

    return kwic_object


if __name__ == "__main__":
    CGIHandler().run(get_sorted_kwic)
