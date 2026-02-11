#!/var/lib/philologic5/philologic_env/bin/python3
"""Output results in JSON or CSV"""

import csv
import io
import os
from wsgiref.handlers import CGIHandler

import regex as re
from orjson import dumps

from philologic.runtime import (
    WebConfig,
    WSGIHandler,
    bibliography_results,
    concordance_results,
    kwic_results,
    collocation_results,
    generate_time_series,
    aggregation_by_field,
)

from custom_functions_loader import get_custom


TAGS = re.compile(r"<[^>]+>")
NEWLINES = re.compile(r"\n+")
SPACES = re.compile(r"\s{2,}")


def export_results(environ, start_response):
    db_path = environ.get("PHILOLOGIC_DBPATH", os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    _WebConfig = get_custom(db_path, "WebConfig", WebConfig)
    _WSGIHandler = get_custom(db_path, "WSGIHandler", WSGIHandler)
    _bibliography_results = get_custom(db_path, "bibliography_results", bibliography_results)
    _concordance_results = get_custom(db_path, "concordance_results", concordance_results)
    _kwic_results = get_custom(db_path, "kwic_results", kwic_results)
    _collocation_results = get_custom(db_path, "collocation_results", collocation_results)
    _generate_time_series = get_custom(db_path, "generate_time_series", generate_time_series)
    _aggregation_by_field = get_custom(db_path, "aggregation_by_field", aggregation_by_field)
    config = _WebConfig(db_path)
    request = _WSGIHandler(environ, config)
    results = []
    if request.report == "bibliography":
        results = _bibliography_results(request, config)["results"]
    if request.report == "concordance":
        for hit in _concordance_results(request, config)["results"]:
            hit_to_save = {
                "metadata_fields": {**hit["metadata_fields"], "philo_id": hit["philo_id"]},
                "context": hit["context"],
            }
            if request.filter_html == "true":
                hit_to_save["context"] = filter_html(hit["context"])
            results.append(hit_to_save)
    elif request.report == "kwic":
        for hit in _kwic_results(request, config)["results"]:
            hit_to_save = {
                "metadata_fields": {**hit["metadata_fields"], "philo_id": hit["philo_id"]},
                "context": hit["context"],
            }
            if request.filter_html == "true":
                hit_to_save["context"] = filter_html(hit["context"])
            results.append(hit_to_save)
    elif request.report == "collocation":
        results_object = _collocation_results(request, config)["collocates"]
    elif request.report == "time_series":
        results_object = _generate_time_series(request, config)["results"]
    elif request.report == "aggregation":
        results_object = _aggregation_by_field(request, config)["results"]

    if request.output_format == "json":
        status = "200 OK"
        headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
        start_response(status, headers)
        yield dumps(results)
    elif request.output_format == "csv":
        status = "200 OK"
        headers = [("Content-type", "text/csv; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
        start_response(status, headers)
        yield csv_output(results).encode("utf8")


def filter_html(html):
    """Strip out all HTML"""
    text = TAGS.sub("", html).strip()
    text = NEWLINES.sub(" ", text)
    text = SPACES.sub(" ", text)
    return text


def csv_output(results):
    """Convert results to CSV representation"""
    output_string = io.StringIO()
    writer = csv.DictWriter(
        output_string, fieldnames=["context", *sorted([field for field in results[0]["metadata_fields"].keys()])]
    )
    writer.writeheader()
    for result in results:
        writer.writerow({**result["metadata_fields"], "context": result["context"]})
    return output_string.getvalue()


if __name__ == "__main__":
    CGIHandler().run(export_results)
