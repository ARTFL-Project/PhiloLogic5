"""Falcon resource for all PhiloLogic5 report endpoints.

Handles all 8 reports via a single dispatch table. Each report resolves a
per-database handler function (allowing custom_functions overrides) and
serializes the result as JSON.
"""

import falcon
import orjson

from philologic.runtime import (
    aggregation_by_field,
    bibliography_results,
    collocation_results,
    concordance_results,
    generate_text_object,
    generate_time_series,
    generate_toc_object,
    kwic_results,
)
from wsgi_helpers import BadRequest, resolve

# Maps report URL name -> (resolve name, default function)
REPORT_HANDLERS = {
    "concordance": ("concordance_results", concordance_results),
    "kwic": ("kwic_results", kwic_results),
    "bibliography": ("bibliography_results", bibliography_results),
    "collocation": ("collocation_results", collocation_results),
    "time_series": ("generate_time_series", generate_time_series),
    "navigation": ("generate_text_object", generate_text_object),
    "table_of_contents": ("generate_toc_object", generate_toc_object),
    "aggregation": ("aggregation_by_field", aggregation_by_field),
}


class ReportResource:
    """Handle GET requests for all report types."""

    def on_get(self, req, resp, db_name, report_name):
        if report_name not in REPORT_HANDLERS:
            raise falcon.HTTPBadRequest(description=f"Invalid report: {report_name}")

        resolve_name, default_fn = REPORT_HANDLERS[report_name]
        config = req.context.config
        request = req.context.request
        handler = resolve(config.db_path, resolve_name, default_fn)

        try:
            result = handler(request, config)
        except BadRequest as e:
            raise falcon.HTTPBadRequest(description=str(e))

        # bibliography_results returns (result_dict, hits) — use first element
        if report_name == "bibliography" and isinstance(result, tuple):
            result = result[0]

        resp.data = orjson.dumps(result)
        resp.content_type = "application/json; charset=UTF-8"
