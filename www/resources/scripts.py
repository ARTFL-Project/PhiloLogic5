"""Falcon resource for standard PhiloLogic5 script endpoints.

Handles all JSON and HTML scripts via a single resource class, dispatching
to plain (request, config) -> result functions in the scripts package.

The 4 special scripts (access_request, export_results, resolve_cite,
get_sorted_kwic) are handled by dedicated resources in streaming.py.
"""

import falcon
import orjson

import scripts
from wsgi_helpers import BadRequest, resolve

# Scripts that return JSON
JSON_SCRIPTS = {
    "alignment_to_text",
    "autocomplete_metadata",
    "autocomplete_term",
    "collocation_time_series",
    "comparative_collocations",
    "get_academic_citation",
    "get_bibliography",
    "get_collocate_distribution",
    "get_filter_list",
    "get_frequency",
    "get_hitlist_stats",
    "get_landing_page_content",
    "get_more_context",
    "get_notes",
    "get_query_terms",
    "get_similar_collocate_distributions",
    "get_sorted_frequency",
    "get_table_of_contents",
    "get_term_groups",
    "get_text_object",
    "get_total_results",
    "get_web_config",
    "get_word_frequency",
    "get_word_property_count",
    "lookup_word",
}

# Scripts that return HTML instead of JSON
HTML_SCRIPTS = {
    "get_custom_landing_page",
    "get_header",
}

ALLOWED_SCRIPTS = JSON_SCRIPTS | HTML_SCRIPTS


class ScriptResource:
    """Handle GET requests for all standard script endpoints."""

    def on_get(self, req, resp, db_name, script_name):
        if script_name not in ALLOWED_SCRIPTS:
            raise falcon.HTTPBadRequest(description=f"Invalid script: {script_name}")

        default_fn = getattr(scripts, script_name)
        config = req.context.config
        request = req.context.request
        fn = resolve(config.db_path, script_name, default_fn)

        try:
            result = fn(request, config)
        except BadRequest as e:
            raise falcon.HTTPBadRequest(description=str(e))

        if script_name in HTML_SCRIPTS:
            resp.text = result
            resp.content_type = "text/html; charset=UTF-8"
        else:
            resp.data = orjson.dumps(result)
            resp.content_type = "application/json; charset=UTF-8"
