"""Falcon resources for the 4 raw-WSGI PhiloLogic5 endpoints.

These endpoints couldn't use @json_endpoint because they need streaming,
custom headers (Set-Cookie), or non-JSON responses (302 redirect).
"""

import io
import os
import sys

import falcon
import orjson
import regex as re

from philologic.runtime import (
    WebConfig,
    WSGIHandler,
    access_control,
    bibliography_results,
    collocation_results,
    concordance_results,
    generate_time_series,
    kwic_results,
    login_access,
    aggregation_by_field,
)
from philologic.runtime.DB import DB
from philologic.runtime.HitWrapper import ObjectWrapper
from wsgi_helpers import resolve


# ---------------------------------------------------------------------------
# access_request — auth endpoint with Set-Cookie headers
# ---------------------------------------------------------------------------

class AccessRequestResource:
    """Handle authentication requests, setting cookies on success."""

    def on_get(self, req, resp, db_name):
        config = req.context.config
        request = req.context.request
        headers = [
            ("Content-type", "application/json; charset=UTF-8"),
            ("Access-Control-Allow-Origin", "*"),
        ]
        access, headers = login_access(req.env, request, config, headers)

        for name, value in headers:
            if name == "Set-Cookie":
                resp.append_header("Set-Cookie", value)

        resp.content_type = "application/json; charset=UTF-8"
        if access:
            resp.data = orjson.dumps({"access": True})
        else:
            incoming_address, domain_name = access_control.get_client_info(req.env)
            resp.data = orjson.dumps({
                "access": False,
                "incoming_address": incoming_address,
                "domain_name": domain_name,
            })


# ---------------------------------------------------------------------------
# export_results — JSON or CSV export
# ---------------------------------------------------------------------------

_TAGS = re.compile(r"<[^>]+>")
_NEWLINES = re.compile(r"\n+")
_SPACES = re.compile(r"\s{2,}")


def _filter_html(html):
    """Strip out all HTML."""
    text = _TAGS.sub("", html).strip()
    text = _NEWLINES.sub(" ", text)
    text = _SPACES.sub(" ", text)
    return text


def _csv_output(results):
    """Convert results to CSV representation."""
    import csv

    output_string = io.StringIO()
    writer = csv.DictWriter(
        output_string,
        fieldnames=["context", *sorted(results[0]["metadata_fields"].keys())],
    )
    writer.writeheader()
    for result in results:
        writer.writerow({**result["metadata_fields"], "context": result["context"]})
    return output_string.getvalue()


class ExportResultsResource:
    """Export search results in JSON or CSV format."""

    def on_get(self, req, resp, db_name):
        config = req.context.config
        request = req.context.request
        db_path = req.context.db_path

        _bibliography_results = resolve(db_path, "bibliography_results", bibliography_results)
        _concordance_results = resolve(db_path, "concordance_results", concordance_results)
        _kwic_results = resolve(db_path, "kwic_results", kwic_results)
        _collocation_results = resolve(db_path, "collocation_results", collocation_results)
        _generate_time_series = resolve(db_path, "generate_time_series", generate_time_series)
        _aggregation_by_field = resolve(db_path, "aggregation_by_field", aggregation_by_field)

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
                    hit_to_save["context"] = _filter_html(hit["context"])
                results.append(hit_to_save)
        elif request.report == "kwic":
            for hit in _kwic_results(request, config)["results"]:
                hit_to_save = {
                    "metadata_fields": {**hit["metadata_fields"], "philo_id": hit["philo_id"]},
                    "context": hit["context"],
                }
                if request.filter_html == "true":
                    hit_to_save["context"] = _filter_html(hit["context"])
                results.append(hit_to_save)
        elif request.report == "collocation":
            results = _collocation_results(request, config)["collocates"]
        elif request.report == "time_series":
            results = _generate_time_series(request, config)["results"]
        elif request.report == "aggregation":
            results = _aggregation_by_field(request, config)["results"]

        if request.output_format == "json":
            resp.content_type = "application/json; charset=UTF-8"
            resp.data = orjson.dumps(results)
        elif request.output_format == "csv":
            resp.content_type = "text/csv; charset=UTF-8"
            resp.data = _csv_output(results).encode("utf8")


# ---------------------------------------------------------------------------
# resolve_cite — 302 redirect to resolved citation URL
# ---------------------------------------------------------------------------

def _nav_query(obj, db):
    """Query all divs within a document."""
    conn = db.dbh
    c = conn.cursor()
    doc_id = int(obj.philo_id[0])
    next_doc_id = doc_id + 1
    c.execute("SELECT rowid FROM toms WHERE philo_id=?", (f"{doc_id} 0 0 0 0 0 0",))
    start_rowid = c.fetchone()[0]
    c.execute("SELECT rowid FROM toms WHERE philo_id=?", (f"{next_doc_id} 0 0 0 0 0 0",))
    try:
        end_rowid = c.fetchone()[0]
    except TypeError:
        c.execute("SELECT max(rowid) FROM toms;")
        end_rowid = c.fetchone()[0]
    c.execute(
        "SELECT * FROM toms WHERE rowid >= ? AND rowid <=? AND philo_type>='div' AND philo_type<='div3'",
        (start_rowid, end_rowid),
    )
    for o in c.fetchall():
        philo_id = [int(n) for n in o["philo_id"].split(" ")]
        yield ObjectWrapper(philo_id, db, row=o)


class ResolveCiteResource:
    """Resolve a citation query and redirect to the matching text object."""

    def on_get(self, req, resp, db_name):
        config = req.context.config
        request = req.context.request
        db = DB(config.db_path + "/data/")
        c = db.dbh.cursor()
        q = request.q

        best_url = config["db_url"]

        if " - " in q:
            milestone = q.split(" - ")[0]
        else:
            milestone = q

        milestone_segments = []
        last_segment = 0
        milestone_prefixes = []
        for separator in re.finditer(r" (?!\.)|\.(?! )", milestone):
            milestone_prefixes += [milestone[: separator.start()]]
            milestone_segments += [milestone[last_segment : separator.start()]]
            last_segment = separator.end()
        milestone_segments += [milestone[last_segment:]]
        milestone_prefixes += [milestone]

        print("SEGMENTS", repr(milestone_segments), file=sys.stderr)
        print("PREFIXES", repr(milestone_prefixes), file=sys.stderr)

        abbrev_match = None
        for pos, v in enumerate(milestone_prefixes):
            print("QUERYING for abbrev = ", v, file=sys.stderr)
            abbrev_q = c.execute("SELECT * FROM toms WHERE abbrev = ?;", (v,)).fetchone()
            if abbrev_q:
                abbrev_match = abbrev_q

        print("ABBREV", abbrev_match["abbrev"], abbrev_match["philo_id"], file=sys.stderr)
        doc_obj = ObjectWrapper(abbrev_match["philo_id"].split(), db)

        nav = _nav_query(doc_obj, db)

        best_match = None
        for n in nav:
            if n["head"] == request.q:
                print("MATCH", n["philo_id"], n["n"], n["head"], file=sys.stderr)
                best_match = n
                break

        if best_match:
            type_offsets = {"doc": 1, "div1": 2, "div2": 3, "div3": 4, "para": 5}
            t = best_match["philo_type"]
            short_id = best_match["philo_id"].split()[: type_offsets[t]]
            # Note: original uses f.make_absolute_object_link which may not exist
            best_url = config["db_url"]

        raise falcon.HTTPFound(best_url)


# ---------------------------------------------------------------------------
# get_sorted_kwic — streaming NDJSON with progress updates
# ---------------------------------------------------------------------------

class SortedKWICResource:
    """Streaming sorted KWIC: collect sort data, sort, paginate over one connection."""

    def on_get(self, req, resp, db_name):
        config = req.context.config
        request = req.context.request

        resp.content_type = "application/x-ndjson; charset=UTF-8"
        resp.set_header("X-Accel-Buffering", "no")
        resp.set_header("Cache-Control", "no-cache")
        resp.stream = self._generate(request, config)

    def _generate(self, request, config):
        """Yield NDJSON progress lines and final result."""
        from scripts.get_sorted_kwic import (
            _collect_metadata_sort,
            _collect_vectorized,
            _get_cache_path,
            _paginate,
            _sort_cache,
        )

        db = DB(config.db_path + "/data/")
        hits = db.query(request["q"], request["method"], request["arg"], **request.metadata)

        cache_path = _get_cache_path(request, db)
        sorted_path = f"{cache_path}.sorted"
        bin_path = cache_path + ".bin"

        # Phase 1: Fast pagination shortcut
        if os.path.exists(sorted_path):
            yield orjson.dumps(_paginate(sorted_path, hits, request, config, db)) + b"\n"
            return

        # Phase 2: Clean stale partial caches
        if os.path.exists(bin_path):
            os.remove(bin_path)
        if os.path.exists(cache_path):
            os.remove(cache_path)

        # Determine sort mode
        metadata_search = not (
            request.first_kwic_sorting_option in ("left", "right", "q", "")
            and request.second_kwic_sorting_option in ("left", "right", "q", "")
            and request.third_kwic_sorting_option in ("left", "right", "q", "")
        )
        colloc_dir = os.path.join(db.path, "collocations")

        # Phase 3: Data collection (streams progress lines)
        if metadata_search:
            metadata_fields = list(config.kwic_metadata_sorting_fields)
            yield from _collect_metadata_sort(hits, bin_path, metadata_fields, config, db)
        else:
            metadata_fields = []
            yield from _collect_vectorized(hits, bin_path, colloc_dir, db)

        # Phase 4: Sort
        _sort_cache(bin_path, sorted_path, request, metadata_fields)

        # Phase 5: Paginate and yield final result
        yield orjson.dumps(_paginate(sorted_path, hits, request, config, db)) + b"\n"
