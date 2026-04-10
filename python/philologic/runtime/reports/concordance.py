#!/var/lib/philologic5/philologic_env/bin/python3
"""Concordance report"""

import csv
import io

import regex as re
from philologic.runtime.citations import citation_links, citations
from philologic.runtime.DB import DB
from philologic.runtime.get_text import get_concordance_text
from philologic.runtime.HitList import CombinedHitlist
from philologic.runtime.pages import page_interval


def concordance_results(request, config):
    """Fetch concordances results."""
    db = DB(config.db_path + "/data/")

    hits = db.query(
        request["q"],
        request["method"],
        request["arg"],
        request["cooc_order"],
        sort_order=request["sort_order"],
        **request.metadata,
    )

    start, end, _ = page_interval(request["results_per_page"], hits, request.start, request.end)

    db.prefetch_hits(hits, start, end)

    concordance_object = {
        "description": {"start": start, "end": end, "results_per_page": request.results_per_page},
        "query": dict([i for i in request]),
        "default_object": db.locals["default_object_level"],
    }

    formatting_regexes = []
    if config.concordance_formatting_regex:
        for pattern, replacement in config.concordance_formatting_regex:
            compiled_regex = re.compile(rf"{pattern}")
            formatting_regexes.append((compiled_regex, replacement))
    results = []
    for hit in hits[start - 1 : end]:
        citation_hrefs = citation_links(db, config, hit)
        citation = citations(hit, citation_hrefs, config, report="concordance")
        metadata_fields = {metadata: hit[metadata] for metadata in db.locals["metadata_fields"]}
        context = get_concordance_text(db, hit, config.db_path, config.concordance_length)
        if formatting_regexes:
            for formatting_regex, replacement in formatting_regexes:
                context = formatting_regex.sub(rf"{replacement}", context)
        result_obj = {
            "philo_id": hit.philo_id,
            "citation": citation,
            "citation_links": citation_hrefs,
            "context": context,
            "metadata_fields": metadata_fields,
            "bytes": hit.bytes,
        }
        results.append(result_obj)

    concordance_object["results"] = results
    concordance_object["results_length"] = len(hits)
    concordance_object["query_done"] = hits.done
    return concordance_object


def concordance_to_csv(results, filter_html=False):
    """Convert concordance results to CSV string."""
    if not results:
        return ""
    tags_re = re.compile(r"<[^>]+>")
    output = io.StringIO()
    metadata_keys = sorted(results[0]["metadata_fields"].keys())
    fieldnames = ["philo_id", "context"] + metadata_keys
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        context = result["context"]
        if filter_html:
            context = tags_re.sub("", context).strip()
        row = {"philo_id": " ".join(str(x) for x in result["philo_id"]), "context": context}
        row.update(result["metadata_fields"])
        writer.writerow(row)
    return output.getvalue()
