#!/var/lib/philologic5/philologic_env/bin/python3
"""Concordance report"""

import struct

import regex as re
from philologic.runtime.citations import citation_links, citations
from philologic.runtime.DB import DB, hit_to_string
from philologic.runtime.get_text import get_concordance_text
from philologic.runtime.HitList import CombinedHitlist
from philologic.runtime.pages import page_interval

LEVEL_MAP = {"doc": 1, "div1": 2, "div2": 3, "div3": 4, "para": 5, "sent": 6}


def _prefetch_hit_rows(db, hits, start, end):
    """Pre-fetch all toms rows needed for a concordance page slice in one batch query."""
    if not hasattr(hits, "filename"):
        return
    needed_levels = {1}  # always need doc level
    for f_type in db.locals["metadata_types"].values():
        if f_type in LEVEL_MAP:
            needed_levels.add(LEVEL_MAP[f_type])
        elif f_type == "div":
            needed_levels.update((2, 3, 4))

    hit_count = min(end, len(hits)) - max(start - 1, 0)
    if hit_count <= 0:
        return

    if hits.sort_order:
        raw_hits = hits.sorted_hitlist[start - 1 : end]
    else:
        offset_bytes = hits.hitsize * (start - 1)
        with open(hits.filename, "rb") as f:
            f.seek(offset_bytes)
            raw_data = f.read(hits.hitsize * hit_count)
        raw_hits = [
            struct.unpack(hits.format, raw_data[i * hits.hitsize : (i + 1) * hits.hitsize])
            for i in range(len(raw_data) // hits.hitsize)
        ]

    unique_ids = set()
    for raw in raw_hits:
        for level in needed_levels:
            unique_ids.add(hit_to_string(raw[:level], db.width))

    db.prefetch_rows(list(unique_ids))


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

    _prefetch_hit_rows(db, hits, start, end)

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
        metadata_fields = {metadata: hit[metadata] for metadata in db.locals["metadata_fields"]}
        citation = citations(hit, citation_hrefs, config, report="concordance")
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
