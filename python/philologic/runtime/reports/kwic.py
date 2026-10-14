#!/var/lib/philologic5/philologic_env/bin/python3
"""KWIC results"""

import csv
import io

import regex as re
from philologic.runtime.citations import citation_links, citations
from philologic.runtime.DB import DB
from philologic.runtime.exceptions import BadRequest
from philologic.runtime.get_text import get_text
from philologic.runtime.ObjectFormatter import adjust_bytes, format_strip
from philologic.runtime.pages import page_interval

SPAN = re.compile(r'<span class="(highlight|xml-w)"[^>]*>|</span>')
TAGS = re.compile(r"<[^>]+>")


def hit_bounds(conc_text):
    """Where the hit starts and ends: from the first highlight to the last, with the <w> spans they are in, so that
    the text before, the hit and the text after are each well-formed"""
    open_spans = []
    start = end = None
    after_highlight = False
    for m in SPAN.finditer(conc_text):
        if m.group(0) == "</span>":
            if not open_spans:  # none that clean_tags writes
                continue
            kind = open_spans.pop()[1]
            if kind == "highlight":
                after_highlight = True
            if after_highlight and not open_spans:
                end = m.end()
                after_highlight = False
        else:
            if m.group(1) == "highlight" and start is None:
                start = open_spans[0][0] if open_spans else m.start()
            open_spans.append((m.start(), m.group(1)))
    if start is None:
        raise ValueError("no highlight")
    return start, end if end is not None else len(conc_text)


def kwic_results(request, config):
    """Fetch KWIC results"""
    if request.no_q:  # without one, the hits are text objects, with no words to show in context
        raise BadRequest("A search term is required.")
    db = DB(config.db_path + "/data/")
    hits = db.query(request["q"], request["method"], request["arg"], **request.metadata)
    start, end, n = page_interval(request.results_per_page, hits, request.start, request.end)

    db.prefetch_hits(hits, start, end)

    kwic_object = {
        "description": {"start": start, "end": end, "results_per_page": request.results_per_page},
        "query": dict([i for i in request]),
    }
    kwic_object["results"] = []

    for hit in hits[start - 1 : end]:
        kwic_result = kwic_hit_object(hit, config, db)
        kwic_object["results"].append(kwic_result)

    # page_interval can't clamp end to the hits there are until the search is done
    kwic_object["description"]["end"] = start + len(kwic_object["results"]) - 1
    kwic_object["results_length"] = len(hits)
    kwic_object["query_done"] = hits.done

    return kwic_object


def kwic_hit_object(hit, config, db):
    """Build an individual kwic concordance"""
    # Get all metadata
    metadata_fields = {}
    for metadata in db.locals["metadata_fields"]:
        metadata_fields[metadata] = f"{hit[metadata]}".strip()

    # Get all links and citations
    citation_hrefs = citation_links(db, config, hit)
    citation = citations(hit, citation_hrefs, config)

    # Determine length of text needed
    byte_distance = hit.bytes[-1] - hit.bytes[0]
    length = config.concordance_length + byte_distance + config.concordance_length

    # Get concordance and align it
    byte_offsets, start_byte = adjust_bytes(hit.bytes, config.concordance_length)
    conc_text = get_text(hit, start_byte, length, config.db_path)
    conc_text = format_strip(conc_text, db.locals["token_regex"], byte_offsets)
    conc_text = conc_text.replace("\n", " ")
    conc_text = conc_text.replace("\r", "")
    conc_text = conc_text.replace("\t", " ")
    highlighted_text = ""
    try:
        start_hit, end_hit = hit_bounds(conc_text)
        start_output = (
            '<span class="kwic-before"><span class="inner-before">' + conc_text[:start_hit] + "</span></span>"
        )
        highlighted_text = TAGS.sub("", conc_text[start_hit:end_hit]).lower()  # for use in KWIC sorting
        end_output = '<span class="kwic-after">' + conc_text[end_hit:] + "</span>"
        conc_text = (
            '<span class="kwic-text">'
            + start_output
            + '&nbsp;<span class="kwic-highlight">'
            + conc_text[start_hit:end_hit]
            + "</span>&nbsp;"
            + end_output
            + "</span>"
        )
    except ValueError as v:
        import sys

        print("KWIC ERROR:", v, file=sys.stderr)

    if config.kwic_formatting_regex:
        for pattern, replacement in config.kwic_formatting_regex:
            conc_text = re.sub(rf"{pattern}", replacement, conc_text)
    kwic_result = {
        "philo_id": hit.philo_id,
        "context": conc_text,
        "highlighted_text": highlighted_text,
        "metadata_fields": metadata_fields,
        "citation_links": citation_hrefs,
        "citation": citation,
        "bytes": hit.bytes,
    }

    return kwic_result


def kwic_to_csv(results, filter_html=False):
    """Convert KWIC results to CSV string."""
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
