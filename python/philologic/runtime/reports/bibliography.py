#!/var/lib/philologic5/philologic_env/bin/python3
"""Bibliography results"""


from philologic.runtime.citations import citation_links, citations
from philologic.runtime.DB import DB
from philologic.runtime.get_text import get_text_obj
from philologic.runtime.pages import page_interval


def bibliography_results(request, config):
    """Fetch bibliography results"""
    db = DB(config.db_path + "/data/")
    if request.full_bibliography is True:  # This is an API call to get the full biblio returned as a JSON object
        hits = db.get_all(
            db.locals["default_object_level"],
        )
        metadata_to_get = [c["field"] for c in config.bibliography_citation]
        return [{metadata: hit[metadata] for metadata in metadata_to_get} for hit in hits], ""
    if request.no_metadata:
        hits = db.get_all(
            db.locals["default_object_level"],
            request["sort_order"],
        )
    else:
        hits = db.query(sort_order=request["sort_order"], **request.metadata)
    if (
        request.simple_bibliography == "all"
    ):  # request from simple landing page report which gets all biblio in load order
        hits.finish()
        start = 1
        end = len(hits)
        page_num = end
    else:
        start, end, page_num = page_interval(request.results_per_page, hits, request.start, request.end)
    bibliography_object = {
        "description": {"start": start, "end": end, "n": page_num, "results_per_page": request.results_per_page},
        "query": dict([i for i in request]),
        "default_object": db.locals["default_object_level"],
    }
    results = []
    result_type = "doc"
    for hit in hits[start - 1 : end]:
        citation_hrefs = citation_links(db, config, hit)
        metadata_fields = {}
        for metadata in db.locals["metadata_fields"]:
            metadata_fields[metadata] = hit[metadata]
        result_type = hit.object_type
        if request.simple_bibliography == "all":
            citation = citations(hit, citation_hrefs, config, report="simple_landing")
        else:
            citation = citations(hit, citation_hrefs, config, report="bibliography")
        if config.dictionary_bibliography is False or result_type == "doc":
            results.append(
                {
                    "citation": citation,
                    "citation_links": citation_hrefs,
                    "philo_id": hit.philo_id,
                    "metadata_fields": metadata_fields,
                    "object_type": result_type,
                }
            )
        else:
            context = get_text_obj(hit, config, request, db.locals["token_regex"], images=False)
            results.append(
                {
                    "citation": citation,
                    "citation_links": citation_hrefs,
                    "philo_id": hit.philo_id,
                    "metadata_fields": metadata_fields,
                    "context": context,
                    "object_type": result_type,
                }
            )
    bibliography_object["results"] = results
    bibliography_object["results_length"] = len(hits)
    bibliography_object["query_done"] = hits.done
    bibliography_object["result_type"] = result_type
    return bibliography_object, hits
