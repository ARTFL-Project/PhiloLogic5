"""The passages of hits of the query, by their number in its hitlist: the example sentences of the word senses report
(get_word_senses), which sends the numbers of its examples' hits."""

import re

import numpy as np

from philologic.runtime.citations import citation_links, citations
from philologic.runtime.DB import DB
from philologic.runtime.exceptions import BadRequest
from philologic.runtime.get_text import get_concordance_text
from philologic.runtime.reports.collocation import collocate_distance, collocation_search_method

from .get_representative_passages import _prefetch_page_rows

MAX_PASSAGES = 100


def get_sense_passages(request, config):
    db = DB(config.db_path + "/data/")
    q = (request.q or "").strip()
    if not q:
        return {"passages": []}
    try:
        numbers = [int(n) for n in (request.hits or "").split(",") if n.strip()]
    except ValueError:
        raise BadRequest(f"Hits are given by their numbers: {request.hits}")
    if len(numbers) > MAX_PASSAGES:
        raise BadRequest(f"At most {MAX_PASSAGES} passages at a time")
    method, method_arg = collocation_search_method(q, db.locals.query_patterns, collocate_distance(request))
    hits = db.query(q, method, method_arg, **request.metadata)
    hits.finish()
    numbers = [n for n in numbers if 0 <= n < len(hits)]
    if not numbers:
        return {"passages": []}
    with hits.open_raw() as f:
        rows = np.frombuffer(f.read(), dtype=np.uint32).reshape(-1, hits.length)
    _prefetch_page_rows(db, rows, numbers)

    formatting = [
        (re.compile(rf"{pattern}"), replacement) for pattern, replacement in config.concordance_formatting_regex
    ]
    passages = []
    for n in numbers:
        hit = hits[n]
        hrefs = citation_links(db, config, hit)
        context = get_concordance_text(db, hit, config.db_path, config.concordance_length)
        for pattern, replacement in formatting:
            context = pattern.sub(rf"{replacement}", context)
        passages.append(
            {
                "hit": n,
                "philo_id": list(hit.philo_id),
                "citation": citations(hit, hrefs, config, report="concordance"),
                "citation_links": hrefs,
                "context": context,
            }
        )
    return {"passages": passages}
