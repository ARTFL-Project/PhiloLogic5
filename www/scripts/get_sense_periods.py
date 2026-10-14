"""How the words of the query's senses change over time (see philologic.runtime.reports.sense_periods)."""

from philologic.runtime.DB import DB
from philologic.runtime.exceptions import BadRequest
from philologic.runtime.reports.collocation import collocate_distance, collocate_filters, common_words
from philologic.runtime.reports.sense_periods import sense_periods
from philologic.runtime.sql_validation import validate_column


def get_sense_periods(request, config):
    db = DB(config.db_path + "/data/")
    q = (request.q or "").strip()
    if not q:
        return {"n_hits": 0, "n_sentences": 0, "n_works": 0, "nodes": [], "senses": [], "filter_list": []}
    filters = collocate_filters(request, db)  # as the collocation report counts them
    filter_words, stopwords_missing = common_words(request, config, filters)
    try:
        year_field = validate_column(config.time_series_year_field, db)
    except Exception:  # no year to place the sentences in time
        year_field = None
    # Rarer words weigh more, and only the query's collocates mark senses, whatever the filter (whose words seed no
    # split)
    try:  # the senses whose periods are wanted (their ids in the senses report), else those of its default view
        nodes = [int(n) for n in (request.nodes or "").split(",") if n.strip()] or None
    except ValueError:
        raise BadRequest(f"Senses are given by their numbers: {request.nodes}")
    result = sense_periods(
        db,
        q,
        filters.count_lemmas,
        filters.properties,
        dict(request.metadata or {}),
        distance=collocate_distance(request),
        year_field=year_field,
        filter_words=filter_words,
        nodes=nodes,
    )
    if stopwords_missing:
        result = {**result, "stopwords_missing": True}
    return result
