"""The senses the query's word is used in, and their course over time (see philologic.runtime.reports.word_senses)."""

from philologic.runtime.DB import DB
from philologic.runtime.reports.collocation import collocate_distance, collocate_filters, common_words
from philologic.runtime.reports.word_senses import word_senses
from philologic.runtime.sql_validation import validate_column


def get_word_senses(request, config):
    db = DB(config.db_path + "/data/")
    q = (request.q or "").strip()
    if not q:
        return {"n_hits": 0, "n_sentences": 0, "n_works": 0, "nodes": [], "time": None, "filter_list": []}
    filters = collocate_filters(request, db)  # as the collocation report counts them
    filter_words, stopwords_missing = common_words(request, config, filters)
    try:
        year_field = validate_column(config.time_series_year_field, db)
    except Exception:  # no year to place the sentences in time
        year_field = None
    # The filter's words are left out of the sentence's words (next to the query they are read, but seed no split);
    # rarer words weigh more, and only the query's collocates mark senses
    result = word_senses(
        db,
        q,
        filters.count_lemmas,
        filters.properties,
        dict(request.metadata or {}),
        distance=collocate_distance(request),
        year_field=year_field,
        filter_words=filter_words,
    )
    if stopwords_missing:
        result = {**result, "stopwords_missing": True}
    return result
