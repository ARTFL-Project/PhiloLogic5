"""The senses the query's word is used in, and their course over time (see philologic.runtime.reports.word_senses)."""

from philologic.runtime.DB import DB
from philologic.runtime.reports.collocation import build_filter_list, collocate_distance
from philologic.runtime.reports.word_senses import word_senses
from philologic.runtime.sql_validation import validate_column


def get_word_senses(request, config):
    db = DB(config.db_path + "/data/")
    q = (request.q or "").strip()
    if not q:
        return {"n_hits": 0, "n_sentences": 0, "n_works": 0, "nodes": [], "time": None, "filter_list": []}
    count_lemmas = "lemma:" in q  # as the collocation report counts them

    attribute = attribute_value = None
    filter_words = None
    stopwords_missing = False
    choice = (request.colloc_filter_choice or "").strip()
    if choice == "attribute":
        attribute = (request.q_attribute or "").strip() or None
        attribute_value = (request.q_attribute_value or "").strip() or None
        if attribute is None or attribute_value is None:
            attribute = attribute_value = None
    elif choice != "nofilter":  # the collocation report's filter: its stopwords or most common words
        filter_words = build_filter_list(request, config, count_lemmas)
        if filter_words is None:  # the stopwords list was not found: nothing is filtered
            stopwords_missing = True
    try:
        year_field = validate_column(config.time_series_year_field, db)
    except Exception:  # no year to place the sentences in time
        year_field = None
    # Rarer words weigh more, and only the query's collocates mark senses, whatever the filter
    result = word_senses(
        db,
        q,
        count_lemmas,
        attribute,
        attribute_value,
        dict(request.metadata or {}),
        distance=collocate_distance(request),
        year_field=year_field,
        filter_words=filter_words,
    )
    if stopwords_missing:
        result = {**result, "stopwords_missing": True}
    return result
