"""The words that newly go with the query's word, grouped into emerging uses (see philologic.runtime.reports.word_trends)."""

from philologic.runtime.DB import DB
from philologic.runtime.reports.collocation import build_filter_list, collocate_distance
from philologic.runtime.reports.word_trends import rising_words
from philologic.runtime.sql_validation import validate_column


def get_rising_words(request, config):
    db = DB(config.db_path + "/data/")
    q = (request.q or "").strip()
    if not q:
        return {"n_hits": 0, "n_sentences": 0, "n_works": 0, "uses": [], "time": None, "filter_list": []}
    count_lemmas = "lemma:" in q
    attribute = attribute_value = None
    filter_words = None
    stopwords_missing = False
    choice = (request.colloc_filter_choice or "").strip()
    if choice == "attribute":
        attribute = (request.q_attribute or "").strip() or None
        attribute_value = (request.q_attribute_value or "").strip() or None
        if attribute is None or attribute_value is None:
            attribute = attribute_value = None
    elif choice != "nofilter":  # the collocation report's filter: its words are no candidates
        filter_words = build_filter_list(request, config, count_lemmas)
        if filter_words is None:
            stopwords_missing = True
    try:
        year_field = validate_column(config.time_series_year_field, db)
    except Exception:  # no year: nothing rises
        year_field = None
    result = rising_words(
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
