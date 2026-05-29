"""Thread detection for a query (collocation evolution).

Returns thread decomposition: each thread has a per-year intensity curve and an
anchor-ranked word list. Built on a HyperLex sense induction of the query's
collocate vocabulary, projected to time bins.
"""

from philologic.runtime.DB import DB
from philologic.runtime.threads import detect_threads
from philologic.runtime.reports.collocation import build_filter_list


def _load_stopwords(request, config, count_lemmas):
    """Return ``(stop_set, raw_words)`` for the current corpus + UI settings,
    using the same logic as the collocation report's filter_list.

    ``stop_set`` has the ``lemma:`` prefix stripped (thread candidates are
    stored without it); ``raw_words`` keeps the prefix so the UI can display
    the filtered words exactly as the frequency view does.
    """
    choice = (request.colloc_filter_choice or "").strip()
    if choice == "nofilter" or choice == "attribute":
        return set(), []
    try:
        words = build_filter_list(request, config, count_lemmas)
    except Exception:
        return set(), []
    stop = set()
    for w in words:
        if w.startswith("lemma:"):
            stop.add(w[6:])
        else:
            stop.add(w)
    return stop, words


def get_threads(request, config):
    db = DB(config.db_path + "/data/")

    q = (request.q or "").strip()
    if not q:
        return {"n_total_hits": 0, "threads": []}
    count_lemmas = q.startswith("lemma:")

    attribute = None
    attribute_value = None
    if (request.colloc_filter_choice or "") == "attribute":
        attribute = (request.q_attribute or "").strip() or None
        attribute_value = (request.q_attribute_value or "").strip() or None
        if attribute is None or attribute_value is None:
            attribute = attribute_value = None

    metadata = dict(request.metadata or {})
    stopwords, filter_words = _load_stopwords(request, config, count_lemmas)

    def _int_or_none(name, lo, hi):
        raw = getattr(request, name, "")
        try:
            v = int(raw) if raw not in (None, "") else None
        except (TypeError, ValueError):
            v = None
        if v is None:
            return None
        return max(lo, min(hi, v))

    top_n_threads = _int_or_none("top_n_threads", 1, 30)
    # "Number of themes": truncate to the top-N senses by mass (absent → all).
    n_clusters = _int_or_none("n_clusters", 2, 30)

    result = detect_threads(
        db, config.db_path + "/data", q, count_lemmas, attribute, attribute_value,
        metadata,
        stopwords=stopwords,
        top_n_threads=top_n_threads,
        n_clusters_override=n_clusters,
        # The network view is the spatial twin of the streamgraph — built from
        # the same clustering, so it ships in the same response (cheap: the
        # distance matrix and communities are already computed).
        include_graph=True,
    )
    # Surface the filtered words so the results summary's filter list matches
    # the frequency view (the streamgraph/word-map don't run the frequency
    # fetch that normally populates it).
    if isinstance(result, dict):
        result["filter_list"] = sorted(filter_words, key=str.lower)
    return result
