"""Thread detection for a query (collocation evolution).

Returns thread decomposition: each thread has a per-year intensity curve
and a ranked word list. Built on a global HDBSCAN clustering of the
query's collocate vocabulary, projected to time bins.
"""

import math
import os

from philologic.runtime.DB import DB
from philologic.runtime.threads import detect_threads
from philologic.runtime.reports.collocation import build_filter_list


# Per-worker cache of corpus-IDF maps, keyed by (db_path, count_lemmas).
# Loaded once per process — the frequency files are ~600-700k lines.
_corpus_idf_cache = {}


def _load_corpus_idf(db_path, count_lemmas):
    """Return (idf_map, default_idf) for corpus-IDF c-TF-IDF label ranking.

    Surface forms: word_frequencies is `word\\tcount`, idf = log(total/count).
    Lemmas: the `lemmas` file is frequency-rank-sorted with no counts, so we
    use a Zipf rank proxy, idf = log(rank + 2). The default (for words absent
    from the table) is the median idf — neither promoted nor demoted.
    """
    key = (db_path, count_lemmas)
    if key in _corpus_idf_cache:
        return _corpus_idf_cache[key]

    freq_dir = os.path.join(db_path, "data", "frequencies")
    idf_map = {}
    default_idf = 1.0
    try:
        if count_lemmas:
            path = os.path.join(freq_dir, "lemmas")
            with open(path, encoding="utf8") as fh:
                for i, line in enumerate(fh):
                    key_w = line.strip()
                    if key_w.startswith("lemma:"):
                        key_w = key_w[6:]
                    if key_w:
                        idf_map[key_w] = math.log(i + 2)
        else:
            path = os.path.join(freq_dir, "word_frequencies")
            total = 0
            tmp = {}
            with open(path, encoding="utf8") as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) == 2:
                        try:
                            c = int(parts[1])
                        except ValueError:
                            continue
                        tmp[parts[0]] = c
                        total += c
            if total > 0:
                idf_map = {w: math.log(total / c) for w, c in tmp.items()}
        if idf_map:
            vals = sorted(idf_map.values())
            default_idf = vals[len(vals) // 2]
    except (OSError, ValueError):
        idf_map = {}
        default_idf = 1.0

    _corpus_idf_cache[key] = (idf_map, default_idf)
    return idf_map, default_idf


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
    # "Number of themes" override; absent/blank/"auto" → plateau auto.
    n_clusters = _int_or_none("n_clusters", 2, 30)

    result = detect_threads(
        db, config.db_path + "/data", q, count_lemmas, attribute, attribute_value,
        metadata,
        stopwords=stopwords,
        # Lazy: detect_threads only loads the corpus-idf map on a cache miss,
        # so the common rerun (cache hit) skips the ~380 ms full-map load
        # entirely — on every worker, warm or cold.
        corpus_idf_loader=lambda: _load_corpus_idf(config.db_path, count_lemmas),
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
