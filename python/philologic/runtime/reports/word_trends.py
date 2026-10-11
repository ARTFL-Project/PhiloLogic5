"""What is new in the use of a word over time: the words that newly go with it, and the works ahead of their time.

Rising words: the query's collocates (those markedly more frequent around it than in the corpus, outside the
collocation filter), in its sentences or just before or after it, whose rate in its sentences rises at some year
(word_senses.rise over the years, its sentences the trials, their dispersion over the works taken out), and
rises more than the word does in the whole corpus at that year: a word that rises everywhere (a spelling, a fashion
of the time) is not new around the query. A word the corpus does not have at all before its rise is kept, marked
new in the corpus (a word coined then: révolutionnaire). Rising words that go together in the query's sentences more
than chance would have them are one emerging use, with its rise, its course over time and its earliest sentences:
where it may have started.

Ahead of their time: a work whose use of the query is unlike the works before it and like those after it, as
resonance is novelty against the past less transience against the future (Barron, Huang, Spang and DeDeo 2018). Each
work's words around the query are set against those of each work of the span before its year and of the span after
(not its author's), and it is ahead when the works after are, as a rule, closer to it than those before (Mann and
Whitney's test over the works, at a threshold made stricter for the many works tested): taken up by several works,
not one that happens to be close. Its words are those the works after have more of than the works before; its
sentences, those that have most of them; the works after closest to it, those that may have taken it up.
"""

import hashlib
import json
import math
import os
import tempfile
import time

import numba
import numpy as np
from scipy.sparse import csr_matrix, diags, hstack
from scipy.special import ndtr, ndtri, xlogy

from philologic.runtime.reports import word_senses as ws
from philologic.runtime.reports.time_series import _corpus_year_counts, _get_doc_year_data

_CACHE_TAG = b"trends6\0"  # bump when the result changes
_SOURCES = 10  # the sentences sent for each use or work: where it may have started
_WORDS = 12  # the words sent for each use or work, the strongest first
_FOLLOWERS = 3  # the works after a work ahead of its time sent with it, the closest first
_WORKS = 50  # the works ahead of their time sent, clearly or perhaps, the first of each


# ----------------------------- Shared ------------------------------


def _cached(db, kind, parts, compute):
    """A report kept in the hitlist directory under the hash of its arguments, computed when it is not there."""
    h = hashlib.sha1(_CACHE_TAG + kind.encode("utf-8"))
    for part in parts:
        h.update(json.dumps(part, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"\0")
    path = os.path.join(db.hitlist_dir, f"{h.hexdigest()}.{kind}.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            pass
    began = time.perf_counter()
    result = compute()
    result["seconds"] = round(time.perf_counter() - began, 2)
    text = json.dumps(result)
    fd, tmp = tempfile.mkstemp(dir=db.hitlist_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except OSError:
        if os.path.exists(tmp):
            os.remove(tmp)
    return json.loads(text)  # as it is read back: plain numbers, not numpy's


def _dated(db, sentences, year_field, metadata):
    """The year of each of the query's sentences (0 when its work has none), and the words of each year."""
    year_array, year_words, _, _, _ = _get_doc_year_data(db, year_field)
    docs = sentences["docs"]
    years = np.where(docs < len(year_array), year_array[np.clip(docs, 0, len(year_array) - 1)], 0).astype(np.int64)
    if metadata:
        year_words, _ = _corpus_year_counts(db, year_array, metadata)
    return years, year_array, year_words


def _work_years(years, works):
    """The works of the dated sentences, each with its year: (index of each dated sentence's work, the year of each
    work as an index into the years, the years, the number of sentences of each work)."""
    dated = np.flatnonzero(years > 0)
    year_ids, year_of = np.unique(years[dated], return_inverse=True)
    work_ids, work_of = np.unique(works[dated], return_inverse=True)
    work_year = np.zeros(len(work_ids), dtype=np.int64)
    work_year[work_of] = year_of
    return dated, work_of, work_year, year_ids, np.bincount(work_of).astype(np.float64)


def _work_counts(B, dated, work_of, n_works):
    """For each column of B (sentences x things, 0 or 1), how many sentences of each work have it (works x columns)."""
    W = csr_matrix((np.ones(len(dated)), (work_of, np.arange(len(dated)))), shape=(n_works, len(dated)))
    return np.asarray((W @ B[dated]).todense())


def _by_year(per_work, work_year, n_years):
    """Counts of each work summed by year (years x columns)."""
    Y = csr_matrix((np.ones(len(work_year)), (work_year, np.arange(len(work_year)))), shape=(n_years, len(work_year)))
    return np.asarray(Y @ per_work)


def _dispersion(k, n):
    """How much more the counts of works (k of their n sentences) vary than sentences each drawn alike would make
    them (Pearson's chi-square over the works, by its degrees of freedom), at least 1: the sentences of a work go
    together."""
    p = k.sum() / max(n.sum(), 1.0)
    if not 0.0 < p < 1.0 or len(n) < 2:
        return 1.0
    return max(1.0, float(((k - n * p) ** 2 / (n * p * (1.0 - p))).sum()) / (len(n) - 1))


def _rises(k, n, work_year, n_years):
    """Where a count of the works' sentences rises (word_senses.rise over the years, the sentences the trials, their
    dispersion over the works taken out), and the dispersion: ((cut, before, after, G2) or None, dispersion)."""
    dispersion = _dispersion(k, n)
    k_by_year = _by_year(k[:, None], work_year, n_years)[:, 0]
    n_by_year = _by_year(n[:, None], work_year, n_years)[:, 0]
    return ws.rise(k_by_year, n_by_year, dispersion, len(n)), dispersion


@numba.njit(cache=True, nogil=True)
def _count_by_year(tokens, offsets, sent_year, wanted, n_wanted, y0, n_years):
    """How often each wanted word (wanted[id]: its column, or -1) is in the corpus each year, and all words."""
    out = np.zeros((n_wanted, n_years), dtype=np.int64)
    words = np.zeros(n_years, dtype=np.int64)
    for s in range(len(offsets) - 1):
        y = sent_year[s]
        if y < y0 or y >= y0 + n_years:
            continue
        yi = y - y0
        words[yi] += offsets[s + 1] - offsets[s]
        for k in range(offsets[s], offsets[s + 1]):
            j = wanted[tokens[k]]
            if j >= 0:
                out[j, yi] += 1
    return out, words


def _corpus_by_year(data_path, count_lemmas, ids, year_array, y0, n_years):
    """How often each word (vocabulary ids) is in the whole corpus each year from y0, and all words each year."""
    colloc_dir = os.path.join(data_path, "collocations")
    tokens = np.load(os.path.join(colloc_dir, "attr_lemma_ids.npy" if count_lemmas else "token_ids.npy"), mmap_mode="r")
    offsets = np.asarray(np.load(os.path.join(colloc_dir, "sent_offsets.npy"), mmap_mode="r"), dtype=np.int64)
    keys = np.load(os.path.join(colloc_dir, "sent_keys_native.npy"), mmap_mode="r")
    docs = np.asarray(keys[:, 0]).astype(np.int64)
    sent_year = np.where(docs < len(year_array), year_array[np.clip(docs, 0, len(year_array) - 1)], 0).astype(np.int64)
    n_vocab = len(ws._vocabulary(data_path, count_lemmas)[1]) - 1
    wanted = np.full(n_vocab, -1, dtype=np.int64)
    wanted[ids] = np.arange(len(ids))
    return _count_by_year(tokens, offsets, sent_year, wanted, len(ids), y0, n_years)


def _g2_over(k, n, e):
    """G2 of k successes in n trials against a rate e, when k is over n * e (else 0)."""
    if n <= 0 or not 0.0 < e < 1.0 or k <= n * e:
        return 0.0
    return 2.0 * (xlogy(k, k / (n * e)) + xlogy(n - k, (n - k) / (n * (1.0 - e))))


def _associations(together, alone, n):
    """For words of sentences (together: how many sentences have both, alone: each), whether each pair goes together
    more than chance would have them (G2 of the 2 x 2 table above the log of the number of sentences, the threshold of
    the Bayesian information criterion), and their normalized pointwise mutual information."""
    a, b = alone[:, None], alone[None, :]
    cells = [together, a - together, b - together, n - a - b + together]
    expected = [a * b / n, a * (n - b) / n, (n - a) * b / n, (n - a) * (n - b) / n]
    g2 = 2.0 * sum(xlogy(c, c / np.maximum(e, 1e-300)) for c, e in zip(cells, expected))
    goes_with = (together * n > a * b) & (g2 > math.log(n))
    np.fill_diagonal(goes_with, False)
    p11 = together / n
    with np.errstate(divide="ignore", invalid="ignore"):
        affinity = np.where(p11 > 0, np.log(p11 * n * n / (a * b)) / -np.log(p11), -1.0)
    return goes_with, affinity


def _earliest(rows, years, first_hit, score):
    """The earliest of rows (sentences), the higher score first within a year: (hit, year) of _SOURCES of them."""
    order = rows[np.lexsort((-score[rows], years[rows]))]
    return [{"hit": int(first_hit[u]), "year": int(years[u])} for u in order[:_SOURCES]]


# ----------------------------- Rising words ------------------------------


def rising_words(
    db,
    q,
    count_lemmas,
    attribute=None,
    attribute_value=None,
    metadata=None,
    distance=None,
    year_field=None,
    filter_words=None,
):
    """The words that newly go with the query, grouped into emerging uses (see the module), as JSON-ready data,
    cached in the hitlist directory."""
    metadata = {k: v for k, v in (metadata or {}).items() if v}
    filter_words = sorted(set(filter_words or []))
    parts = [q, bool(count_lemmas), attribute, attribute_value, metadata, distance, year_field, filter_words]

    def compute():
        result = _rising_words(
            db, q, count_lemmas, attribute, attribute_value, metadata, distance, year_field, filter_words
        )
        result["filter_list"] = filter_words
        return result

    return _cached(db, "rising", parts, compute)


def _rising_words(db, q, count_lemmas, attribute, attribute_value, metadata, distance, year_field, filter_words):
    sentences = ws.query_sentences(db, q, count_lemmas, attribute, attribute_value, metadata, distance)
    empty = {"n_hits": 0, "n_sentences": 0, "n_works": 0, "uses": [], "time": None}
    if sentences is None:
        return empty
    n = sentences["n_units"]
    docs = sentences["docs"]
    base = {"n_hits": int(sentences["n_hits"]), "n_sentences": int(n), "n_works": int(len(np.unique(docs)))}
    if not year_field:
        return {**empty, **base}
    years, year_array, year_words = _dated(db, sentences, year_field, metadata)
    if len(np.unique(years[years > 0])) < 2:
        return {**empty, **base}
    channels, names = ws._senses_channels(sentences, count_lemmas)
    common = ws._filtered(sentences, filter_words)
    dated, work_of, work_year, year_ids, work_n = _work_years(years, docs)
    sentences_by_year = _by_year(work_n[:, None], work_year, len(year_ids))[:, 0]

    # The candidates: the query's collocates outside the filter, in each group of words
    candidates = []  # (group, local index, vocabulary id)
    blocks = []
    for ci, (X, p, _, _) in enumerate(channels):
        counts = np.asarray(X.sum(axis=0)).ravel()
        cols = ws.markedly_more_frequent(counts, p)[0]
        cols = np.sort(cols[~common[names[ci][cols]]])
        candidates.extend((ci, int(c), int(names[ci][c])) for c in cols)
        if len(cols):
            block = csr_matrix(X)[:, cols].tocsr()
            block.data = np.ones_like(block.data, dtype=np.float64)  # whether a sentence has the word
            blocks.append(block)
    if not candidates:
        return {**empty, **base}
    B = hstack(blocks).tocsr()
    per_work = _work_counts(B, dated, work_of, len(work_n))

    # Each candidate's rise around the query, against its course in the whole corpus
    y0 = int(year_ids[0])
    ids = np.unique([v for *_, v in candidates])
    corpus, words = _corpus_by_year(
        sentences["data_path"], count_lemmas, ids, year_array, y0, int(year_ids[-1]) - y0 + 1
    )
    corpus_of = {int(v): k for k, v in enumerate(ids)}
    rising = []
    for j, (ci, c, v) in enumerate(candidates):
        found, dispersion = _rises(per_work[:, j], work_n, work_year, len(year_ids))
        if found is None:
            continue
        cut, before, after, g2 = found
        onset = int(year_ids[cut])
        at = onset - y0
        c0, c1 = corpus[corpus_of[v], :at].sum(), corpus[corpus_of[v], at:].sum()
        w0, w1 = words[:at].sum(), words[at:].sum()
        new_word = c0 == 0
        if not new_word:
            # The rate after that the corpus's course would give: the rate before, times the word's rise in the corpus
            expected = before * (c1 / max(w1, 1)) / (c0 / max(w0, 1))
            k, m = per_work[work_year >= cut, j].sum(), sentences_by_year[cut:].sum()
            works_after = int((work_year >= cut).sum())
            if expected >= 1.0 or _g2_over(k, m, expected) / dispersion <= math.log(max(works_after, 2)):
                continue
        rising.append(
            {
                "j": j,
                "group": ci,
                "local": c,
                "id": v,
                "onset": onset,
                "before": before,
                "after": after,
                "g2": g2,
                "new_word": bool(new_word),
                "corpus_before": c0 / max(w0, 1),
                "corpus_after": c1 / max(w1, 1),
            }
        )
    timeline_start, timeline_end = int(year_ids[0]), int(year_ids[-1])
    if not rising:
        return {
            **base,
            "uses": [],
            "time": {"start": timeline_start, "end": timeline_end},
            "candidates": len(candidates),
        }

    # Rising words that go together in the query's sentences more than chance would have them, grouped as HyperLex
    # groups the words of a sense: from the strongest rise down, a word that goes with no hub yet is a hub; every
    # other word joins the hub it goes with most (normalized pointwise mutual information); one with none stays alone
    R = B[:, [r["j"] for r in rising]].tocsc()
    together = np.asarray((R.T @ R).todense())
    alone = np.diag(together).astype(np.float64)
    goes_with, affinity = _associations(together, alone, n)
    order = sorted(range(len(rising)), key=lambda a: -rising[a]["g2"])
    hubs = []
    for a in order:
        if not any(goes_with[a, h] for h in hubs):
            hubs.append(a)
    groups = {h: [h] for h in hubs}
    for a in order:
        if a in groups:
            continue
        linked = [h for h in hubs if goes_with[a, h]]
        if linked:
            groups[max(linked, key=lambda h: affinity[a, h])].append(a)

    # Each use: its words, its rise, its course over time and its earliest sentences
    phrases = ws._Phrases(db, sentences)
    near = phrases.near(np.arange(n))
    vocab_blob, vocab_offsets = sentences["v_blob"], sentences["v_offsets"]

    def word_of(r):
        s = vocab_blob[int(vocab_offsets[r["id"]]) : int(vocab_offsets[r["id"] + 1])].decode("utf-8", errors="replace")
        item = {
            "word": s[6:] if s.startswith("lemma:") else s,
            "group": r["group"],
            "onset": r["onset"],
            "before": round(r["before"], 4),
            "after": round(r["after"], 4),
            "new_word": r["new_word"],
            "corpus_before": round(r["corpus_before"] * 1e6, 2),
            "corpus_after": round(r["corpus_after"] * 1e6, 2),
        }
        if r["group"] > 0:
            phrase = phrases.text(r["group"], near.get(r["group"], {}).get(r["id"]))
            if phrase:
                item["phrase"] = phrase
        return item

    uses = []
    indicators = []
    for members in groups.values():
        members = sorted(members, key=lambda a: -rising[a]["g2"])
        cols = [rising[a]["j"] for a in members]
        has = np.asarray(B[:, cols].sum(axis=1)).ravel()  # how many of the use's words each sentence has
        # Its sentences: those with two of its words (its one word, when it has one), not one of many old words
        present = (has >= min(2, len(members))).astype(np.float64)
        found, _ = _rises(
            _work_counts(csr_matrix(present[:, None]), dated, work_of, len(work_n))[:, 0],
            work_n,
            work_year,
            len(year_ids),
        )
        rows = np.flatnonzero((present > 0) & (years > 0))
        if not len(rows):
            continue
        # Without a rise of its own (too few sentences to tell), its first appearance
        onset = int(year_ids[found[0]]) if found else int(years[rows].min())
        uses.append(
            {
                "words": [word_of(rising[a]) for a in members[:_WORDS]],
                "n_words": len(members),
                "onset": onset,
                "rises": found is not None,
                "before": round(found[1], 4) if found else None,
                "after": round(found[2], 4) if found else None,
                "sentences": int(present.sum()),
                "works": int(len(np.unique(docs[present > 0]))),
                "sentences_before": int(((present > 0) & (years > 0) & (years < onset)).sum()),
                "sources": _earliest(rows, years, sentences["first_hit"], has),
            }
        )
        indicators.append(present)
    order = sorted(range(len(uses)), key=lambda u: (uses[u]["onset"], -uses[u]["n_words"]))
    uses = [uses[u] for u in order]
    curves = ws.sense_curves(np.array([indicators[u] for u in order]).T, years, docs, year_words)
    return {**base, "uses": uses, "time": curves, "candidates": len(candidates), "rising": len(rising)}


# ----------------------------- Ahead of their time ------------------------------


def ahead_of_time(
    db,
    q,
    count_lemmas,
    attribute=None,
    attribute_value=None,
    metadata=None,
    distance=None,
    year_field=None,
    filter_words=None,
    span=10,
):
    """The works whose use of the query the works after them take up (see the module), as JSON-ready data, cached in
    the hitlist directory. span: the years before and after a work it is set against (the database's span for time
    series)."""
    metadata = {k: v for k, v in (metadata or {}).items() if v}
    filter_words = sorted(set(filter_words or []))
    span = max(1, int(span or 10))
    parts = [q, bool(count_lemmas), attribute, attribute_value, metadata, distance, year_field, filter_words, span]

    def compute():
        result = _ahead_of_time(
            db, q, count_lemmas, attribute, attribute_value, metadata, distance, year_field, filter_words, span
        )
        result["filter_list"] = filter_words
        return result

    return _cached(db, "ahead", parts, compute)


def _works_metadata(db):
    """{the first number of a work's philo_id: (its author, its title)}, empty strings for what the database lacks."""
    cursor = db.dbh.cursor()
    columns = {row[1] for row in cursor.execute("PRAGMA table_info(toms)")}
    fields = [f if f in columns else "''" for f in ("author", "title")]
    cursor.execute(f"SELECT philo_id, {fields[0]}, {fields[1]} FROM toms WHERE philo_type='doc'")
    return {int(pid.split(" ")[0]): (str(author or ""), str(title or "")) for pid, author, title in cursor.fetchall()}


def _rank_test(before, after):
    """Mann-Whitney's test of whether the values after are, as a rule, higher than those before: (z, the share of
    the pairs (one after, one before) in which the one after is higher, ties counting half)."""
    nb, na = len(before), len(after)
    values = np.sort(np.concatenate([before, after]))
    ranks = (np.searchsorted(values, after, "left") + np.searchsorted(values, after, "right") + 1) / 2.0
    u = float(ranks.sum()) - na * (na + 1) / 2.0
    ties = np.diff(np.flatnonzero(np.r_[True, values[1:] != values[:-1], True])).astype(np.float64)
    n = nb + na
    var = na * nb / 12.0 * ((n + 1) - float((ties**3 - ties).sum()) / (n * (n - 1)))
    z = (u - na * nb / 2.0) / math.sqrt(var) if var > 0 else 0.0
    return z, u / (na * nb)


def _ahead_of_time(db, q, count_lemmas, attribute, attribute_value, metadata, distance, year_field, filter_words, span):
    sentences = ws.query_sentences(db, q, count_lemmas, attribute, attribute_value, metadata, distance)
    empty = {"n_hits": 0, "n_sentences": 0, "n_works": 0, "clear": [], "perhaps": [], "tested": 0, "periods": []}
    empty["span"] = span
    if sentences is None:
        return empty
    docs = sentences["docs"]
    base = {"n_hits": int(sentences["n_hits"]), "n_sentences": int(sentences["n_units"])}
    base["n_works"] = int(len(np.unique(docs)))
    if not year_field:
        return {**empty, **base}
    years, _, _ = _dated(db, sentences, year_field, metadata)
    dated = np.flatnonzero(years > 0)
    if len(np.unique(years[dated])) < 2:
        return {**empty, **base}

    # Each work's words around the query, each weighed as the senses weigh its evidence (by its inverse document
    # frequency): only the query's collocates (as only they may mark a sense), and of those of the sentence, not the
    # filter's. The other words tell of the language of a time (its spellings, its common words), not of the query's use
    channels, names = ws._senses_channels(sentences, count_lemmas)
    unshown = ws._filtered(sentences, filter_words)
    work_ids, work_of = np.unique(docs[dated], return_inverse=True)
    n_works = len(work_ids)
    work_year = np.zeros(n_works, dtype=np.int64)
    work_year[work_of] = years[dated]
    blocks = []
    for ci, c in enumerate(channels):
        X = csr_matrix(c[0])
        weights = np.zeros(X.shape[1])
        collocates = ws.markedly_more_frequent(np.asarray(X.sum(axis=0)).ravel(), np.asarray(c[1]))[0]
        weights[collocates] = np.asarray(c[3], dtype=np.float64)[collocates]
        if ci == 0:
            weights[unshown[names[0]]] = 0.0
        blocks.append(X[dated] @ diags(weights))
    offsets = np.cumsum([0] + [b.shape[1] for b in blocks])
    by_sentence = hstack(blocks).tocsr()
    to_work = csr_matrix((np.ones(len(dated)), (work_of, np.arange(len(dated)))), shape=(n_works, len(dated)))
    V = (to_work @ by_sentence).tocsr()
    norms = np.sqrt(np.asarray(V.multiply(V).sum(axis=1)).ravel())
    V = (diags(1.0 / np.where(norms > 0, norms, 1.0)) @ V).tocsr()
    # How close one work is to another (the cosine of their words), relative to how close all works are to the
    # other on average: a work with many sentences is a little like every work, and works with more of them after
    # (a word more frequent then) would make every work look ahead
    level = np.asarray(V @ np.asarray(V.mean(axis=0)).ravel()).ravel()
    level = np.where(level > 0, level, 1.0)
    metadata_of = _works_metadata(db)
    authors = np.array([metadata_of.get(int(w), ("", ""))[0] for w in work_ids], dtype=object)

    # Each work against the works of the span before its year and those of the span after (not its author's): it is
    # ahead of its time when the works after are, as a rule, closer to it than those before
    def compared(i, closeness):
        """The works before i and after it, and how close i is to each, from its closeness to the works near."""
        y = work_year[i]
        near = np.flatnonzero((np.abs(work_year - y) <= span) & (work_year != y))
        keep = authors[near] != authors[i] if authors[i] else np.ones(len(near), dtype=np.bool_)
        later = work_year[near] > y
        return near[keep & ~later], near[keep & later], closeness[keep & ~later], closeness[keep & later]

    tested = []
    for y in np.unique(work_year):
        here = np.flatnonzero(work_year == y)
        near = np.flatnonzero((np.abs(work_year - y) <= span) & (work_year != y))
        if len(near) < 4:
            continue
        closeness = (V[here] @ V[near].T).toarray() / level[near][None, :]
        for k, i in enumerate(here):
            before, after, close_before, close_after = compared(i, closeness[k])
            if len(before) >= 2 and len(after) >= 2:
                tested.append((i, *_rank_test(close_before, close_after)))
    # Clearly ahead: as strict as a split's test (EVIDENCE standard errors), for all the works tested together
    # (Bonferroni); perhaps ahead: as strict for a single work, which some works pass by chance alone
    threshold = float(ndtri(1.0 - ndtr(-ws.EVIDENCE) / max(1, len(tested))))
    clear = sorted((t for t in tested if t[1] > threshold), key=lambda t: (-t[2], -t[1]))
    perhaps = sorted((t for t in tested if ws.EVIDENCE < t[1] <= threshold), key=lambda t: -t[1])

    vocab_blob, vocab_offsets = sentences["v_blob"], sentences["v_offsets"]
    phrases = ws._Phrases(db, sentences)
    rows_of = np.split(np.argsort(work_of, kind="stable"), np.cumsum(np.bincount(work_of, minlength=n_works))[:-1])

    def word_item(j, near):
        ci = int(np.searchsorted(offsets, j, "right") - 1)
        vid = int(names[ci][j - offsets[ci]])
        s = vocab_blob[int(vocab_offsets[vid]) : int(vocab_offsets[vid + 1])].decode("utf-8", errors="replace")
        item = {"word": s[6:] if s.startswith("lemma:") else s, "group": ci}
        phrase = phrases.text(ci, near.get(ci, {}).get(vid)) if ci > 0 else None
        if phrase:
            item["phrase"] = phrase
        return item

    def described(i):
        return {
            "doc": int(work_ids[i]),
            "year": int(work_year[i]),
            "author": authors[i] or "",
            "title": metadata_of.get(int(work_ids[i]), ("", ""))[1],
        }

    def work_item(i, z, share):
        near = np.flatnonzero((np.abs(work_year - work_year[i]) <= span) & (work_year != work_year[i]))
        before, after, _, close_after = compared(i, (V[i] @ V[near].T).toarray().ravel() / level[near])
        # What the works after have more of than those before, among its words; its sentences that have most of it
        drift = np.asarray(V[after].mean(axis=0)).ravel() - np.asarray(V[before].mean(axis=0)).ravel()
        mine = V[i].toarray().ravel() * drift
        rows = dated[rows_of[i]]
        near_words = phrases.near(rows)
        ahead = np.asarray(by_sentence[rows_of[i]] @ drift).ravel()
        order = np.argsort(-ahead, kind="stable")[:_SOURCES]
        return {
            **described(i),
            "z": round(z, 2),
            "share": round(share, 3),
            "before": int(len(before)),
            "after": int(len(after)),
            "sentences": int(len(rows)),
            "words": [word_item(int(j), near_words) for j in np.argsort(-mine)[:_WORDS] if mine[j] > 0],
            "sources": [{"hit": int(sentences["first_hit"][rows[o]]), "year": int(work_year[i])} for o in order],
            # The works after closest to it: those that may have taken it up
            "followers": [described(after[f]) for f in np.argsort(-close_after)[:_FOLLOWERS]],
        }

    # How many works were tested, and found ahead, period by period (the span's)
    first, last = int(work_year.min()), int(work_year.max())
    periods = []
    for p0 in range(first - first % span, last + 1, span):
        inside = [t for t in tested if p0 <= work_year[t[0]] < p0 + span]
        periods.append(
            {
                "start": p0,
                "end": p0 + span - 1,
                "tested": len(inside),
                "clear": sum(t[1] > threshold for t in inside),
                "perhaps": sum(ws.EVIDENCE < t[1] <= threshold for t in inside),
            }
        )
    return {
        **base,
        "clear": [work_item(*t) for t in clear[:_WORKS]],
        "perhaps": [work_item(*t) for t in perhaps[:_WORKS]],
        "n_clear": len(clear),
        "n_perhaps": len(perhaps),
        "tested": len(tested),
        "by_chance": round(len(tested) * float(ndtr(-ws.EVIDENCE)), 1),
        "threshold": round(threshold, 2),
        "evidence": ws.EVIDENCE,
        "span": span,
        "periods": periods,
    }
