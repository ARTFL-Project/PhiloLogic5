"""How the words of each sense change over time: its works cut into as many periods as the data supports, each period
with the words new in it and the words fading, against all the periods before it, and its most representative works.

The senses are those of the senses tab (word_senses), at its default view; a sense's sentences are those clearly in it
(on its side of every split), dated. Its works, in year order, are cut into K periods of as many works (at year edges:
the works of a year stay together), for K = 2, 3, ...: K grows while every period after the first has at least one new
word, and stops at the first K where one has none; the periods are those of the last K where all had. A sense whose
two halves already show nothing new is one period.

New and fading words: the query's collocates (those markedly more frequent around it than in the corpus, as the senses
take them) markedly more, or less, frequent in a period than in all the periods before it: G2 of the 2 x 2 table (the
word and the other words, in the period and before it: Dunning's log-likelihood, as for the keywords of one corpus
against another) above the log of the words of both (the threshold of the Bayesian information criterion), and still
so without the most prolific author of the side the word is more frequent in (one author repeating a name does not make
it the period's). Words count once in a sentence; the filter's words are left out of the counts altogether. The first
period, with nothing before it, has instead the words that set it apart from the periods after it.

The most representative works of a period: those whose sentences the period's words (fitted without the work) predict
best against those of the periods before (after, for the first): the mean gain of their sentences less its standard
error, so that one lucky sentence does not outrank many; one work for an author.
"""

import hashlib
import json
import math
import os
import tempfile

import numba
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import xlogy

from philologic.runtime.reports import word_senses as ws
from philologic.runtime.reports.time_series import _get_doc_year_data

_CACHE_TAG = b"periods4\0"  # bump when the result changes (or the senses' membership: word_senses._CACHE_TAG)
_WORDS = 8  # the new words listed for a period, the strongest first
_FADING = 5  # the fading words listed for a period
_WORKS = 3  # the representative works of a period
_EXAMPLES = ws._WORD_EXAMPLES  # the sentences sent for a word or a work, the most typical of the period first


def sense_periods(
    db,
    q,
    count_lemmas,
    properties=(),
    metadata=None,
    distance=None,
    year_field=None,
    filter_words=None,
    nodes=None,
):
    """The periods of the query's senses (see the module), as JSON-ready data: of the senses nodes (their ids in the
    senses report), or of those of its default view. Each sense's periods are cached in the hitlist directory, so that
    the senses of another detail only cost their own."""
    metadata = {k: v for k, v in (metadata or {}).items() if v}
    filter_words = sorted(set(filter_words or []))
    args = (db, q, count_lemmas, properties, metadata, distance, year_field, filter_words)
    report, membership = ws.senses_with_membership(*args)
    slim = [
        {k: node[k] for k in ("id", "parent", "children", "sense", "rank", "share", "words", "before", "after")}
        for node in report["nodes"]
    ]
    for node in slim:  # the words a label needs, not their sentences
        for group in ("words", "before", "after"):
            node[group] = [{k: v for k, v in w.items() if k in ("word", "phrase")} for w in node[group][:6]]
    base = {k: report.get(k) for k in ("n_hits", "n_sentences", "n_works", "detail")}
    base["nodes"] = slim
    base["filter_list"] = filter_words
    if nodes is None:
        nodes = [n["id"] for n in ws.leaves_at(report["nodes"], report["detail"]) if n["sense"] is not None]
    nodes = [i for i in nodes if 0 <= i < len(report["nodes"]) and report["nodes"][i]["sense"] is not None]
    if not report["nodes"] or not year_field:
        return {**base, "senses": []}

    h = hashlib.sha1(_CACHE_TAG)
    for part in (q, bool(count_lemmas), properties, metadata, distance, year_field, filter_words):
        h.update(json.dumps(part, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"\0")
    paths = {i: os.path.join(db.hitlist_dir, f"{h.hexdigest()}.periods-{i}.json") for i in nodes}
    found = {}
    for i, path in paths.items():
        try:
            with open(path, encoding="utf-8") as f:
                found[i] = json.load(f)
        except (OSError, ValueError):
            pass
    missing = [i for i in nodes if i not in found]
    if missing:
        context = _Context(db, report, membership, *args[1:])
        for i in missing:
            text = json.dumps(context.periods(i))
            found[i] = json.loads(text)  # as it is read back: plain numbers
            fd, tmp = tempfile.mkstemp(dir=db.hitlist_dir, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(text)
                os.replace(tmp, paths[i])
            except OSError:  # a cache that can't be written is no cache
                if os.path.exists(tmp):
                    os.unlink(tmp)
    return {**base, "senses": [found[i] for i in nodes if found[i] is not None]}


class _Context:
    """What the periods of any sense of a query read: its sentences, their words and years, and the senses'
    membership."""

    def __init__(
        self,
        db,
        report,
        membership,
        q,
        count_lemmas,
        properties,
        metadata,
        distance,
        year_field,
        filter_words,
    ):
        self.report, self.membership, self.shared = report, membership, None
        sentences = ws.query_sentences(db, q, count_lemmas, properties, metadata, distance)
        if sentences is None:
            return
        try:
            year_array = _get_doc_year_data(db, year_field)[0]
        except Exception:  # no year field the database can read
            return
        docs = np.asarray(sentences["docs"])
        years = np.where(docs < len(year_array), year_array[np.clip(docs, 0, len(year_array) - 1)], 0).astype(np.int64)
        if len(np.unique(years[years > 0])) < 2:
            return
        self.years, self.docs = years, docs
        channels, names = ws._senses_channels(sentences, count_lemmas)
        common = ws._filtered(sentences, filter_words)
        Xall = [csr_matrix(c[0]) for c in channels]
        corpus = [ws.Dist(np.asarray(c[1], dtype=np.float64), float(c[2])) for c in channels]
        counts = [np.asarray(X.sum(axis=0)).ravel() for X in Xall]
        collocates = [np.isin(np.arange(len(c)), ws.markedly_more_frequent(c, d.p)[0]) for c, d in zip(counts, corpus)]
        self.shared = _Shared(
            db, sentences, channels, names, common, Xall, corpus, counts, collocates, years, year_array
        )

    def periods(self, i):
        """The periods of the sense of id i, or None when it has no dated sentences of two works."""
        if self.shared is None:
            return None
        rows = np.flatnonzero(self.membership[:, i] & (self.years > 0))
        if len(np.unique(self.docs[rows])) < 2:
            return None
        periods, tried = _Sense(self.shared, rows).periods()
        return {"id": i, "sense": self.report["nodes"][i]["sense"], "periods": periods, "tried": tried}


@numba.njit(cache=True, nogil=True)
def _count_by_year(tokens, offsets, sent_year, wanted, n_wanted, y0, n_years):
    """How often each wanted word (wanted[id]: its column, or -1) is in the corpus each year, and all words."""
    out = np.zeros((n_years, n_wanted), dtype=np.float64)
    words = np.zeros(n_years, dtype=np.float64)
    for s in range(len(offsets) - 1):
        y = sent_year[s]
        if y < y0 or y >= y0 + n_years:
            continue
        yi = y - y0
        words[yi] += offsets[s + 1] - offsets[s]
        for k in range(offsets[s], offsets[s + 1]):
            j = wanted[tokens[k]]
            if j >= 0:
                out[yi, j] += 1
    return out, words


class _Shared:
    """What every sense of the query reads: its words without the filter's, the authors of its works, the corpus's
    rate of each word each year (to tell how much of a word's rise the whole corpus had), and the phrases."""

    def __init__(self, db, sentences, channels, names, common, Xall, corpus, counts_all, collocates, years, year_array):
        self.names, self.corpus, self.years = names, corpus, years
        self.weights = [np.asarray(c[3], dtype=np.float64) for c in channels]
        self.discounts = [ws.discount(c) for c in counts_all]
        self.keep = [np.flatnonzero(~common[ids]) for ids in names]  # the words left in, by group
        self.X = [X[:, keep].tocsr() for X, keep in zip(Xall, self.keep)]
        self.listable = [collocates[ci][keep] for ci, keep in enumerate(self.keep)]
        self.sent_words = [np.asarray(X.sum(axis=1)).ravel() for X in self.X]
        self.first_hit = sentences["first_hit"]
        self.docs = np.asarray(sentences["docs"])
        cursor = db.dbh.cursor()
        columns = {row[1] for row in cursor.execute("PRAGMA table_info(toms)")}
        fields = [f if f in columns else "''" for f in ("author", "title")]
        cursor.execute(f"SELECT philo_id, {fields[0]}, {fields[1]} FROM toms WHERE philo_type='doc'")
        self.works = {int(pid.split(" ")[0]): (str(a or ""), str(t or "")) for pid, a, t in cursor.fetchall()}
        self.authors = np.array([self.works.get(int(d), ("", ""))[0] or f"#{int(d)}" for d in self.docs], dtype=object)
        self.year_array = year_array
        self.vocab = (sentences["v_blob"], sentences["v_offsets"])
        self.phrases = ws._Phrases(db, sentences)
        # The corpus's words each year, for the words of the query's sentences (every group has the same vocabulary)
        dated = years[years > 0]
        self.y0, n_years = int(dated.min()), int(dated.max() - dated.min() + 1)
        union = np.unique(np.concatenate([ids for ids in names]))
        colloc_dir = os.path.join(sentences["data_path"], "collocations")
        tokens = np.load(
            os.path.join(colloc_dir, "attr_lemma_ids.npy" if sentences["count_lemmas"] else "token_ids.npy"),
            mmap_mode="r",
        )
        offsets = np.asarray(np.load(os.path.join(colloc_dir, "sent_offsets.npy"), mmap_mode="r"), dtype=np.int64)
        keys = np.load(os.path.join(colloc_dir, "sent_keys_native.npy"), mmap_mode="r")
        sent_docs = np.asarray(keys[:, 0]).astype(np.int64)
        sent_year = np.where(
            sent_docs < len(year_array), year_array[np.clip(sent_docs, 0, len(year_array) - 1)], 0
        ).astype(np.int64)
        wanted = np.full(len(sentences["v_offsets"]) - 1, -1, dtype=np.int64)
        wanted[union] = np.arange(len(union))
        counts, words = _count_by_year(tokens, offsets, sent_year, wanted, len(union), self.y0, n_years)
        self.rates = counts / np.maximum(words, 1.0)[:, None]  # years x words of the union
        self.union_col = [np.searchsorted(union, ids[keep]) for ids, keep in zip(names, self.keep)]

    def word(self, ci, j):
        """The word (group ci, j among the words left in), as the senses tab names it."""
        vid = int(self.names[ci][self.keep[ci][j]])
        blob, offsets = self.vocab
        s = blob[int(offsets[vid]) : int(offsets[vid + 1])].decode("utf-8", errors="replace")
        return vid, (s[6:] if s.startswith("lemma:") else s)


class _Sense:
    """One sense's sentences in year order, and its periods."""

    def __init__(self, shared, rows):
        self.sh = shared
        rows = rows[np.argsort(shared.years[rows], kind="stable")]
        self.rows, self.yr, self.docs = rows, shared.years[rows], shared.docs[rows]
        self.authors = shared.authors[rows]
        self.X = [X[rows] for X in shared.X]
        self.sent_words = [w[rows] for w in shared.sent_words]

    # ---- the periods ----
    def cut(self, K):
        """The period of each sentence: the works in year order, K groups of about as many works, at year edges."""
        work_ids, first = np.unique(self.docs, return_index=True)
        order = np.lexsort((work_ids, self.yr[first]))
        years = self.yr[first][order]
        n = len(order)
        edges = [0]
        for k in range(1, K):
            target = round(k * n / K)
            changes = np.flatnonzero(years[1:] != years[:-1]) + 1  # where a year ends
            changes = changes[changes > edges[-1]]
            if not len(changes):
                break
            edge = int(changes[np.argmin(np.abs(changes - target))])
            if edge not in edges:
                edges.append(edge)
        edges.append(n)
        period_of_work = np.zeros(n, dtype=np.int64)  # by the works' index in work_ids
        for p, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            period_of_work[order[a:b]] = p
        return period_of_work[np.searchsorted(work_ids, self.docs)], len(edges) - 1

    def periods(self):
        """The periods of the largest K (from 2 up, to the first that fails) whose periods after the first all have a
        new word, described; and the Ks tried, with whether they held."""
        tried, best = [], None
        n_works = len(np.unique(self.docs))
        for K in range(2, n_works // 2 + 1):
            period, n = self.cut(K)
            if n < K:  # no more year edges to cut at
                break
            ok = all(self.has_new(period, k) for k in range(1, n))
            tried.append([K, bool(ok)])
            if not ok:
                break
            best = period
        if best is None:
            best = np.zeros(len(self.rows), dtype=np.int64)
        return self.describe(best), tried

    # ---- new and fading words ----
    def _counts(self, ci, rows):
        return np.asarray(self.X[ci][rows].sum(axis=0)).ravel()

    def _marked(self, ci, mine, other, more):
        """The words left in of group ci markedly more (more) or less frequent in rows mine than in rows other, the
        strongest first: (index among the words left in, G2, rate ratio), before the author guard."""
        a, c = self._counts(ci, mine), self._counts(ci, other)
        n1, n2 = float(a.sum()), float(c.sum())
        if n1 <= 0 or n2 <= 0:
            return [], a, c, n1, n2
        side = (a / n1 > c / n2) if more else (a / n1 < c / n2)
        cand = np.flatnonzero(self.sh.listable[ci] & side & ((a + c) > 0))
        g2 = _g2(a[cand], n1, c[cand], n2)
        ok = g2 > math.log(n1 + n2)
        order = np.argsort(-g2[ok], kind="stable")
        return list(zip(cand[ok][order], g2[ok][order])), a, c, n1, n2

    def _robust(self, ci, j, mine, other, more):
        """Whether word j still passes without the most prolific author of the side it is more frequent in."""
        side = mine if more else other
        col = self.X[ci][side][:, j].toarray().ravel()
        has = np.flatnonzero(col > 0)
        if not len(has):
            return False
        names, counts = np.unique(self.authors[side][has], return_counts=True)
        drop = self.authors[side] == names[np.argmax(counts)]
        a, n1 = self._counts(ci, mine)[j], float(self.sent_words[ci][mine].sum())
        c, n2 = self._counts(ci, other)[j], float(self.sent_words[ci][other].sum())
        if more:
            a, n1 = a - col[drop].sum(), n1 - self.sent_words[ci][side][drop].sum()
        else:
            c, n2 = c - col[drop].sum(), n2 - self.sent_words[ci][side][drop].sum()
        if n1 <= 0 or n2 <= 0 or (a / n1 > c / n2) != more:
            return False
        return float(_g2(np.array([a]), n1, np.array([c]), n2)[0]) > math.log(n1 + n2)

    def has_new(self, period, k):
        mine, past = np.flatnonzero(period == k), np.flatnonzero(period < k)
        for ci in range(len(self.X)):
            marked, *_ = self._marked(ci, mine, past, True)
            if any(self._robust(ci, j, mine, past, True) for j, _ in marked):
                return True
        return False

    def words(self, mine, other, more, n, typical=None):
        """The words of rows mine against rows other (see _marked, with the author guard), as the report sends them."""
        found = []
        for ci in range(len(self.X)):
            marked, a, c, n1, n2 = self._marked(ci, mine, other, more)
            kept = 0
            for j, g in marked:
                if kept == n:
                    break
                if not self._robust(ci, j, mine, other, more):
                    continue
                kept += 1
                found.append((float(g), ci, int(j), a, c, n1, n2))
        found.sort(key=lambda t: -t[0])
        out = []
        near = self.sh.phrases.near(self.rows[mine]) if more else {}
        e_mine, e_other = self.corpus_rates(mine), self.corpus_rates(other)
        for g, ci, j, a, c, n1, n2 in found[:n]:
            vid, word = self.sh.word(ci, j)
            ratio = (a[j] / n1) / (c[j] / n2) if c[j] > 0 else None
            col = self.sh.union_col[ci][j]
            corpus = e_mine[col] / e_other[col] if e_other[col] > 0 else None
            item = {"word": word, "group": ci, "g2": round(g, 1), "ratio": None if ratio is None else round(ratio, 3)}
            item["corpus"] = None if corpus is None else round(float(corpus), 3)
            if ci > 0:
                phrase = self.sh.phrases.text(ci, near.get(ci, {}).get(vid))
                if phrase:
                    item["phrase"] = phrase
            if more:  # the sentences of the period with the word, the most typical of the period first
                has = mine[self.X[ci][mine][:, j].toarray().ravel() > 0]
                if typical is not None:
                    has = has[np.argsort(-typical[has], kind="stable")]
                item["hits"] = [int(self.sh.first_hit[self.rows[r]]) for r in has[:_EXAMPLES]]
            out.append(item)
        return out

    def corpus_rates(self, sub):
        """The corpus's rate of each word of the union in the years of rows sub, each year as often as it has rows."""
        yi = self.yr[sub] - self.sh.y0
        weights = np.bincount(yi, minlength=self.sh.rates.shape[0]).astype(np.float64)
        return (weights @ self.sh.rates) / max(len(sub), 1)

    # ---- representative works ----
    def typicality(self, mine, other):
        """How much more each sentence of rows mine the words of mine (without its own work) predict than those of
        rows other: its gain, for the sentences of mine (zero elsewhere)."""
        sh = self.sh
        gains = np.zeros(len(self.rows))
        works = self.docs[mine]
        by_work = {int(w): mine[works == w] for w in np.unique(works)}
        for ci in range(len(self.X)):
            full = len(sh.names[ci])
            keep = sh.keep[ci]

            def counts(rows):
                c = np.zeros(full)
                c[keep] = self._counts(ci, rows)
                return c

            other_dist = ws.fit_dist(counts(other), sh.corpus[ci], sh.discounts[ci])
            c_mine = counts(mine)
            w = sh.weights[ci][keep]
            log_other = w * np.log(other_dist.p[keep])
            for rows in by_work.values():
                loo = ws.fit_dist(c_mine - counts(rows), other_dist, sh.discounts[ci])
                gains[rows] += self.X[ci][rows] @ (w * np.log(loo.p[keep]) - log_other)
        return gains

    def representative(self, mine, gains):
        sh = self.sh
        g_mine = gains[mine]
        pooled_sd = float(g_mine.std(ddof=1)) if len(mine) > 1 else 0.0
        scored = []
        for work in np.unique(self.docs[mine]):
            rows = mine[self.docs[mine] == work]
            g = gains[rows]
            sd = float(g.std(ddof=1)) if len(g) > 1 else pooled_sd
            scored.append((float(g.mean()) - sd / math.sqrt(len(g)), int(work), rows))
        scored.sort(key=lambda t: -t[0])
        out, seen = [], set()
        for _, work, rows in scored:
            author, title = sh.works.get(work, ("", ""))
            if (author or f"#{work}") in seen:
                continue
            seen.add(author or f"#{work}")
            best = rows[np.argsort(-gains[rows], kind="stable")][:_EXAMPLES]
            out.append(
                {
                    "author": author,
                    "title": title,
                    "year": int(self.yr[rows[0]]),
                    "sentences": int(len(rows)),
                    "hits": [int(sh.first_hit[self.rows[r]]) for r in best],
                }
            )
            if len(out) == _WORKS:
                break
        return out

    def describe(self, period):
        out = []
        n = int(period.max()) + 1
        for k in range(n):
            mine = np.flatnonzero(period == k)
            # The first period against the periods after it; every other against those before
            other = np.flatnonzero(period > k) if k == 0 else np.flatnonzero(period < k)
            entry = {
                "start": int(self.yr[mine].min()),
                "end": int(self.yr[mine].max()),
                "sentences": int(len(mine)),
                "works": int(len(np.unique(self.docs[mine]))),
                "authors": int(len(set(self.authors[mine]))),
            }
            if len(other):
                gains = self.typicality(mine, other)
                listed = self.words(mine, other, True, _WORDS, gains)
                if k == 0:
                    entry["characteristic"] = listed
                else:
                    entry["new"] = listed
                    entry["fading"] = self.words(mine, other, False, _FADING)
                entry["representative"] = self.representative(mine, gains)
            out.append(entry)
        return out


def _g2(a, n1, c, n2):
    """Dunning's G2 of words counted a times among n1 words and c times among n2 (arrays a, c)."""
    a, c = np.asarray(a, dtype=np.float64), np.asarray(c, dtype=np.float64)
    n = n1 + n2
    k = a + c
    observed = (a, n1 - a, c, n2 - c)
    expected = (n1 * k / n, n1 * (n - k) / n, n2 * k / n, n2 * (n - k) / n)
    return 2.0 * sum(xlogy(o, o / np.maximum(e, 1e-300)) for o, e in zip(observed, expected))
