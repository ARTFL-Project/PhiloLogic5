"""The senses a word is used in, and how their use rises and falls.

The sentences the query is in are split in two, again and again, by the words around it. Each sentence counts once
(identical sentences, such as repeated titles or reprints, once in all), as three groups of words: its words (or those
within n of the query, when collocates are counted within n words), the word just before the query and the word just
after it.

The two senses of a split differ from the sense they were split from (at the top, from the whole corpus) only in the
words that mark either: the query's collocates (words markedly more frequent around it than in the corpus) markedly
more or less frequent in one of them than in their parent, by a likelihood ratio above the threshold of the Bayesian
information criterion. Both have their own rates for those words, discounted (so that a word of the other sense keeps
a small rate), and every other word keeps its parent's rate (Law, Figueiredo and Jain 2004: features salient or not),
so splitting the sentences does not estimate again, from fewer of them, the rates of the many words that have nothing
to do with the senses. At the top, the rates are the query's own, by Katz backoff with absolute discounting (Ney,
Essen and Kneser 1994) to the corpus's. No list of words is filtered: in telling which sense a sentence is in, each
word's evidence weighs by its inverse document frequency, so that frequent words (je, vous, les, de) count for less
than rarer ones.

A split starts from the sentences with one of the query's collocates, the seed: of those whose sentences differ most
from the others in their other collocates, the most promising are tried (none of the collocation filter's words, mostly
function words: par, leur, mon tell the register of works more than a sense), and expectation-maximization moves the
sentences to the sense they fit, each with a probability of being in either (first letting every collocate tell the
senses apart, since few words of the seed's sense are marked on its side yet, then only those that mark either, chosen
at every step, and held if that choice cycles; the first pass, on a sense of many sentences, on a sample of them). A
split is kept if, fitted without each of N_FOLDS groups of works in turn, it predicts the sentences of the works left
out better than its unsplit sense, paying for saying which sense each is in, by more than EVIDENCE standard errors over
the sentences. Holding out whole works keeps the sentences of a passage, which share their words, from predicting one
another. But the gain is reckoned with the same IDF weights as the evidence, which is not a likelihood: with weights
above 1, a mixture of the two senses fits any sentence at least as well as one sense, so almost every split passes
(99.9% of the splits of sentences whose words were shuffled, in a study of October 2026), and what decides how far the
tree goes is FINEST, the importance under which splits are not looked for. A test on the plain likelihood is honest but
finds fewer senses on thin data and costs more time; it was left for later. For a query of more than _GROWN sentences,
splits are looked for and tested on _GROWN of them, spread over all, and a split once kept is fitted again on all the
sentences of its sense, which go to its two senses as they fit: the senses have all the sentences, their words and their
course over time; the tests, their sample.

Where the query has few sentences, the senses' rates for its collocates lean on the whole corpus: each sense's rates are
drawn toward the corpus profile of the words that mark it (the other words of the corpus's sentences with them, the
query's own sentences left out), by a Dirichlet prior whose strength the sense's own counts decide (empirical Bayes) and
the corpus's counts behind the profile cap. The profiles come from a table of the sentences each pair of words is in,
written at load time.

Each split has its importance: how much it improves the prediction of the words around the query, per use of the
word. The distinctions are shown from the most important down, to the detail asked for: at first, down to the elbow
of their importances (elbow), and further until the senses cover most of the sentences (default_detail); down to
FINEST, under which none are looked for.

Over time, the sentences of each year are shared out among the senses by their probabilities. The share of each sense
and its uses per million words are smoothed by a Whittaker-Henderson smoother whose smoothness generalized
cross-validation chooses, with a band from resampling the works.
"""

import hashlib
import html
import json
import math
import os
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

import numba
import numpy as np
import regex
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix, hstack
from scipy.special import gammaincinv, gammaln, ndtr

from philologic.runtime.reports.collocation import (
    _filter_hashes,
    _load_sentence_df,
    _load_token_counts,
    _nb_build_filter,
    _nb_span_with,
    collocation_search_method,
    phrase_words,
)
from philologic.runtime.get_text import _get_mmap
from philologic.runtime.word_properties import _nb_compare_keys, _nb_word_kept, property_masks
from philologic.runtime.reports.time_series import _corpus_year_counts, _get_doc_year_data

# The conventions the method rests on: none is fitted to any corpus.
N_FOLDS = 5  # folds of the sentences a split is tested on, the usual number for cross-validation
EVIDENCE = 2  # standard errors over the sentences a split's gain must exceed: the usual 95% or so
N_RESAMPLES = 200  # resamplings of the works for the bands over time
BAND = (2.5, 97.5)  # percentiles of the resamplings the band spans: 95% of them
MOST = 0.5  # at first, the senses shown cover more than this share of the sentences: most of them

# Numerical settings, not choices about the data
_TOL = 1e-6
_MAX_ITER = 200
FINEST = 0.10  # the finest detail that can be shown: distinctions less important are not looked for
_SEEDS_TRIED = 5  # seeds tried at each split, the most promising first (the search, not the test)
_SAMPLE_EVERY = 4  # a search's first pass runs on every 4th sentence of a sense of many (search_sample)
_SAMPLE_LEAST = 10000  # ... when that leaves at least this many (the first pass only finds where the second starts)
_GROWN = 100000  # the most sentences splits are looked for and tested on (growth_sample): enough to find them as all do
_PARTS = 2  # the parts a split is fitted again in on all its sentences (_fit_all), at once: a request's threads
_LAMBDAS = 10.0 ** np.arange(-3.0, 9.01, 0.25)  # smoothness values generalized cross-validation chooses among
_MAX_WORDS = 40  # words sent for each sense and group of words (the response would be large otherwise)
_MAX_EXAMPLES = 50  # example sentences sent for each sense
_WORD_EXAMPLES = 10  # example sentences sent for each word of a sense: those of the sense with the word
_TEXT_WINDOW = 400  # bytes of a text read on either side of a hit, to reach the word next to it
_CACHE_TAG = b"senses36\0"  # bump when the result changes


class Dist:
    """The probabilities a sense gives the words of a group: p over the words of the sentences, rest for all others;
    markers (a mask of the words, or None for all) are the words that may mark a sense below it; weights (or None),
    how much each word's evidence counts in telling which sense a sentence is in; discount, the absolute discount of
    the group's counts."""

    __slots__ = ("p", "rest", "markers", "weights", "discount")

    def __init__(self, p, rest, markers=None, weights=None, discount=0.0):
        self.p = p
        self.rest = rest
        self.markers = markers
        self.weights = weights
        self.discount = discount


class Node:
    """A sense: its sentences (the indices of those most likely in it), its probabilities for each group of words and
    the soft counts they were fitted on. Split, it has the model of its two children and the test of the split."""

    def __init__(self, units, dists, counts, parent_dists):
        self.units = units
        self.dists = dists
        self.counts = counts
        self.parent_dists = parent_dists
        self.children = []
        self.k = None  # the component of the parent's model this sense is
        self.model = None  # (pi, dists[channel][component]) once split
        self.test = None
        self.reason = None
        self.id = None  # its place in the tree, each sense before those it splits into
        self.parent = None
        self.prior = None  # the strength of the prior of its split's senses, and the count it weighs against


# ----------------------------- The sense model ------------------------------


def discount(counts):
    """The absolute discount of a group of words, from how many of them are in exactly one and exactly two sentences:
    n1 / (n1 + 2 n2) (Ney, Essen and Kneser 1994). With no word in exactly one, as if one were: a discount of 0 would
    leave nothing for the words a sense has not seen, and make the sentences of unseen works impossible."""
    counts = np.rint(counts)
    n1 = max(1, int(np.count_nonzero(counts == 1)))
    n2 = int(np.count_nonzero(counts == 2))
    return n1 / (n1 + 2 * n2)


def fit_dist(c, parent, d):
    """A sense's probabilities for a group of words from its (soft) counts c: each word's count less the discount d,
    and the discounted mass shared out among the words the sense has not seen, as the parent's probabilities are (Katz
    backoff with absolute discounting). Interpolating every word with the parent instead would compound the parent's
    own discount for words new to both, and predict the words of unseen works far too rarely."""
    total = float(c.sum())
    if total <= 0:
        return parent
    kept = np.maximum(c - d, 0.0)
    seen = kept > 0
    backoff = 1.0 - float(kept.sum()) / total
    unseen = parent.rest + float(parent.p[~seen].sum())  # the parent's probability of the words the sense has not seen
    scale = backoff / unseen if unseen > 0 else 0.0
    return Dist(np.where(seen, kept / total, parent.p * scale), parent.rest * scale)


def markedly_more_frequent(c, parent_p, or_less=False):
    """The words more frequent in a sense than expected from its parent by a likelihood ratio G2 above the log of the
    sense's total count (the threshold of the Bayesian information criterion: the better fit is worth one more
    number), from its (soft) counts c; or_less, more or less frequent. (local indices, G2), most significant first."""
    return _markedly(c, parent_p, or_less, None)


def _markedly(c, parent_p, or_less, markers):
    total = float(c.sum())
    if total <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0)
    use = markers is not None
    over, g2 = _g2_over(c, parent_p, total, float(np.log(total)), or_less, markers if use else _NO_MASK, use)
    order = np.argsort(-g2, kind="stable")
    return over[order], g2[order]


_NO_MASK = np.zeros(1, dtype=np.bool_)


@numba.njit(cache=True, nogil=True)
def _xlogy(x, y):
    return 0.0 if x == 0 else x * math.log(y)


@numba.njit(cache=True, nogil=True)
def _g2_over(c, parent_p, total, threshold, or_less, markers, use_markers):
    """The words (among markers, if use_markers) more (or_less: more or less) frequent in counts c of the given total
    than parent_p expects, by G2 above threshold, in the order of their index, and their G2."""
    n = len(c)
    out = np.empty(n, dtype=np.int64)
    g2 = np.empty(n, dtype=np.float64)
    m = 0
    for i in range(n):
        if use_markers and not markers[i]:
            continue
        co = c[i]
        eo = total * parent_p[i]
        if not (co != eo if or_less else co > eo):
            continue
        g = 2.0 * (_xlogy(co, co / eo) + _xlogy(total - co, (total - co) / (total - eo)))
        if g > threshold:
            out[m] = i
            g2[m] = g
            m += 1
    return out[:m], g2[:m]


def describe(c, dist, parent_p, corpus_p):
    """The words markedly more frequent in a sense than in its parent (local indices, most significant first), with
    their count in the sense, the ratio of their probability in it to their parent's and to the corpus's, and G2."""
    listed, g2 = _described(c, dist, parent_p)
    return [
        (int(i), float(c[i]), float(dist.p[i] / parent_p[i]), float(dist.p[i] / corpus_p[i]), float(g))
        for i, g in zip(listed, g2)
    ]


def _described(c, dist, parent_p):
    """The words of describe, in its order, and their G2."""
    listed, g2 = markedly_more_frequent(c, parent_p)
    if dist.markers is not None:
        keep = dist.markers[listed]
        listed, g2 = listed[keep], g2[keep]
    if dist.weights is not None:  # the strongest evidence first, as the words weigh in telling senses apart
        order = np.argsort(-g2 * dist.weights[listed], kind="stable")
        listed, g2 = listed[order], g2[order]
    return listed, g2


def describe_rarer(c, dist, parent_p):
    """The words markedly less frequent in a sense than in its parent, among those that may mark a sense: what a sense
    that is the rest of its parent lacks. (local index, count in the sense, ratio of its probability in it to its
    parent's, weighted G2), the strongest evidence first."""
    listed, g2 = _described_rarer(c, dist, parent_p)
    return [(int(i), float(c[i]), float(dist.p[i] / parent_p[i]), float(g)) for i, g in zip(listed, g2)]


def _described_rarer(c, dist, parent_p):
    """The words of describe_rarer, in its order, and their weighted G2."""
    listed, g2 = markedly_more_frequent(c, parent_p, or_less=True)
    less = c[listed] < float(c.sum()) * parent_p[listed]
    listed, g2 = listed[less], g2[less]
    if dist.markers is not None:
        keep = dist.markers[listed]
        listed, g2 = listed[keep], g2[keep]
    if dist.weights is not None:
        g2 = g2 * dist.weights[listed]
    order = np.argsort(-g2, kind="stable")
    return listed[order], g2[order]


def _logsumexp2(ll):
    """log(exp(ll[:, 0]) + exp(ll[:, 1])), computed as scipy.special.logsumexp does (the larger term apart, for
    precision), without its generality: numpy's exponentials and logarithms, the rest compiled."""
    s = np.empty(ll.shape[0])
    hi = np.empty(ll.shape[0])
    _lse_apart(ll, s, hi)
    with np.errstate(invalid="ignore", over="ignore"):
        np.exp(s, out=s)
    np.log1p(s, out=s)
    if _lse_join(ll, s, hi, 0.0 + math.log(2.0)):  # infinite or undefined: as exp and log make it
        bad = ~np.isfinite(s)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            s[bad] = np.log(np.exp(ll[bad, 0]) + np.exp(ll[bad, 1]))
    return s


@numba.njit(cache=True, nogil=True)
def _lse_apart(ll, s, hi):
    """For each row of ll, the larger term (hi) and the smaller less it (s), undefined if either is (as numpy's maximum
    and minimum make them)."""
    for i in range(ll.shape[0]):
        a = ll[i, 0]
        b = ll[i, 1]
        larger = a >= b
        h = a if larger else b
        lo = b if larger else a
        if a != a or b != b:
            h = a + b
            lo = h
        hi[i] = h
        s[i] = lo - h


@numba.njit(cache=True, nogil=True)
def _lse_join(ll, s, hi, log2):
    """s = log1p(exp(smaller - larger)) + larger, log 2 + larger where the terms tie; whether any is not finite."""
    bad = False
    for i in range(ll.shape[0]):
        if ll[i, 0] == ll[i, 1]:
            s[i] = log2 + hi[i]
        else:
            s[i] = s[i] + hi[i]
        bad = bad or not np.isfinite(s[i])
    return bad


def _evidence(d):
    """What each word of a group tells of a sense: its log-probability, weighted if the words are."""
    return np.log(d.p) if d.weights is None else d.weights * np.log(d.p)


def _loglik(Xs, dists):
    """The (weighted) log-probability of each sentence's words under one sense."""
    return sum(X @ _evidence(d) for X, d in zip(Xs, dists))


def _mixture_loglik(Xs, pi, dists):
    """The log-probability of each sentence's words under each sense of a split (with its share), n x 2."""
    ll = np.tile(np.log(pi), (Xs[0].shape[0], 1))
    for X, ds in zip(Xs, dists):
        ll += X @ np.column_stack([_evidence(ds[0]), _evidence(ds[1])])
    return ll


def marked(c, parent):
    """The words that mark a sense from its (soft) counts c: the parent's markers markedly more or less frequent in it
    than in the parent."""
    return _markedly(c, parent.p, True, parent.markers)[0]


def fit_sparse(c, parent, own=None):
    """A sense's probabilities for a group of words from its (soft) counts c, as a departure from its parent's: its own
    rate for the words that mark it (own; by default those markedly more or less frequent in it), the parent's rate
    for every other word, scaled alike to make room. A sense so differs from its parent only where it is marked:
    splitting the sentences does not estimate again, from fewer of them, the rates of the many words that have nothing
    to do with the sense, whose noise would otherwise outweigh the sense in thin data. Its own rates are discounted as
    the counts of the group are, the discount of all its words shared among them as the parent's rates are (absolute
    discounting, interpolated): a word that marks the other sense of a split, and is (nearly) absent from this one,
    keeps a small rate, not its parent's."""
    return _fit_sparse(c, parent, own)[0]


def _fit_sparse(c, parent, own=None):
    """fit_sparse, and the scale of the parent's rates it gives the words that are not its own (1 for the parent)."""
    total = float(c.sum())
    if total <= 0:
        return parent, 1.0
    if own is None:
        own = marked(c, parent)
    if len(own) == 0:
        return parent, 1.0
    kept = np.maximum(c[own] - parent.discount, 0.0) / total
    freed = float(np.minimum(c, parent.discount).sum()) / total  # the discount of every word the sense has
    p_own = kept + freed * parent.p[own]
    others = 1.0 - float(parent.p[own].sum())  # the parent's probability of every other word, the rest included
    scale = (1.0 - float(p_own.sum())) / others if others > 0 else 0.0
    p = parent.p * scale
    p[own] = p_own
    return Dist(p, parent.rest * scale, parent.markers, parent.weights, parent.discount), scale


def _split_words(c, parent, every_marker):
    """The words both senses of a split have their own rates for, from their counts c (words x 2): those that mark
    either, or every one of the parent's markers they have."""
    if every_marker:
        present = np.flatnonzero(c.sum(axis=1) > 0)
        return present if parent.markers is None else present[parent.markers[present]]
    return np.union1d(marked(c[:, 0], parent), marked(c[:, 1], parent))


# ----------------------------- The corpus's profiles of the collocates ------------------------------
# On few sentences, a sense's rates for the query's collocates come from few counts: a word that goes with the sense
# but that its sentences happen not to have gets no rate of its own there, and a sentence with it is not told to be in
# the sense. The rest of the corpus can tell: elsewhere, the words that mark a sense are in sentences with the words
# they go with. So the rates of each sense for the collocates that may seed a split (of the sentence's words) have a
# prior: a Dirichlet distribution centred on the corpus profile of the words that mark the sense (the distribution of
# those collocates in their sentences outside the query's), as strong as empirical Bayes finds the corpus worth at the
# sense split (the strength under which its sentences' counts are most likely), but never surer than the corpus's
# counts the profile is made from. Where the sentences are many, their counts outweigh it and the senses are those
# they alone make; where they are few, the corpus carries them; where the corpus is thin too (a small database, or
# words of the query's own texts, such as an author's spellings, with few sentences outside them), the profiles are
# made from few counts, and the prior is worth no more than those.


class _Profiles:
    """The corpus profiles of some of the sentence's words (columns of the first group's matrix): for each that the
    corpus has sentences of outside the query's, the distribution of the others in those sentences, each once a
    sentence (not itself: the prior of a word is what the sentences of the others say of it, not its own count again),
    by Katz backoff with absolute discounting to their rates in the corpus (as fit_dist
    fits a sense), so that a word never in a sentence with another keeps a small rate there. Q, words x words, rows
    summing to 1; slot, the place of each column among them (-1 for none, or one without sentences outside the
    query's: the corpus says nothing of it)."""

    def __init__(self, columns, cooc, n_columns):
        columns = np.asarray(columns, dtype=np.int64)
        df = np.ascontiguousarray(np.diag(cooc)).astype(np.float64)
        self.slot = np.full(n_columns, -1, dtype=np.int64)
        has = df > 0
        self.slot[columns[has]] = np.flatnonzero(has)
        self.Q = np.empty(cooc.shape, dtype=np.float32)  # single precision: 10,000 words take 400 MB
        self.tokens = cooc.sum(axis=1, dtype=np.float64) - df  # what each profile is made from: its row's counts
        if has.any():
            base = df / df.sum()
            workers = max(1, int(numba.config.NUMBA_NUM_THREADS))
            cuts = np.linspace(0, len(df), workers + 1).astype(np.int64)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for done in [pool.submit(_profile_rows, cooc, base, lo, hi, self.Q) for lo, hi in zip(cuts, cuts[1:])]:
                    done.result()


@numba.njit(cache=True, nogil=True)
def _profile_rows(cooc, base, lo, hi, Q):
    """Rows lo to hi of the profiles (_Profiles): each word's counts with the others (cooc; not with itself), less the
    discount of its counts (discount), the discounted mass shared out among the words it is never with, itself
    included, as base has them (fit_dist)."""
    n = cooc.shape[0]
    for a in range(lo, hi):
        total = 0.0
        n1 = 0
        n2 = 0
        for j in range(n):
            c = cooc[a, j] if j != a else 0
            total += c
            if c == 1:
                n1 += 1
            elif c == 2:
                n2 += 1
        if total <= 0:
            for j in range(n):
                Q[a, j] = base[j]
            continue
        n1 = max(1, n1)
        d = n1 / (n1 + 2 * n2)
        kept = 0.0
        unseen = 0.0
        for j in range(n):
            k = (cooc[a, j] if j != a else 0) - d
            if k > 0:
                kept += k
            else:
                unseen += base[j]
        scale = (1.0 - kept / total) / unseen if unseen > 0 else 0.0
        for j in range(n):
            k = (cooc[a, j] if j != a else 0) - d
            Q[a, j] = k / total if k > 0 else base[j] * scale


@numba.njit(cache=True, nogil=True)
def _anchors(c, parent_p, total, threshold, w):
    """The weight of each word in a sense's corpus profile (w): its count c (of a total) if it marks the sense (markedly
    more frequent in it than in its parent, of rates parent_p: G2 above threshold), else 0; when none marks it, its
    count. Returns whether any marks it."""
    n = len(c)
    sw = 0.0
    for i in range(n):
        w[i] = 0.0
        co = c[i]
        eo = total * parent_p[i]
        if co <= eo:
            continue
        if eo > 0 and total > eo:
            g = 2.0 * (_xlogy(co, co / eo) + _xlogy(total - co, (total - co) / (total - eo)))
            if not g > threshold:
                continue
        w[i] = co
        sw += co
    if sw > 0.0:
        return True
    for i in range(n):
        w[i] = c[i]
    return False


@numba.njit(cache=True, nogil=True)
def _centre(Q, slots, w, parent_p, out, tokens):
    """The mean of the profiles (Q's rows at slots) of the words of weights w, over the same words (out, summing to 1);
    the parent's rates parent_p without any weight. Returns the number of the corpus's counts the mean is as sure as:
    (sum w)^2 / sum (w^2 / tokens), tokens the counts each profile is made from (a weighted mean of estimates from
    those counts varies as one from that many would), over the profiles made from any; 0 when none is."""
    n = len(slots)
    sw = 0.0
    for i in range(n):
        sw += w[i]
    spread = 0.0
    evidenced = 0.0
    for i in range(n):
        t = tokens[slots[i]]
        if w[i] > 0.0 and t > 0:
            spread += w[i] * w[i] / t
            evidenced += w[i]
    if sw <= 0.0:
        for i in range(n):
            out[i] = parent_p[i]
    else:
        # Each profile's whole row added in (a pass the processor makes several numbers at a time), then the words'
        # places read: the same sums, in the same order, as adding up each word's place in turn
        acc = np.zeros(Q.shape[1])
        for a in range(n):
            if w[a] <= 0.0:
                continue
            wa = w[a] / sw
            row = Q[slots[a]]
            for j in range(Q.shape[1]):
                acc[j] += wa * row[j]
        for i in range(n):
            out[i] = acc[slots[i]]
    s = 0.0
    for i in range(n):
        s += out[i]
    if s > 0:
        for i in range(n):
            out[i] /= s
    return evidenced * evidenced / spread if spread > 0 else 0.0


def _profile(Q, slots, c, parent_p, total, threshold, out, tokens):
    """The corpus profile of a sense over some of the profiled words (slots, their places in Q; out, summing to 1):
    the mean of the profiles of the words that mark it (_anchors, from its counts c of a total), each as often as the
    sense has it; of all its words when none marks it; the parent's rates when it has none."""
    w = np.empty(len(c))
    _anchors(c, parent_p, total, threshold, w)
    return _centre(Q, slots, w, parent_p, out, tokens)


def prior_strength(c, q):
    """The strength of a Dirichlet prior of mean q (summing to 1) under which the counts c (soft, of the same words)
    are most likely: the maximum of their Dirichlet-multinomial marginal likelihood (empirical Bayes). On few counts
    any mean near theirs fits them as well, and this says the prior is worth about as much as they are, however thin
    the corpus it comes from: its callers hold it to what the prior's mean is made from (_centre). Looked for
    between the strength whose weight against the counts' is lost in rounding and that under which theirs is (N eps to
    N / eps, N their total, eps the machine epsilon): from nothing to everything. 0 for no counts."""
    c = np.asarray(c, dtype=np.float64)
    N = float(c.sum())
    if N <= 0:
        return 0.0
    used = c > 0
    cu, qu = c[used], np.asarray(q, dtype=np.float64)[used]
    if not np.all(qu > 0):
        return 0.0

    def minus(log_a):
        a = math.exp(log_a)
        return -(float(gammaln(a)) - float(gammaln(a + N)) + float(np.sum(gammaln(a * qu + cu) - gammaln(a * qu))))

    eps = np.finfo(np.float64).eps
    found = minimize_scalar(minus, bounds=(math.log(N * eps), math.log(N / eps)), method="bounded")
    return float(math.exp(found.x))


def _with_prior(profiles, local, dists, c0, against):
    """A sense's rates (dists) for the profiled words its sentences have (local, columns of the first group) with the
    prior of its corpus profile, as strong as its counts c0 (of every word of the first group) find it (prior_strength)
    but no stronger than the corpus's counts the profile is made from (a prior of strength a is as sure as a + 1
    counts): the posterior mean (count p + a pi q) / (count + a), pi its rate for all of those words, so that every
    other word keeps its own.
    against: the rates of the sense it is told from, for which words mark it (_profile). Returns (dists, strength,
    count)."""
    local = np.asarray(local, dtype=np.int64)
    c = np.ascontiguousarray(np.asarray(c0, dtype=np.float64)[local])
    total = float(np.sum(c0))
    if len(local) == 0 or total <= 0:
        return dists, 0.0, 0.0
    q = np.empty(len(local))
    parent_p = np.ascontiguousarray(np.asarray(against.p, dtype=np.float64)[local])
    sure = _profile(profiles.Q, profiles.slot[local], c, parent_p, total, math.log(total), q, profiles.tokens)
    a = min(prior_strength(c, q), max(0.0, sure - 1.0))  # no surer than the corpus's counts it is made from
    d = dists[0]
    p = np.array(d.p, dtype=np.float64)
    N = float(c.sum())
    pi = float(p[local].sum())
    p[local] = (N * p[local] + a * pi * q) / (N + a)
    return [Dist(p, d.rest, d.markers, d.weights, d.discount)] + list(dists[1:]), a, N


def _prior_places(profiles, Xs, parent):
    """The profiled words of the first group a packing's sentences have: their places among the markers they have (as
    _em_fit_marked's rates have them), among the words they have (as the counts), and among the profiled words (Q's)."""
    mk_had, _, mkh_off = Xs.marking(parent)[4][:3]
    m = np.arange(mkh_off[0], mkh_off[1], dtype=np.int64)
    slot = profiles.slot[Xs.words[0][mk_had[m]]]
    keep = slot >= 0
    return m[keep], np.ascontiguousarray(mk_had[m[keep]]), np.ascontiguousarray(slot[keep])


@numba.njit(cache=True, nogil=True)
def _prior_rates(p, counts, set_m, set_cmp, q, strengths, by_word):
    """Either sense's rates for the profiled words (p at set_m: markers x 2, or 2 x words when by_word) with its prior
    (centred on q, 2 x those words, of strength a: strengths, of each), from its counts of them (counts at set_cmp):
    (N p + a pi q) / (N + a), N its count of them and pi its rate for them all, so that every other word keeps its
    rate."""
    for k in range(2):
        a = strengths[k]
        N = 0.0
        pi = 0.0
        for i in range(len(set_m)):
            N += counts[set_cmp[i], k]
            pi += p[k, set_m[i]] if by_word else p[set_m[i], k]
        if N + a <= 0:
            continue
        for i in range(len(set_m)):
            if by_word:
                p[k, set_m[i]] = (N * p[k, set_m[i]] + a * pi * q[k, i]) / (N + a)
            else:
                p[set_m[i], k] = (N * p[set_m[i], k] + a * pi * q[k, i]) / (N + a)


def _load_cooccurrence(data_path, count_lemmas):
    """The co-occurrence table of the corpus's words, or lemmas, the loader wrote next to the collocation arrays
    (PostFilters.write_cooccurrence): for each, the words it is in a sentence with and in how many sentences, each pair
    in the row of one of its words only. Mapped, not read, for each request, as the corpus counts are. Returns indptr,
    partners and counts: a word's partners and their counts from indptr[word] to indptr[word + 1]."""
    kind = "lemma" if count_lemmas else "word"
    colloc_dir = os.path.join(data_path, "collocations")
    script = "python3 -m philologic.utils.write_cooccurrence"
    try:
        indptr, partners, counts = (
            np.load(os.path.join(colloc_dir, f"cooccurrence_{kind}_{part}.npy"), mmap_mode="r")
            for part in ("indptr", "partners", "counts")
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No co-occurrence table of the {kind}s in {colloc_dir}: the database was loaded before the loader wrote "
            f"it. Write it with {script} <database>"
        ) from None
    if not len(partners) == len(counts) == indptr[-1]:
        raise ValueError(f"The co-occurrence table in {colloc_dir} is not whole: write it again with {script} --force")
    return indptr, partners, counts


def corpus_cooccurrence(data_path, count_lemmas, words, skipped):
    """For the vocabulary ids words, the number of the corpus's sentences with each pair of them (the diagonal: with
    each), but those of skipped (their indices in the collocation index: the query's), each word once a sentence, as
    the query's sentences count them: words x words. The loader's table has the pairs of all the corpus's sentences
    (_load_cooccurrence), its corpus counts the words' (sentence_df); the skipped sentences' are counted on those
    alone, and taken off. Each step in as many parts as numba has threads, of about as much work, each part of its own
    rows."""
    indptr, partners, counts = _load_cooccurrence(data_path, count_lemmas)
    df, _ = _load_sentence_df(data_path, count_lemmas)
    if len(indptr) != len(df) + 1:
        raise ValueError(
            f"The co-occurrence table of {data_path} is not of its vocabulary: write it again with "
            "python3 -m philologic.utils.write_cooccurrence --force"
        )
    colloc_dir = os.path.join(data_path, "collocations")
    sent_offsets = np.load(os.path.join(colloc_dir, "sent_offsets.npy"), mmap_mode="r")
    count_ids = np.load(
        os.path.join(colloc_dir, "attr_lemma_ids.npy" if count_lemmas else "token_ids.npy"), mmap_mode="r"
    )
    words = np.asarray(words, dtype=np.int64)
    n = len(words)
    slot = np.full(len(df), -1, dtype=np.int32)
    slot[words] = np.arange(n, dtype=np.int32)
    cooc = np.zeros((n, n), dtype=np.int32)
    parts = max(1, int(numba.config.NUMBA_NUM_THREADS))

    def cut(work):  # from, to: parts of about as much work
        total = np.concatenate([[0], np.cumsum(work, dtype=np.float64)])
        cuts = np.unique(np.searchsorted(total, np.linspace(0, total[-1], parts + 1)[1:-1]))
        return list(zip(np.concatenate([[0], cuts]), np.concatenate([cuts, [len(work)]])))

    skipped = np.asarray(skipped, dtype=np.int64)
    with ThreadPoolExecutor(max_workers=parts) as pool:

        def each(kernel, cuts, *args):
            return [done.result() for done in [pool.submit(kernel, *args, lo, hi) for lo, hi in cuts if hi > lo]]

        each(_table_pairs, cut(indptr[words + 1] - indptr[words]), indptr, partners, counts, words, slot, cooc)
        triangle = cut(n - np.arange(n))  # the rows' parts of as much of the upper triangle
        each(_symmetric, triangle, cooc)  # each pair of the table is in one of its rows
        # The skipped sentences' words of these, each once, ascending; for each word the skipped sentences with it
        tokens = (sent_offsets[skipped + 1] - sent_offsets[skipped]).astype(np.int64)
        offsets, places = [np.zeros(1, dtype=np.int64)], [np.zeros(0, dtype=np.int32)]
        for part_offsets, part_places in each(_own_words, cut(tokens), sent_offsets, count_ids, slot, skipped, n):
            offsets.append(part_offsets[1:] + sum(len(p) for p in places))
            places.append(part_places)
        offsets, places = np.concatenate(offsets), np.concatenate(places)
        starts, after, ends, work = _sentences_with(offsets, places, n)
        each(_less_own, cut(work), cooc, places, starts, after, ends)  # above the diagonal
        cooc[np.arange(n), np.arange(n)] = df[words] - np.diff(starts)
        each(_mirror, triangle, cooc)
    return cooc


@numba.njit(cache=True, nogil=True)
def _own_words(sent_offsets, count_ids, slot, skipped, n, lo, hi):
    """The words of slot (their places, -1 for none) in each of the sentences skipped[lo:hi], each once, ascending:
    (offsets, places) of each sentence's."""
    bound = 0
    for q in range(lo, hi):
        bound += np.int64(sent_offsets[skipped[q] + 1] - sent_offsets[skipped[q]])
    places = np.empty(bound, dtype=np.int32)
    offsets = np.zeros(hi - lo + 1, dtype=np.int64)
    stamp = np.full(n, -1, dtype=np.int64)
    e = 0
    for q in range(lo, hi):
        for k in range(sent_offsets[skipped[q]], sent_offsets[skipped[q] + 1]):
            w = slot[count_ids[k]]
            if w >= 0 and stamp[w] != q:
                stamp[w] = q
                places[e] = w
                e += 1
        places[offsets[q - lo] : e].sort()
        offsets[q - lo + 1] = e
    return offsets, places[:e]


@numba.njit(cache=True, nogil=True)
def _sentences_with(offsets, places, n):
    """For each place (word) of _own_words' sentences, from starts[w] to starts[w + 1], the sentences with it, by
    where the words after it in each start and end in places (after, ends); and their number, summed (the work of its
    row in _less_own)."""
    starts = np.zeros(n + 1, dtype=np.int64)
    for e in range(len(places)):
        starts[places[e] + 1] += 1
    for w in range(n):
        starts[w + 1] += starts[w]
    cursor = starts[:-1].copy()
    after = np.empty(len(places), dtype=np.int64)
    ends = np.empty(len(places), dtype=np.int64)
    work = np.zeros(n, dtype=np.int64)
    for q in range(len(offsets) - 1):
        for e in range(offsets[q], offsets[q + 1]):
            w = places[e]
            after[cursor[w]] = e + 1
            ends[cursor[w]] = offsets[q + 1]
            cursor[w] += 1
            work[w] += offsets[q + 1] - e
    return starts, after, ends, work


@numba.njit(cache=True, nogil=True)
def _table_pairs(indptr, partners, counts, words, slot, cooc, lo, hi):
    """The table's counts of the pairs of words in the rows of words lo..hi (cooc's rows; each pair is in one of its
    two)."""
    for i in range(lo, hi):
        row = cooc[i]
        for e in range(indptr[words[i]], indptr[words[i] + 1]):
            j = slot[partners[e]]
            if j >= 0:
                row[j] = counts[e]


_BLOCK = 64  # rows and columns of cooc added to their mirror at once: a square of each in a core's first cache


@numba.njit(cache=True, nogil=True)
def _symmetric(cooc, lo, hi):
    """Each count of rows lo..hi above the diagonal and its mirror below set to their sum (one of them is 0), in squares
    of _BLOCK."""
    n = cooc.shape[0]
    for i0 in range(lo, hi, _BLOCK):
        i1 = min(hi, i0 + _BLOCK)
        for j0 in range(i0, n, _BLOCK):
            j1 = min(n, j0 + _BLOCK)
            for i in range(i0, i1):
                for j in range(max(j0, i + 1), j1):
                    v = cooc[i, j] + cooc[j, i]
                    cooc[i, j] = v
                    cooc[j, i] = v


@numba.njit(cache=True, nogil=True)
def _less_own(cooc, places, starts, after, ends, lo, hi):
    """Rows lo..hi, above the diagonal, less the pairs of the sentences with their word (_sentences_with): the words
    after it in each (whose places are in order), each row at once."""
    for i in range(lo, hi):
        row = cooc[i]
        for x in range(starts[i], starts[i + 1]):
            for e in range(after[x], ends[x]):
                row[places[e]] -= 1


@numba.njit(cache=True, nogil=True)
def _mirror(cooc, lo, hi):
    """Rows lo..hi above the diagonal copied onto their mirror below, in squares of _BLOCK."""
    n = cooc.shape[0]
    for i0 in range(lo, hi, _BLOCK):
        i1 = min(hi, i0 + _BLOCK)
        for j0 in range(i0, n, _BLOCK):
            j1 = min(n, j0 + _BLOCK)
            for i in range(i0, i1):
                for j in range(max(j0, i + 1), j1):
                    cooc[j, i] = cooc[i, j]


# The EM of a split, compiled: the words of all groups in one vocabulary (each group's words after the last group's),
# the sentences' matrices in CSR. It computes what fit_sparse and _split_words do, in the same order (numpy's pairwise
# sums included), so that the senses' rates are the same to the last bit for the same counts; logarithms and
# exponentials are left to numpy, whose own implementations differ from the C library's in the last bit. Its
# iterations only follow the words the sentences have: those of a sense's sentences are a fraction of the query's. A
# word that marks no sense has its parent's rate, scaled, under both senses of every split (fit_sparse): what such
# words tell of a sentence is then the sum of their parent's log-rates, the same at every step, and their weighted
# count times the log of the scale. So each step reads only the entries of the markers, from slices of their own, the
# others once a fit: the sentences' log-probabilities are _mixture_loglik's, added up in another order (to rounding).


class _Mats(list):
    """The matrices of the groups of words of the same sentences, packed once for the compiled EM: the words the
    sentences have, in one vocabulary (cols, their index in the vocabulary of all words; pos, among their group's
    words; each group's from cmp_off), indices into it, and whether each group's counts are all ones (whose counts are
    then not read: one times a number is that number). The sentences may be in blocks (blocks, the block of each: the
    folds a split is tested on), packed a block after the other (perm, the sentences in that order, inv its inverse;
    bounds, each block's first and n), so that a fit leaves a block out by not reading its entries, and the fits of
    every fold share one packing. parent, the sense split, whose markers are packed apart (marking); marked, when
    given, the markers' entries of all the query's sentences and the indices of these among them (_Marked), which the
    markers' entries are taken from; words, when given, the words of each group the sentences have, or more (those of
    all the parts of a sense: see _Parts). by_words: whether the words' counts in either sense are made from slices of
    each word's entries (columns), else from the sentences' (entries): the slices take a pass of their own to make,
    worth it for the many fits of a sense's search and tests, not for the few of a refit on all its sentences."""

    def __init__(self, Xs, blocks=None, parent=None, marked=None, words=None, by_words=True):
        super().__init__(csr_matrix(X) for X in Xs)
        n = self[0].shape[0]
        self.marked = marked
        self.blocks = None if blocks is None else np.asarray(blocks)
        if blocks is None:
            self.perm = self.inv = None
            self.bounds = np.array([0, n], dtype=np.int64)
        else:
            self.perm = np.argsort(self.blocks, kind="stable")
            self.inv = np.argsort(self.perm)
            self.bounds = np.concatenate([[0], np.cumsum(np.bincount(self.blocks))]).astype(np.int64)
        packed = self._packed()
        n_groups = len(self)
        voc_off = np.zeros(n_groups + 1, dtype=np.int64)
        cmp_off = np.zeros(n_groups + 1, dtype=np.int64)
        had = []
        for c, X in enumerate(packed):
            had.append(
                np.flatnonzero(np.bincount(X.indices[: X.nnz], minlength=X.shape[1])) if words is None else words[c]
            )
            voc_off[c + 1] = voc_off[c] + X.shape[1]
            cmp_off[c + 1] = cmp_off[c] + len(had[c])
        pos = np.concatenate(had).astype(np.int64)
        cols = pos + np.repeat(voc_off[:-1], np.diff(cmp_off))
        self.ones = np.array([bool(np.all(X.data[: X.nnz] == 1)) for X in packed], dtype=np.bool_)
        self.words = (pos, cols, cmp_off, voc_off)
        self.first = _block_slices(self.bounds)
        indptr, indices, data = self._entries(packed)
        # The slices of the sentences' entries are only read for the held-out sentences' log-probability under the
        # unsplit sense (block_loglik): made for a packing by fold, else when asked for
        self._rows = None if blocks is None else _row_slices(indptr, indices, data, self.ones, len(cols), self.bounds)
        self._rows_lock = threading.Lock()
        self._word_counts = None
        self.columns = _column_slices(indptr, indices, data, self.ones, cmp_off, self.bounds) if by_words else None
        self.entries = None if by_words else (indptr, indices, data)
        self._marking = []
        self._unmarked_loglik = []
        if parent is not None:
            self.marking(parent, packed)

    def _packed(self):
        return list(self) if self.perm is None else [X[self.perm] for X in self]

    def _entries(self, packed):
        """The entries of the groups' matrices (packed, in the order of perm) in one: each group's indptr (groups x n +
        1, into the entries of all), each entry's word, by its index among the words the sentences have, and the counts
        (read only for the groups not all ones)."""
        pos, cols, cmp_off, voc_off = self.words
        nnz_off = np.concatenate([[0], np.cumsum([X.nnz for X in packed])]).astype(np.int64)
        indptr = np.empty((len(packed), self[0].shape[0] + 1), dtype=np.int64)
        indices = np.empty(nnz_off[-1], dtype=np.int32)
        data = np.empty(nnz_off[-1], dtype=np.float64)
        for c, X in enumerate(packed):
            to_cmp = np.zeros(X.shape[1], dtype=np.int32)
            to_cmp[pos[cmp_off[c] : cmp_off[c + 1]]] = np.arange(cmp_off[c], cmp_off[c + 1], dtype=np.int32)
            indptr[c] = X.indptr + nnz_off[c]
            indices[nnz_off[c] : nnz_off[c + 1]] = to_cmp[X.indices[: X.nnz]]
            if not self.ones[c]:
                data[nnz_off[c] : nnz_off[c + 1]] = X.data[: X.nnz]
        return indptr, indices, data

    @property
    def rows(self):
        """The slices of the sentences' entries (_row_slices), made once."""
        with self._rows_lock:
            if self._rows is None:
                indptr, indices, data = self._entries(self._packed())
                self._rows = _row_slices(indptr, indices, data, self.ones, len(self.words[1]), self.bounds)
        return self._rows

    def word_counts(self, blocks):
        """Each word's count in the sentences of the blocks given (of the words they have, and 0 for the slices'
        padding), made once for each block."""
        with self._rows_lock:
            if self._word_counts is None:
                n = self[0].shape[0]
                ones = np.ones(n + 1)
                ones[n] = 0.0
                made = np.empty((len(self.bounds) - 1, len(self.words[1]) + 1))
                for b in range(len(made)):
                    block = np.array([b], dtype=np.int64)
                    if self.columns is None:
                        _em_row_counts(*self.entries, self.ones, ones, self.bounds, block, made[b])
                    else:
                        _em_counts(*self.columns, self.ones, ones, block, made[b])
                self._word_counts = made
        return self._word_counts[blocks].sum(axis=0)

    def ranges(self, left_out=None):
        """The slices of the sentences' entries of every block but left_out (k x 2, from and to, as _em_loglik takes
        them), and those blocks."""
        first = self.first
        n_blocks = len(self.bounds) - 1
        if left_out is None:
            ranges = [(0, first[-1])]
        else:
            ranges = [r for r in ((0, first[left_out]), (first[left_out + 1], first[-1])) if r[1] > r[0]]
        blocks = np.array([b for b in range(n_blocks) if b != left_out], dtype=np.int64)
        return np.array(ranges, dtype=np.int64).reshape(-1, 2), blocks

    def block_loglik(self, dists, block):
        """The (weighted) log-probability of each sentence of a block under one sense: sum(X @ _evidence(d)) over the
        groups, to rounding, compiled."""
        pos, cols, cmp_off, voc_off = self.words
        evidence = np.zeros((len(cols) + 1, 2))
        evidence[: len(cols)] = np.concatenate([_evidence(d) for d in dists])[cols, None]
        order, start, idx, sdata, first = self.rows
        ranges = np.array([[first[block], first[block + 1]]], dtype=np.int64)
        ll = _em_loglik(
            order, start, idx, sdata, self.ones, evidence.reshape(-1), np.zeros(2), self[0].shape[0], ranges
        )
        return ll[self.bounds[block] : self.bounds[block + 1], 0].copy()

    def marking(self, parent, packed=None):
        """The markers of the groups of words (all of them; their indices; each group's first; their places among the
        words the sentences have, -1 for none), and those the sentences have: their places among the words they have,
        the place of each of those among them (-1 for none), each group's first, their weights (and the padding's),
        the slices of their entries, and each sentence's weighted count of its other words in each group (groups x n).
        The same for every sense of a tree, its root's, so made once."""
        key = [(d.markers, d.weights) for d in parent]
        for known, made in self._marking:
            if all(a[0] is b[0] and a[1] is b[1] for a, b in zip(known, key)):
                return made
        pos, cols, cmp_off, voc_off = self.words
        markers = np.concatenate([np.ones(len(d.p), np.bool_) if d.markers is None else d.markers for d in parent])
        weights = np.concatenate([np.ones(len(d.p)) if d.weights is None else d.weights for d in parent])
        mk = np.flatnonzero(markers)
        had = np.full(int(voc_off[-1]), -1, dtype=np.int64)
        had[cols] = np.arange(len(cols))
        mk_cmp = had[mk]
        # The markers the sentences have (their places among the words they have), and their entries apart
        mk_had = mk_cmp[mk_cmp >= 0]
        mkh_off = np.searchsorted(mk_had, cmp_off)
        w_had = weights[cols]
        n = self[0].shape[0]
        if self.marked is not None and self.marked[0].made_for(key):
            # Those of all the query's sentences, taken for these
            whole, of = self.marked
            to_had = np.full(len(mk), -1, dtype=np.int32)
            to_had[mk_cmp >= 0] = np.arange(len(mk_had), dtype=np.int32)
            of = np.asarray(of if self.perm is None else of[self.perm], dtype=np.int64)
            m_indptr, m_indices, m_data, unmarked = _marker_rows(*whole.entries, of, to_had, self.ones)
        else:
            m_indptr = np.empty((len(self), n + 1), dtype=np.int64)
            parts, unmarked = [], np.empty((len(self), n))
            total = 0
            for c, X in enumerate(self._packed() if packed is None else packed):
                to_mk = np.full(X.shape[1], -1, dtype=np.int64)
                to_mk[pos[mk_had[mkh_off[c] : mkh_off[c + 1]]]] = np.arange(mkh_off[c], mkh_off[c + 1])
                w = np.zeros(X.shape[1])
                w[pos[cmp_off[c] : cmp_off[c + 1]]] = w_had[cmp_off[c] : cmp_off[c + 1]]
                ptr, idx, data, unmarked[c] = _marker_entries(X.indptr, X.indices, X.data, bool(self.ones[c]), to_mk, w)
                m_indptr[c] = ptr + total
                total += len(idx)
                parts.append((idx, data))
            m_indices = np.concatenate([idx for idx, _ in parts]) if parts else np.empty(0, np.int32)
            m_data = np.concatenate([data for _, data in parts]) if parts else np.empty(0)
        rows = _row_slices(m_indptr, m_indices, m_data, self.ones, len(mk_had), self.bounds)
        weights_mk = np.append(w_had[mk_had], 1.0)  # and the padding's
        mk_pos = np.full(len(cols), -1, dtype=np.int64)
        mk_pos[mk_had] = np.arange(len(mk_had))
        made = (
            markers,
            mk,
            np.searchsorted(mk, voc_off),
            mk_cmp,
            (mk_had, mk_pos, mkh_off, weights_mk, rows, unmarked),
        )
        self._marking.append((key, made))
        return made

    def unmarked_loglik(self, parent):
        """What the words that mark no sense tell of each sentence under the parent: the sum of their weighted
        log-rates, over the groups (n, in the order of perm), as the sentences' matrices of those words times the
        log-rates add them up. The same for every fit from the same parent, so made once for each."""
        for known, made in self._unmarked_loglik:
            if known is parent:
                return made
        markers = self.marking(parent)[0]
        ll = np.zeros(self[0].shape[0])
        lo = 0
        for X, d, ones in zip(self, parent, self.ones):
            evidence = _evidence(d)
            evidence[markers[lo : lo + len(d.p)]] = 0.0  # the markers' read at every step
            _add_rows_one(X.indptr, X.indices, X.data, bool(ones), evidence, ll)
            lo += len(d.p)
        made = ll if self.perm is None else ll[self.perm]
        self._unmarked_loglik.append((parent, made))
        return made


class _Marked:
    """The entries of the markers of a tree's senses (the root's: the same for every sense) in all of the query's
    sentences, apart (by their index among the markers; indptr groups x n + 1, into the entries of all; their counts,
    each group's from data_off, none for a group of counts all ones), and each sentence's weighted count of its other
    words in each group (groups x n), as _Mats.marking makes them: made once, so that each sense's packing takes those
    of its sentences instead of sorting out its entries again."""

    def __init__(self, Xall, parent):
        self.key = [(d.markers, d.weights) for d in parent]
        n = Xall[0].shape[0]
        indptr = np.empty((len(Xall), n + 1), dtype=np.int64)
        parts, unmarked = [], np.empty((len(Xall), n))
        total = lo = mk_lo = 0
        for c, (X, d) in enumerate(zip(Xall, parent)):
            mk = np.arange(len(d.p)) if d.markers is None else np.flatnonzero(d.markers)
            to_mk = np.full(X.shape[1], -1, dtype=np.int32)
            to_mk[mk] = np.arange(mk_lo, mk_lo + len(mk))
            w = np.ones(X.shape[1]) if d.weights is None else np.asarray(d.weights, dtype=np.float64)
            ones = bool(np.all(X.data[: X.nnz] == 1))
            ptr, idx, data, unmarked[c] = _marker_split(X.indptr, X.indices, X.data, ones, to_mk, w)
            indptr[c] = ptr + total
            total += len(idx)
            parts.append((idx, data))
            lo += len(d.p)
            mk_lo += len(mk)
        indices = np.concatenate([idx for idx, _ in parts]) if parts else np.empty(0, np.int32)
        data = np.concatenate([data for _, data in parts]) if parts else np.empty(0)
        data_off = np.concatenate([[0], np.cumsum([len(data) for _, data in parts])]).astype(np.int64)
        self.entries = (indptr, indices, data, data_off, unmarked)

    def made_for(self, key):
        """Whether these are the markers (and weights) of key, as _Mats.marking has them."""
        return all(a[0] is b[0] and a[1] is b[1] for a, b in zip(self.key, key))


@numba.njit(cache=True, nogil=True)
def _marker_split(indptr, indices, data, ones, to_mk, w):
    """_marker_entries in one pass, its counts left out for a group of counts all ones (ones: they are not read)."""
    n = len(indptr) - 1
    one = np.uint64(1)
    m_indptr = np.zeros(n + 1, dtype=np.int64)
    m_indices = np.empty(len(indices), dtype=np.int32)
    m_data = np.empty(0 if ones else len(indices))
    unmarked = np.zeros(n)
    m = np.uint64(0)
    for i in range(n):
        s = 0.0
        jj = np.uint64(indptr[i])
        end = np.uint64(indptr[i + 1])
        while jj < end:
            j = np.uint64(indices[jj])
            t = to_mk[j]
            if t >= 0:
                m_indices[m] = t
                if not ones:
                    m_data[m] = data[jj]
                m += one
            else:
                s += (1.0 if ones else data[jj]) * w[j]
            jj += one
        unmarked[i] = s
        m_indptr[i + 1] = m
    return m_indptr, m_indices[:m], m_data[: 0 if ones else m], unmarked


@numba.njit(cache=True, nogil=True)
def _marker_rows(indptr, indices, data, data_off, unmarked_all, rows, to_had, ones):
    """The markers' entries of the sentences rows of all the query's (from _Marked: indptr ... unmarked_all), each by
    the place to_had gives its marker, and their weighted counts of their other words: as _marker_entries makes them
    from those sentences' matrices (indptr groups x n + 1, into the entries of all, indices, counts; groups x n), but
    for the counts of the groups whose counts are all ones (ones), which are not read."""
    n_groups = indptr.shape[0]
    n = len(rows)
    one = np.uint64(1)
    m_indptr = np.empty((n_groups, n + 1), dtype=np.int64)
    total = 0
    for c in range(n_groups):
        for q in range(n):
            m_indptr[c, q] = total
            r = np.uint64(rows[q])
            total += indptr[c, r + one] - indptr[c, r]
        m_indptr[c, n] = total
    m_indices = np.empty(total, dtype=np.int32)
    m_data = np.empty(total)
    unmarked = np.empty((n_groups, n))
    e = np.uint64(0)
    for c in range(n_groups):
        for q in range(n):
            r = np.uint64(rows[q])
            unmarked[c, q] = unmarked_all[c, r]
            jj = np.uint64(indptr[c, r])
            end = np.uint64(indptr[c, r + one])
            if ones[c]:
                while jj < end:
                    m_indices[e] = to_had[np.uint64(indices[jj])]
                    e += one
                    jj += one
            else:
                before = np.uint64(indptr[c, 0] - data_off[c])  # the entries of the groups before without counts
                while jj < end:
                    m_indices[e] = to_had[np.uint64(indices[jj])]
                    m_data[e] = data[jj - before]
                    e += one
                    jj += one
    return m_indptr, m_indices, m_data, unmarked


class _Parts(list):
    """The sentences of a sense in parts, each packed apart (_Mats) with the words of all, for an EM that makes each
    step's counts and log-probabilities of every part at once on the pool's threads, the counts then added up (each
    part's in the order of its sentences, as the blocks of one packing would be: see _em)."""


def _now(fn, *args):
    """fn(*args) run at once, as a future: the pool of a single thread."""
    future = Future()
    future.set_result(fn(*args))
    return future


class _Task:
    """fn(item), run by the first thread to get to it."""

    def __init__(self, fn, item):
        self.fn, self.item = fn, item
        self.lock = threading.Lock()
        self.started = False
        self.done = threading.Event()
        self.result = self.error = None

    def run(self):
        with self.lock:
            if self.started:
                return
            self.started = True
        try:
            self.result = self.fn(self.item)
        except BaseException as error:  # handed to the thread that waits for it
            self.error = error
        finally:
            self.done.set()

    def get(self):
        self.done.wait()
        if self.error is not None:
            raise self.error
        return self.result


def _at_once(submit, fn, items):
    """fn of each of items, at once: all but the first handed to the pool's threads (submit), the first done here,
    then any of the others no thread has started yet, so that a thread of the pool that runs this never waits for work
    queued behind its own. The results, in order."""
    if len(items) == 1:
        return [fn(items[0])]
    tasks = [_Task(fn, item) for item in items]
    for task in tasks[1:]:
        submit(task.run)
    for task in tasks:
        task.run()
    return [task.get() for task in tasks]


def _block_slices(bounds):
    """The first slice of each block of sentences (from bounds, the first of each and n), and the number of slices:
    _WIDTH sentences a slice, each block's apart (as _row_slices lays them out)."""
    slices = (np.diff(bounds) + _WIDTH - 1) // _WIDTH
    return np.concatenate([[0], np.cumsum(slices)]).astype(np.int64)


@numba.njit(cache=True, nogil=True)
def _add_rows_one(indptr, indices, data, ones, evidence, ll):
    """ll += X @ evidence (n), X in CSR: each row's sum of its entries in order, from 0 (with ones, the counts are not
    read: one times a number is that number)."""
    for i in range(len(indptr) - 1):
        s = 0.0
        if ones:
            for j in range(indptr[i], indptr[i + 1]):
                s += evidence[np.uint64(indices[j])]
        else:
            for j in range(indptr[i], indptr[i + 1]):
                s += data[j] * evidence[np.uint64(indices[j])]
        ll[i] += s


# The sums of the EM run over slices of the entries of the sentences' matrices (sliced ELLPACK): _WIDTH sentences
# (or words) at a time, those of about the same number of entries (sorted by it within each _WINDOW sentences, which
# keeps the slices' making within the cache), each padded to the longest of its slice with an entry of no weight
# (adding zero changes no sum), so that their sums, apart, keep the processor busy while each still adds its entries
# in order. Indices are unsigned: numba checks a signed index for being negative.
_WIDTH = 4
_WINDOW = 1024


@numba.njit(cache=True, nogil=True)
def _by_length(lengths):
    """The indices by length, longest first, in order among equals (a counting sort)."""
    longest = 0
    for i in range(len(lengths)):
        longest = max(longest, lengths[i])
    at = np.zeros(longest + 2, dtype=np.int64)
    for i in range(len(lengths)):
        at[longest - lengths[i] + 1] += 1
    for m in range(longest + 1):
        at[m + 1] += at[m]
    order = np.empty(len(lengths), dtype=np.int64)
    for i in range(len(lengths)):
        q = longest - lengths[i]
        order[at[q]] = i
        at[q] += 1
    return order


@numba.njit(cache=True, nogil=True)
def _row_slices(indptr, indices, data, ones, pad, bounds):
    """The entries of each group of the sentences in slices of sentences, those of each block of sentences (from
    bounds, the first of each and n) apart: the sentences of each slice (order, groups x _WIDTH a slice, by number of
    entries within each _WINDOW of a block, the sentence after the last, n, in the places past a block's last), the
    start of each slice's entries (groups x slices + 1), the entries t of its sentences stored together (idx, the word
    pad for padding, and their counts, sdata, for the groups not all ones), and each block's first slice."""
    n_groups = indptr.shape[0]
    n = indptr.shape[1] - 1
    n_blocks = len(bounds) - 1
    block_slices = np.zeros(n_blocks + 1, dtype=np.int64)
    for b in range(n_blocks):
        block_slices[b + 1] = block_slices[b] + (bounds[b + 1] - bounds[b] + _WIDTH - 1) // _WIDTH
    n_slices = block_slices[n_blocks]
    order = np.full((n_groups, _WIDTH * n_slices), n, dtype=np.uint64)
    start = np.empty((n_groups, n_slices + 1), dtype=np.uint64)
    lengths = np.zeros((n_groups, n + 1), dtype=np.int64)
    total = 0
    for c in range(n_groups):
        for i in range(n):
            lengths[c, i] = indptr[c, i + 1] - indptr[c, i]
        for b in range(n_blocks):
            lo, hi = bounds[b], bounds[b + 1]
            first = _WIDTH * block_slices[b] - lo
            for w in range(lo, hi, _WINDOW):
                by_length = _by_length(lengths[c, w : min(w + _WINDOW, hi)])
                for q in range(len(by_length)):
                    order[c, first + w + q] = w + by_length[q]
        for sl in range(n_slices):
            start[c, sl] = total
            longest = 0
            for q in range(sl * _WIDTH, (sl + 1) * _WIDTH):
                longest = max(longest, lengths[c, order[c, q]])
            total += _WIDTH * longest
        start[c, n_slices] = total
    idx = np.full(total, pad, dtype=np.uint32)
    sdata = np.zeros(total)
    for c in range(n_groups):
        for q in range(_WIDTH * n_slices):
            i = order[c, q]
            base = start[c, q // _WIDTH] + q % _WIDTH
            for t in range(lengths[c, i]):
                idx[base + t * _WIDTH] = indices[indptr[c, i] + t]
                if not ones[c]:
                    sdata[base + t * _WIDTH] = data[indptr[c, i] + t]
    return order, start, idx, sdata, block_slices


@numba.njit(cache=True, nogil=True)
def _column_slices(indptr, indices, data, ones, cmp_off, bounds):
    """The entries of the words the sentences have in slices of words, each group's and each block of sentences' (from
    bounds, as for _row_slices) apart: the words of each slice (order, _WIDTH a slice, the word after the last for
    none; of a block, only the words it has), the start of each slice's entries, the sentences of each word's entries,
    in order, stored together (idx, the sentence after the last for padding, and the entries' counts, cdata, for the
    groups not all ones), and each block's and group's first slice (g_off, blocks x groups + 1)."""
    n_groups = indptr.shape[0]
    n = indptr.shape[1] - 1
    n_cols = cmp_off[n_groups]
    n_blocks = len(bounds) - 1
    lengths = np.zeros((n_blocks, n_cols), dtype=np.int64)
    for b in range(n_blocks):
        for c in range(n_groups):
            for jj in range(indptr[c, bounds[b]], indptr[c, bounds[b + 1]]):
                lengths[b, indices[jj]] += 1
    g_off = np.zeros((n_blocks, n_groups + 1), dtype=np.int64)
    sorted_words = np.empty((n_blocks, n_cols), dtype=np.int64)
    n_slices = 0
    for b in range(n_blocks):
        g_off[b, 0] = n_slices
        for c in range(n_groups):
            by_length = _by_length(lengths[b, cmp_off[c] : cmp_off[c + 1]])
            had = 0
            for q in range(len(by_length)):
                sorted_words[b, cmp_off[c] + q] = cmp_off[c] + by_length[q]
                if lengths[b, cmp_off[c] + by_length[q]] > 0:
                    had += 1
            n_slices += (had + _WIDTH - 1) // _WIDTH
            g_off[b, c + 1] = n_slices
    order = np.full(_WIDTH * n_slices, n_cols, dtype=np.uint64)
    slot = np.full((n_blocks, n_cols), -1, dtype=np.int64)  # each word's place among a block's slices' words
    for b in range(n_blocks):
        for c in range(n_groups):
            for q in range(_WIDTH * (g_off[b, c + 1] - g_off[b, c])):
                if q < cmp_off[c + 1] - cmp_off[c]:
                    j = sorted_words[b, cmp_off[c] + q]
                    if lengths[b, j] > 0:
                        order[_WIDTH * g_off[b, c] + q] = j
                        slot[b, j] = _WIDTH * g_off[b, c] + q
    start = np.empty(n_slices + 1, dtype=np.uint64)
    total = 0
    b = 0
    for sl in range(n_slices):
        while sl >= g_off[b, n_groups]:
            b += 1
        start[sl] = total
        first = order[_WIDTH * sl]
        total += _WIDTH * (lengths[b, first] if first < n_cols else 0)
    start[n_slices] = total
    idx = np.full(total, n, dtype=np.uint32)
    cdata = np.zeros(total)
    at = np.empty(n_cols, dtype=np.int64)  # where each word's next entry goes, in the block
    for b in range(n_blocks):
        for j in range(n_cols):
            if slot[b, j] >= 0:
                at[j] = start[slot[b, j] // _WIDTH] + slot[b, j] % _WIDTH
        for c in range(n_groups):
            for i in range(bounds[b], bounds[b + 1]):
                for jj in range(indptr[c, i], indptr[c, i + 1]):
                    j = indices[jj]
                    idx[at[j]] = i
                    if not ones[c]:
                        cdata[at[j]] = data[jj]
                    at[j] += _WIDTH
    return order, start, idx, cdata, g_off


@numba.njit(cache=True, nogil=True)
def _marker_entries(indptr, indices, data, ones, to_mk, w):
    """A group's entries of the markers (indptr, their index among the markers to_mk gives, -1 for the other words,
    and their counts), and each sentence's weighted count of its other words, as the matrix of those words times
    their weights w adds it up (in the order of its entries, from 0)."""
    n = len(indptr) - 1
    m_indptr = np.zeros(n + 1, dtype=np.int64)
    unmarked = np.zeros(n)
    m = 0
    for i in range(n):
        s = 0.0
        for jj in range(indptr[i], indptr[i + 1]):
            j = indices[jj]
            if to_mk[j] >= 0:
                m += 1
            else:
                s += (1.0 if ones else data[jj]) * w[j]
        unmarked[i] = s
        m_indptr[i + 1] = m
    m_indices = np.empty(m, dtype=np.int32)
    m_data = np.empty(m)
    m = 0
    for i in range(n):
        for jj in range(indptr[i], indptr[i + 1]):
            j = indices[jj]
            if to_mk[j] >= 0:
                m_indices[m] = to_mk[j]
                m_data[m] = data[jj]
                m += 1
    return m_indptr, m_indices, m_data, unmarked


@numba.njit(cache=True, nogil=True)
def _block_sum(a, lo, n):
    """numpy's sum of a block of a[lo:lo + n], n <= 128: 8 accumulators."""
    if n < 8:
        res = 0.0
        for i in range(lo, lo + n):
            res += a[i]
        return res
    r0, r1, r2, r3 = a[lo], a[lo + 1], a[lo + 2], a[lo + 3]
    r4, r5, r6, r7 = a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]
    i = 8
    m = n - (n % 8)
    while i < m:
        r0 += a[lo + i]
        r1 += a[lo + i + 1]
        r2 += a[lo + i + 2]
        r3 += a[lo + i + 3]
        r4 += a[lo + i + 4]
        r5 += a[lo + i + 5]
        r6 += a[lo + i + 6]
        r7 += a[lo + i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[lo + i]
        i += 1
    return res


@numba.njit(cache=True, nogil=True)
def _pairwise_sum(a, lo, n):
    """numpy's sum of a[lo:lo + n], the same to the last bit: halves summed apart (each a multiple of 8 long, but the
    last) down to blocks of 128 at most. Without recursion, which numba's cache does not take."""
    if n <= 128:
        return 0.0 + _block_sum(a, lo, n)
    st_lo = np.empty(64, dtype=np.int64)
    st_n = np.empty(64, dtype=np.int64)
    st_stage = np.zeros(64, dtype=np.int64)  # 0: halves to sum; 1: the first being summed; 2: the second
    st_first = np.empty(64)
    top = 0
    st_lo[0] = lo
    st_n[0] = n
    while True:
        if st_n[top] <= 128:
            val = _block_sum(a, st_lo[top], st_n[top])
            top -= 1
            while top >= 0:  # hand the sum up
                if st_stage[top] == 1:
                    st_first[top] = val
                    st_stage[top] = 2
                    half = st_n[top] // 2
                    half -= half % 8
                    st_lo[top + 1] = st_lo[top] + half
                    st_n[top + 1] = st_n[top] - half
                    st_stage[top + 1] = 0
                    top += 1
                    break
                val = st_first[top] + val
                top -= 1
            if top < 0:
                return 0.0 + val
        else:
            st_stage[top] = 1
            half = st_n[top] // 2
            half -= half % 8
            st_lo[top + 1] = st_lo[top]
            st_n[top + 1] = half
            st_stage[top + 1] = 0
            top += 1


# The compiled EM's steps (see _em). Either sense's probabilities of a sentence sum to 1, so a word's count in the
# second sense is its count in the sentences fitted on less its count in the first, and which sense a sentence is in
# depends only on the difference of its log-probabilities under the two: each step reads one number for each entry of
# the sentences' matrices (in the counts, the first sense's probability; in the log-probabilities, the difference of
# the senses' evidence), not two, and the sums over all of a split's sentences (each sense's total log-probability, its
# share) are made from the words' counts, not from each sentence. Sums over the sentences or the words run in any
# order (fastmath's reassociation, _SUMS): the processor then makes several at a time; rounding, the last bits, differs.
# Exponentials and logarithms are made with the arithmetic alone (_exp, _log), without a branch, so that a loop of
# them runs several at a time (numba's own call a function for each number).
_SUMS = {"reassoc", "contract"}
_NO_INDEX = np.empty(0, dtype=np.int64)  # the arguments of the prior's rates, without one
_NO_PROFILES = np.empty((0, 0), dtype=np.float32)
_NO_RATES = np.empty(0)
_LEAST = np.finfo(np.float64).tiny  # the least positive normal double
_LOG_LEAST = math.log(_LEAST)
_LN2_HI = 6.93147180369123816490e-01  # log 2 in two parts (fdlibm's): k times the first is exact for |k| < 2^20
_LN2_LO = 1.90821492927058770002e-10


@numba.njit(cache=True, nogil=True, error_model="numpy", fastmath={"contract"})
def _exp(x):
    """exp(x) for x <= 0, to a unit in the last place, 0 below the least normal double: x = k log 2 + r with |r| <=
    log(2) / 2, exp(r) by its series to the term under the rounding of a double (r^14 / 14! < 2^-53), times 2^k made
    of its bits."""
    y = max(x, _LOG_LEAST)
    k = np.floor(y * 1.4426950408889634 + 0.5)  # y / log 2, rounded
    r = (y - k * _LN2_HI) - k * _LN2_LO
    e = 1.0 / 6227020800.0
    e = e * r + 1.0 / 479001600.0
    e = e * r + 1.0 / 39916800.0
    e = e * r + 1.0 / 3628800.0
    e = e * r + 1.0 / 362880.0
    e = e * r + 1.0 / 40320.0
    e = e * r + 1.0 / 5040.0
    e = e * r + 1.0 / 720.0
    e = e * r + 1.0 / 120.0
    e = e * r + 1.0 / 24.0
    e = e * r + 1.0 / 6.0
    e = e * r + 0.5
    e = e * r + 1.0
    e = e * r + 1.0
    scale = np.int64((np.int64(k) + 1023) << 52).view(np.float64)
    return e * scale if x >= _LOG_LEAST else 0.0


@numba.njit(cache=True, nogil=True, error_model="numpy", fastmath={"contract"})
def _log(x):
    """log(x) for a positive normal double x, to two units in the last place: x = m 2^k with m in [sqrt(1/2),
    sqrt(2)), log m = 2 atanh(s), s = (m - 1) / (m + 1), by its series to the term under the rounding of a double
    (s^22 / 23 < 2^-53, s^2 <= 0.0295), plus k log 2."""
    bits = np.float64(x).view(np.int64)
    k = (bits >> 52) - 1023
    m = np.int64((bits & 0xFFFFFFFFFFFFF) | 0x3FF0000000000000).view(np.float64)
    big = m > 1.4142135623730951
    m = 0.5 * m if big else m
    k = k + 1 if big else k
    f = m - 1.0
    s = f / (2.0 + f)
    z = s * s
    q = 1.0 / 23.0
    q = q * z + 1.0 / 21.0
    q = q * z + 1.0 / 19.0
    q = q * z + 1.0 / 17.0
    q = q * z + 1.0 / 15.0
    q = q * z + 1.0 / 13.0
    q = q * z + 1.0 / 11.0
    q = q * z + 1.0 / 9.0
    q = q * z + 1.0 / 7.0
    q = q * z + 1.0 / 5.0
    q = q * z + 1.0 / 3.0
    fk = np.float64(k)
    return fk * _LN2_HI + (2.0 * s + 2.0 * s * z * q + fk * _LN2_LO)


@numba.njit(cache=True, nogil=True)
def _em_start(r, perm, spans, unmarked, unmarked_ll, r0, r1, U):
    """A part's sentences' probabilities of either sense to start from (r0, r1: r's, n x 2, in the order of its packing,
    perm, or as they are when perm is empty) for the sentences fitted on (spans: the others' stay 0), and what the words
    that mark no sense tell of those (each group's weighted count of them, unmarked, added to U). Returns the sums of r0
    and r1, and of the sentences' log-probabilities under the parent from those words (unmarked_ll)."""
    w0 = 0.0
    w1 = 0.0
    ll = 0.0
    for s in range(spans.shape[0]):
        for i in range(spans[s, 0], spans[s, 1]):
            j = perm[i] if len(perm) else i
            r0[i] = r[j, 0]
            r1[i] = r[j, 1]
            w0 += r0[i]
            w1 += r1[i]
            ll += unmarked_ll[i]
            for c in range(U.shape[0]):
                U[c] += unmarked[c, i]
    return w0, w1, ll


@numba.njit(cache=True, nogil=True)
def _em_counts(order, start, idx, cdata, g_off, ones, r0, blocks, counts):
    """Each word's (soft) count in the first sense (X.T @ r0 for each group), in the sentences of the blocks given,
    for the words the sentences have (and one more, for the slices' empty places), into counts (n_cols + 1), from the
    slices of the words' entries. r0: n + 1 (the padding's last, 0)."""
    one, two, three, four = np.uint64(1), np.uint64(2), np.uint64(3), np.uint64(4)
    counts[:] = 0.0
    for b in blocks:
        for c in range(len(ones)):
            for sl in range(g_off[b, c], g_off[b, c + 1]):
                a0 = 0.0
                a1 = 0.0
                a2 = 0.0
                a3 = 0.0
                e = start[sl]
                end = start[sl + 1]
                if ones[c]:
                    while e < end:
                        a0 += r0[np.uint64(idx[e])]
                        a1 += r0[np.uint64(idx[e + one])]
                        a2 += r0[np.uint64(idx[e + two])]
                        a3 += r0[np.uint64(idx[e + three])]
                        e += four
                else:
                    while e < end:
                        a0 += cdata[e] * r0[np.uint64(idx[e])]
                        a1 += cdata[e + one] * r0[np.uint64(idx[e + one])]
                        a2 += cdata[e + two] * r0[np.uint64(idx[e + two])]
                        a3 += cdata[e + three] * r0[np.uint64(idx[e + three])]
                        e += four
                q = _WIDTH * sl
                counts[order[q]] += a0
                counts[order[q + 1]] += a1
                counts[order[q + 2]] += a2
                counts[order[q + 3]] += a3


@numba.njit(cache=True, nogil=True)
def _em_row_counts(indptr, indices, data, ones, r0, bounds, blocks, counts):
    """_em_counts from the sentences' entries (indptr groups x n + 1, into the entries of all; each entry's word among
    the n_cols the sentences have; counts, read for the groups not all ones), each added to its word's count."""
    one = np.uint64(1)
    counts[:] = 0.0
    for b in blocks:
        for c in range(indptr.shape[0]):
            for i in range(bounds[b], bounds[b + 1]):
                r = r0[i]
                jj = np.uint64(indptr[c, i])
                end = np.uint64(indptr[c, i + 1])
                if ones[c]:
                    while jj < end:
                        counts[np.uint64(indices[jj])] += r
                        jj += one
                else:
                    while jj < end:
                        counts[np.uint64(indices[jj])] += data[jj] * r
                        jj += one


@numba.njit(cache=True, nogil=True, error_model="numpy", fastmath=_SUMS)
def _em_sums(c0, T, cmp_off, discount, totals, least):
    """Each group's total count in either sense, and total of the counts' least with its discount, over the words the
    sentences have (groups x 2 each): the first sense's counts c0, the second's each word's count T less those."""
    for c in range(len(cmp_off) - 1):
        d = discount[c]
        s0 = 0.0
        s1 = 0.0
        l0 = 0.0
        l1 = 0.0
        first = np.uint64(cmp_off[c])
        for q in range(cmp_off[c + 1] - cmp_off[c]):
            j = first + np.uint64(q)  # unsigned: numba checks a signed index for being negative, and the loop is not
            a = c0[j]  # made several at a time
            b = max(T[j] - a, 0.0)
            s0 += a
            s1 += b
            l0 += min(a, d)
            l1 += min(b, d)
        totals[c, 0] = s0
        totals[c, 1] = s1
        least[c, 0] = l0
        least[c, 1] = l1


@numba.njit(cache=True, nogil=True)
def _em_fit_marked(
    c0,
    T,
    totals,
    least,
    thresholds,
    parent_mk,
    parent_had,
    mk,
    mk_cmp,
    mk_off,
    discount,
    every,
    marked,
    p,
    scales,
    chosen,
):
    """_em_fit for the markers the sentences have, from the counts of the words they have (c0, the first sense's; T,
    each word's), the groups' totals and sums of the counts' least with the discount over all their words (least),
    and their parent's probabilities (parent_mk, of the markers mk; parent_had, of the words): the markers'
    probabilities (p, 2 x markers, and 1 for the padding of the slices), the scale of either sense's rates for the words
    not its own (scales, groups x 2), and which of the markers have rates of their own (chosen); returns whether any
    word tells the senses apart. mk are the markers of each group (from mk_off), mk_cmp their place among the words
    the sentences have (-1 for none: such a word may still mark a sense, as rarer in both than in the parent); marked,
    those the sentences have (Mats.marking), whose probabilities are made (the markers of the root, when those chosen
    are held)."""
    mk_had, mk_pos, mkh_off = marked
    n_groups = len(mkh_off) - 1
    n_mk = len(mk_had)
    p[0, n_mk] = 1.0  # the padding of the slices of the markers' entries: its evidence is 0
    p[1, n_mk] = 1.0
    scales[:, :] = 1.0
    any_own = False
    chosen[:] = False
    own_parent = np.empty(len(mk))
    own_cmp = np.empty(len(mk), dtype=np.int64)
    for c in range(n_groups):
        n_own = 0
        for m in range(mk_off[c], mk_off[c + 1]):
            q = mk_cmp[m]
            c0_m = c0[q] if q >= 0 else 0.0
            c1_m = max(T[q] - c0_m, 0.0) if q >= 0 else 0.0
            is_own = False
            if every:
                is_own = c0_m + c1_m > 0
            else:
                for k in range(2):
                    t = totals[c, k]
                    if t <= 0:
                        continue
                    co = c0_m if k == 0 else c1_m
                    eo = t * parent_mk[m]
                    if co != eo:
                        if eo > 0 and t > eo:  # G2 <= 2 t (co - eo)^2 / (eo (t - eo)), as KL <= chi-square
                            bound = 2.0 * (co - eo) * (co - eo) * t / (eo * (t - eo))
                            if bound * (1.0 + 1e-9) + 1e-9 * (co + eo) < thresholds[c, k]:  # margins over rounding's
                                continue
                        g = 2.0 * (_xlogy(co, co / eo) + _xlogy(t - co, (t - co) / (t - eo)))
                        if g > thresholds[c, k]:
                            is_own = True
                            break
            if is_own:
                chosen[m] = True
                own_parent[n_own] = parent_mk[m]
                own_cmp[n_own] = q
                n_own += 1
        any_own = any_own or n_own > 0
        a, b = mkh_off[c], mkh_off[c + 1]
        d = discount[c]
        for k in range(2):
            t = totals[c, k]
            if t <= 0 or n_own == 0:
                for m in range(a, b):
                    p[k, m] = parent_had[mk_had[m]]
                continue
            freed = least[c, k] / t
            p_own = np.empty(n_own)
            for m in range(n_own):
                q = own_cmp[m]
                co = 0.0
                if q >= 0:
                    co = c0[q] if k == 0 else max(T[q] - c0[q], 0.0)
                p_own[m] = max(co - d, 0.0) / t + freed * own_parent[m]
            others = 1.0 - _pairwise_sum(own_parent, 0, n_own)
            scale = (1.0 - _pairwise_sum(p_own, 0, n_own)) / others if others > 0 else 0.0
            scales[c, k] = scale
            for m in range(a, b):
                p[k, m] = parent_had[mk_had[m]] * scale
            for m in range(n_own):
                if own_cmp[m] >= 0:
                    p[k, mk_pos[own_cmp[m]]] = p_own[m]
    return any_own


@numba.njit(cache=True, nogil=True)
def _prior_rates_marked(p, c0, T, set_m, set_cmp, q, strengths):
    """_prior_rates for the markers' rates p (2 x markers), from the counts of the words the sentences have (c0, the
    first sense's; T, each word's)."""
    for k in range(2):
        a = strengths[k]
        N = 0.0
        pi = 0.0
        for i in range(len(set_m)):
            j = set_cmp[i]
            N += c0[j] if k == 0 else max(T[j] - c0[j], 0.0)
            pi += p[k, set_m[i]]
        if N + a <= 0:
            continue
        for i in range(len(set_m)):
            p[k, set_m[i]] = (N * p[k, set_m[i]] + a * pi * q[k, i]) / (N + a)


@numba.njit(cache=True, nogil=True, error_model="numpy", fastmath=_SUMS)
def _em_evidence(p0s, p1s, w, T_mk, de):
    """The difference of what each marker tells of either sense (de: w log p0 - w log p1, markers + 1, the padding's
    0, from either sense's rates p0s, p1s); returns the sum of what they tell of the second over the sentences fitted on (each marker's times its count
    there, T_mk), and whether any probability is not a positive normal double (then made again by numpy's log, and
    told as False if not finite)."""
    n = len(w)
    s1 = 0.0
    odd = 0
    for m in range(n):  # either sense's rates apart: numba reads a 2-D array through a stride it multiplies by
        p0 = p0s[m]
        p1 = p1s[m]
        e1 = w[m] * _log(p1)
        de[m] = w[m] * _log(p0) - e1
        s1 += e1 * T_mk[m]
        odd += not (p0 >= _LEAST and p0 <= 1.0 and p1 >= _LEAST and p1 <= 1.0)
    if odd == 0:
        return s1, True
    s1 = 0.0
    for m in range(n):
        e1 = w[m] * np.log(p1s[m])
        de[m] = w[m] * np.log(p0s[m]) - e1
        if not (np.isfinite(e1) and np.isfinite(de[m])):
            return 0.0, False
        s1 += e1 * T_mk[m]
    return s1, True


@numba.njit(cache=True, nogil=True, error_model="numpy", fastmath=_SUMS)
def _em_other_words(unmarked, dls, dpi, spans, d):
    """The difference of the log-probabilities of the sentences fitted on (spans, k x 2, from and to) under either
    sense from their shares (dpi) and from the words that mark no sense: each group's weighted count of those words
    (unmarked, groups x n) times the difference of the logs of the scales of either sense's rates for them (dls), into
    d. What those words tell under the parent is the same in both, and drops out."""
    for s in range(spans.shape[0]):
        first = np.uint64(spans[s, 0])
        for q in range(spans[s, 1] - spans[s, 0]):
            d[first + np.uint64(q)] = dpi  # unsigned, as in _em_sums
    for c in range(unmarked.shape[0]):
        g = dls[c]
        row = unmarked[c]
        for s in range(spans.shape[0]):
            first = np.uint64(spans[s, 0])
            for q in range(spans[s, 1] - spans[s, 0]):
                i = first + np.uint64(q)
                d[i] += row[i] * g


@numba.njit(cache=True, nogil=True)
def _em_marker_sums(order, start, idx, sdata, ones, de, ranges, d):
    """Each group's sum of the differences de of each sentence's markers' entries, added to d (n + 1), for the slices
    in the ranges given (k x 2, from and to): _WIDTH sentences' at once."""
    n_groups = order.shape[0]
    one, two, three, four = np.uint64(1), np.uint64(2), np.uint64(3), np.uint64(4)
    for c in range(n_groups):
        for g in range(ranges.shape[0]):
            for sl in range(ranges[g, 0], ranges[g, 1]):
                a0 = 0.0
                a1 = 0.0
                a2 = 0.0
                a3 = 0.0
                e = start[c, sl]
                end = start[c, sl + 1]
                if ones[c]:
                    while e < end:
                        a0 += de[np.uint64(idx[e])]
                        a1 += de[np.uint64(idx[e + one])]
                        a2 += de[np.uint64(idx[e + two])]
                        a3 += de[np.uint64(idx[e + three])]
                        e += four
                else:
                    while e < end:
                        a0 += sdata[e] * de[np.uint64(idx[e])]
                        a1 += sdata[e + one] * de[np.uint64(idx[e + one])]
                        a2 += sdata[e + two] * de[np.uint64(idx[e + two])]
                        a3 += sdata[e + three] * de[np.uint64(idx[e + three])]
                        e += four
                q = sl * _WIDTH
                d[order[c, q]] += a0
                d[order[c, q + 1]] += a1
                d[order[c, q + 2]] += a2
                d[order[c, q + 3]] += a3


@numba.njit(cache=True, nogil=True, error_model="numpy", fastmath=_SUMS)
def _em_posteriors(d, r0, r1):
    """Each sentence's probability of either sense (r0, r1) from the difference d of its log-probabilities under them:
    (the sums of r0 and r1, the sum of log(exp(d) + 1), each sentence's log-probability less the second sense's, and
    how many d are undefined)."""
    w0 = 0.0
    w1 = 0.0
    soft = 0.0
    undefined = 0
    for i in range(len(d)):  # from 0: the index is not checked for being negative
        x = d[i]
        t = _exp(-abs(x))
        big = 1.0 / (1.0 + t)
        small = t * big
        first = x >= 0.0
        a = big if first else small
        b = small if first else big
        r0[i] = a
        r1[i] = b
        w0 += a
        w1 += b
        soft += max(x, 0.0) + _log(1.0 + t)
        undefined += x != x
    return w0, w1, soft, undefined


@numba.njit(cache=True, nogil=True)
def _em_shares(rows, ranges, spans, de, dls, dpi, unmarked, d, r0, r1):
    """A part's E step: its sentences' probabilities of either sense (r0, r1), from the markers' entries (rows: their
    slices, as _Mats.marking makes them) and the other words' (unmarked), of the sentences fitted on (spans, and their
    slices, ranges); d, n + 1, the differences of their log-probabilities. Returns the sums of _em_posteriors."""
    order, start, idx, sdata, ones = rows
    _em_other_words(unmarked, dls, dpi, spans, d)
    _em_marker_sums(order, start, idx, sdata, ones, de, ranges, d)
    w0 = 0.0
    w1 = 0.0
    soft = 0.0
    undefined = 0
    for s in range(spans.shape[0]):
        lo, hi = spans[s, 0], spans[s, 1]
        a0, a1, a2, a3 = _em_posteriors(d[lo:hi], r0[lo:hi], r1[lo:hi])
        w0 += a0
        w1 += a1
        soft += a2
        undefined += a3
    return w0, w1, soft, undefined


@numba.njit(cache=True, nogil=True)
def _em_m_step(c0, T, fit, prior):
    """The M step from the counts of the words the sentences have (c0, the first sense's; T, each word's): the groups'
    totals and thresholds, the markers' rates (and the prior's, the corpus centre set at the first step), what they tell
    of the sentences (de, and dls for the other words). Returns (0 if done, 1 if no word tells the senses apart, 2 if a
    rate is not a positive number; the sum of what the markers and the scales tell of the second sense over the
    sentences fitted on)."""
    cmp_off, discount, parent_mk, parent_had, mk, mk_cmp, mk_off, every, marked, w_mk, T_mk, U = fit[0]
    totals, least, thresholds, p, scales, chosen, de, dls = fit[1]
    _em_sums(c0, T, cmp_off, discount, totals, least)
    for c in range(totals.shape[0]):
        for k in range(2):
            thresholds[c, k] = np.log(totals[c, k]) if totals[c, k] > 0 else -np.inf
    any_own = _em_fit_marked(
        c0,
        T,
        totals,
        least,
        thresholds,
        parent_mk,
        parent_had,
        mk,
        mk_cmp,
        mk_off,
        discount,
        every,
        marked,
        p,
        scales,
        chosen,
    )
    if not any_own:
        return 1, 0.0
    set_m, set_cmp, Q, slots, tokens, strength, centre, strengths, centre_set = prior
    if len(set_m) > 0:
        if not centre_set[0]:  # set where the fit starts, from the counts it starts from, then held (see _em)
            parent_set = np.empty(len(set_cmp))
            c_k = np.empty(len(set_cmp))
            w = np.empty(len(set_cmp))
            for k in range(2):
                for i in range(len(set_cmp)):
                    j = set_cmp[i]
                    parent_set[i] = parent_had[j]
                    c_k[i] = c0[j] if k == 0 else max(T[j] - c0[j], 0.0)
                _anchors(c_k, parent_set, totals[0, k], thresholds[0, k], w)
                sure = _centre(Q, slots, w, parent_set, centre[k], tokens)
                strengths[k] = min(strength, max(0.0, sure - 1.0))  # as sure as its own centre
            centre_set[0] = True
        _prior_rates_marked(p, c0, T, set_m, set_cmp, centre, strengths)
    s1, fine = _em_evidence(p[0], p[1], w_mk, T_mk, de)
    for c in range(len(dls)):
        l0 = np.log(scales[c, 0])
        l1 = np.log(scales[c, 1])
        dls[c] = l0 - l1
        if U[c] != 0.0:
            s1 += l1 * U[c]
        fine = fine and np.isfinite(dls[c]) and np.isfinite(l1)
    return (0 if fine else 2), s1


@numba.njit(cache=True, nogil=True)
def _em_step(columns, r0, r1, T, c0, fit, prior, shares, dpi):
    """An EM step of a split packed whole (by words: _Mats.columns): the counts of the words in the first sense, the M
    step and the E step, at once. Returns (_em_m_step's two, then _em_shares' four, 0 if not made)."""
    order, start, idx, cdata, g_off, ones, blocks = columns
    _em_counts(order, start, idx, cdata, g_off, ones, r0, blocks, c0)
    status, s1 = _em_m_step(c0, T, fit, prior)
    if status != 0:
        return status, s1, 0.0, 0.0, 0.0, 0
    rows, ranges, spans, unmarked, d = shares
    w0, w1, soft, undefined = _em_shares(rows, ranges, spans, fit[1][6], fit[1][7], dpi, unmarked, d, r0, r1)
    return status, s1, w0, w1, soft, undefined


@numba.njit(cache=True, nogil=True)
def _em_fit(counts, totals, thresholds, parent_p, parent_rest, markers, discount, voc_off, every_marker):
    """Both senses' probabilities for each group (fit_sparse with the words of _split_words): p (2 x words), rest
    (groups x 2), whether each is its parent's, and whether any word tells the senses apart."""
    n_groups = len(voc_off) - 1
    n_words = len(parent_p)
    own = np.zeros(n_words, dtype=np.bool_)
    any_own = False
    for c in range(n_groups):
        for j in range(voc_off[c], voc_off[c + 1]):
            if not markers[j]:
                continue
            if every_marker:
                own[j] = counts[j, 0] + counts[j, 1] > 0
            else:
                for k in range(2):
                    t = totals[c, k]
                    if t <= 0:
                        continue
                    co = counts[j, k]
                    eo = t * parent_p[j]
                    if co != eo:
                        g = 2.0 * (_xlogy(co, co / eo) + _xlogy(t - co, (t - co) / (t - eo)))
                        if g > thresholds[c, k]:
                            own[j] = True
                            break
            any_own = any_own or own[j]
    p = np.empty((2, n_words))
    rest = np.empty((n_groups, 2))
    is_parent = np.zeros((n_groups, 2), dtype=np.bool_)
    for c in range(n_groups):
        lo = voc_off[c]
        hi = voc_off[c + 1]
        own_c = np.flatnonzero(own[lo:hi]) + lo
        d = discount[c]
        for k in range(2):
            t = totals[c, k]
            if t <= 0 or len(own_c) == 0:
                is_parent[c, k] = True
                p[k, lo:hi] = parent_p[lo:hi]
                rest[c, k] = parent_rest[c]
                continue
            least = np.empty(hi - lo)
            for j in range(lo, hi):
                least[j - lo] = min(counts[j, k], d)
            freed = _pairwise_sum(least, 0, hi - lo) / t
            p_own = np.empty(len(own_c))
            parent_own = np.empty(len(own_c))
            for m in range(len(own_c)):
                j = own_c[m]
                p_own[m] = max(counts[j, k] - d, 0.0) / t + freed * parent_p[j]
                parent_own[m] = parent_p[j]
            others = 1.0 - _pairwise_sum(parent_own, 0, len(own_c))
            scale = (1.0 - _pairwise_sum(p_own, 0, len(own_c))) / others if others > 0 else 0.0
            for j in range(lo, hi):
                p[k, j] = parent_p[j] * scale
            for m in range(len(own_c)):
                p[k, own_c[m]] = p_own[m]
            rest[c, k] = parent_rest[c] * scale
    return p, rest, is_parent, any_own


@numba.njit(cache=True, nogil=True)
def _em_loglik(order, start, idx, sdata, ones, evidence, log_pi, n, ranges):
    """Each of the n sentences' (weighted) log-probability under either sense of a split, with its share: n + 1 x 2
    (the last, the padding's, to be ignored), from the slices of the sentences' entries in the ranges given (those of
    the other sentences only their share). evidence: words x 2, flat, the padding's last (0). Each sentence's sum of
    each group still adds its entries in order, from 0, and the groups' sums in order: _WIDTH sentences' at once."""
    ll = np.empty(2 * (n + 1))
    for i in range(n + 1):
        ll[2 * i] = log_pi[0]
        ll[2 * i + 1] = log_pi[1]
    _em_loglik_add(order, start, idx, sdata, ones, evidence, ll, ranges)
    return ll.reshape(n + 1, 2)


@numba.njit(cache=True, nogil=True)
def _em_marked_loglik(order, start, idx, sdata, ones, evidence, log_pi, unmarked_ll, unmarked, log_scale, ranges):
    """_em_loglik from the markers' entries alone (order ... evidence: their slices and evidence), the other words'
    from what they tell under the parent (unmarked_ll, n) and their weighted counts (unmarked, groups x n) times the log
    of the scale of either sense's rates (log_scale, groups x 2): for each sentence, the share and the other words'
    log-probability, each group's count times its log-scale, then each group's sum of the markers' entries, as the
    senses' matrices of those words make them (the markers' of the slices in the ranges given): n + 1 x 2, as
    _em_loglik."""
    n_groups, n = unmarked.shape
    ll = np.zeros(2 * (n + 1))
    for i in range(n):
        ll[2 * i] = log_pi[0] + unmarked_ll[i]
        ll[2 * i + 1] = log_pi[1] + unmarked_ll[i]
    for c in range(n_groups):
        l0 = log_scale[c, 0]
        l1 = log_scale[c, 1]
        if np.isfinite(l0) and np.isfinite(l1):
            for i in range(n):
                ll[2 * i] += unmarked[c, i] * l0
                ll[2 * i + 1] += unmarked[c, i] * l1
        else:  # a scale of 0: nothing for a sentence without such words, not 0 times its log
            for i in range(n):
                if unmarked[c, i] != 0.0:
                    ll[2 * i] += unmarked[c, i] * l0
                    ll[2 * i + 1] += unmarked[c, i] * l1
    _em_loglik_add(order, start, idx, sdata, ones, evidence, ll, ranges)
    return ll.reshape(n + 1, 2)


@numba.njit(cache=True, nogil=True)
def _em_loglik_add(order, start, idx, sdata, ones, evidence, ll, ranges):
    """Each group's sum of the evidence of each sentence's entries, added to ll (n + 1 x 2, flat), for the slices in
    the ranges given (k x 2, from and to)."""
    n_groups = order.shape[0]
    one, two, three, four = np.uint64(1), np.uint64(2), np.uint64(3), np.uint64(4)
    for c in range(n_groups):
        for g in range(ranges.shape[0]):
            for sl in range(ranges[g, 0], ranges[g, 1]):
                a0 = 0.0
                b0 = 0.0
                a1 = 0.0
                b1 = 0.0
                a2 = 0.0
                b2 = 0.0
                a3 = 0.0
                b3 = 0.0
                e = start[c, sl]
                end = start[c, sl + 1]
                if ones[c]:
                    while e < end:
                        j = np.uint64(idx[e]) * two
                        a0 += evidence[j]
                        b0 += evidence[j + one]
                        j = np.uint64(idx[e + one]) * two
                        a1 += evidence[j]
                        b1 += evidence[j + one]
                        j = np.uint64(idx[e + two]) * two
                        a2 += evidence[j]
                        b2 += evidence[j + one]
                        j = np.uint64(idx[e + three]) * two
                        a3 += evidence[j]
                        b3 += evidence[j + one]
                        e += four
                else:
                    while e < end:
                        j = np.uint64(idx[e]) * two
                        a0 += sdata[e] * evidence[j]
                        b0 += sdata[e] * evidence[j + one]
                        j = np.uint64(idx[e + one]) * two
                        a1 += sdata[e + one] * evidence[j]
                        b1 += sdata[e + one] * evidence[j + one]
                        j = np.uint64(idx[e + two]) * two
                        a2 += sdata[e + two] * evidence[j]
                        b2 += sdata[e + two] * evidence[j + one]
                        j = np.uint64(idx[e + three]) * two
                        a3 += sdata[e + three] * evidence[j]
                        b3 += sdata[e + three] * evidence[j + one]
                        e += four
                q = sl * _WIDTH
                i = order[c, q] * two
                ll[i] += a0
                ll[i + one] += b0
                i = order[c, q + 1] * two
                ll[i] += a1
                ll[i + one] += b1
                i = order[c, q + 2] * two
                ll[i] += a2
                ll[i + one] += b2
                i = order[c, q + 3] * two
                ll[i] += a3
                ll[i + one] += b3


def _weights(r):
    """r.sum(axis=0) of an n x 2 array: in order down each column, as numpy sums a C-contiguous one, compiled."""
    return _column_sums(r) if r.flags.c_contiguous else r.sum(axis=0)


@numba.njit(cache=True, nogil=True)
def _column_sums(r):
    s0 = 0.0
    s1 = 0.0
    for i in range(r.shape[0]):
        s0 += r[i, 0]
        s1 += r[i, 1]
    out = np.empty(2)
    out[0] = s0
    out[1] = s1
    return out


class _Fit:
    """A split fitted (_em): (pi, dists, r, counts), and the log-probabilities of the sentences left out when a block
    was, read as a tuple is (indexed, sliced, unpacked); the senses' rates for every word (dists) and their counts are
    made when first read: of the candidate splits of a sense, only the one kept needs them."""

    def __init__(self, pi, r, held_out, senses):
        self._items = [pi, None, r, None] + ([] if held_out is None else [held_out])
        self._senses = senses
        self._lock = threading.Lock()

    def __len__(self):
        return len(self._items)

    def __getitem__(self, k):
        if isinstance(k, slice):
            return tuple(self[i] for i in range(*k.indices(len(self))))
        k = range(len(self))[k]
        if k in (1, 3):
            with self._lock:
                if self._senses is not None:
                    self._items[1], self._items[3] = self._senses()
                    self._senses = None
        return self._items[k]

    def __iter__(self):
        return iter(self[:])


def _em(Xs, parent, r, every_marker, switch=True, left_out=None, submit=_now, prior=None, held_only=False):
    """The EM of a split (see fit_split), from r, each sentence's probabilities of either sense (summing to 1). The
    words chosen at every step (every_marker false), once the set chosen comes back to one chosen before, after
    another, are held from then on (switch): the choice cycles, and the fit would not settle. left_out, a block of the
    sentences (Xs a _Mats with blocks: a fold), fits the split without its sentences (their entries not read, their
    probabilities of either sense zero), whose log-probabilities under the fit are then returned too (n_block x 2, in
    their order); held_only, those alone (the fit's senses are not made). Xs may be in parts (_Parts), whose steps are
    made at once on the pool's threads (submit). prior, (profiles, strength) or None: the senses' rates for the
    profiled words have the prior of their corpus profiles (_Profiles), as the fit starts (the words that mark either
    then, from the counts of the sentences it is fitted on), of that strength or of what each profile is made from if
    less (a MAP fit: the prior's centre stays where it was set). None when a sense ends up with less than a sentence,
    none of the words tells them apart, or a rate is not a positive number."""
    parts = list(Xs) if isinstance(Xs, _Parts) else [Xs if isinstance(Xs, _Mats) else _Mats(Xs, parent=parent)]
    Xs = parts[0]
    pos, cols, cmp_off, voc_off = Xs.words  # the same for every part
    at = np.concatenate([[0], np.cumsum([part[0].shape[0] for part in parts])]).astype(np.int64)  # each part's first
    n = int(at[-1])
    lo, hi = (n, n) if left_out is None else (Xs.bounds[left_out], Xs.bounds[left_out + 1])
    n_fit = n - (hi - lo)
    n_cols = len(cols)
    discount = np.array([d.discount for d in parent], dtype=np.float64)
    markers, mk, mk_off, mk_cmp, (mk_had, mk_pos, mkh_off, w_mk, _, _) = Xs.marking(parent)
    marked = (mk_had, mk_pos, mkh_off)
    # The parent's rates of the words the sentences have, and of the markers
    parent_had = np.concatenate([d.p[pos[cmp_off[c] : cmp_off[c + 1]]] for c, d in enumerate(parent)], dtype=np.float64)
    parent_mk = np.concatenate(
        [d.p[mk[mk_off[c] : mk_off[c + 1]] - voc_off[c]] for c, d in enumerate(parent)], dtype=np.float64
    )
    no_prior = (_NO_INDEX, _NO_INDEX, _NO_PROFILES, _NO_INDEX, _NO_RATES, 0.0, np.empty((2, 0)), np.zeros(2))
    prior_args = no_prior + (np.ones(1, dtype=np.bool_),)
    if prior is not None and prior[1] > 0:
        set_m, set_cmp, slots = _prior_places(prior[0], Xs, parent)
        if len(set_m):
            profiles = prior[0]
            centre, strengths = np.empty((2, len(set_m))), np.full(2, float(prior[1]))
            prior_args = (set_m, set_cmp, profiles.Q, slots, profiles.tokens, float(prior[1]), centre, strengths)
            prior_args += (np.zeros(1, dtype=np.bool_),)
    # Each part's sentences in the order of its packing, with one more (the padding of the slices, in no sense): their
    # probabilities of either sense, the differences of their log-probabilities under the two; those of the sentences
    # left out stay 0
    steps = []
    T = np.zeros(n_cols + 1)  # each word's count in the sentences fitted on
    U = np.zeros(len(parent))  # each group's weighted count of the words that mark no sense in them
    L = 0.0  # what those words tell of them under the parent
    weight = np.zeros(2)
    for p, part in enumerate(parts):
        a, b = at[p], at[p + 1]
        ranges, blocks = part.ranges(left_out)
        bounds = part.bounds
        spans = np.array([[bounds[k], bounds[k + 1]] for k in blocks], dtype=np.int64).reshape(-1, 2)
        m_rows, unmarked = part.marking(parent)[4][4:]
        unmarked_ll = part.unmarked_loglik(parent)
        r0, r1 = np.zeros(b - a + 1), np.zeros(b - a + 1)
        perm = _NO_INDEX if part.perm is None else part.perm
        w0, w1, ll = _em_start(
            np.ascontiguousarray(r[a:b], dtype=np.float64), perm, spans, unmarked, unmarked_ll, r0, r1, U
        )
        weight += [w0, w1]
        L += ll
        T += part.word_counts(blocks)
        rows = (*m_rows[:4], part.ones)
        steps.append((part, blocks, r0, r1, (rows, ranges, spans, unmarked, np.empty(b - a + 1)), unmarked_ll))
    T_mk = np.append(T[mk_had], 0.0)
    n_mk = len(mk_had)
    buffers = (
        np.empty((len(parent), 2)),  # totals
        np.empty((len(parent), 2)),  # least
        np.empty((len(parent), 2)),  # thresholds
        np.empty((2, n_mk + 1)),  # the markers' rates
        np.empty((len(parent), 2)),  # scales
        np.empty(len(mk), dtype=np.bool_),  # chosen
        np.empty(n_mk + 1),  # de
        np.empty(len(parent)),  # dls
    )
    fit = ((cmp_off, discount, parent_mk, parent_had, mk, mk_cmp, mk_off, bool(every_marker), marked, w_mk, T_mk, U),)
    fit += (buffers,)
    whole = len(parts) == 1 and Xs.columns is not None  # each step in one compiled call
    c0 = np.empty(n_cols + 1)

    def counts_of(p):  # each word's count in the first sense in a part's sentences
        part, blocks, r0 = steps[p][:3]
        out = np.empty(n_cols + 1)
        if part.columns is None:
            _em_row_counts(*part.entries, part.ones, r0, part.bounds, blocks, out)
        else:
            _em_counts(*part.columns, part.ones, r0, blocks, out)
        return out

    def shares_of(p, dpi):  # a part's E step
        r0, r1, shares = steps[p][2:5]
        rows, ranges, spans, unmarked, d = shares
        return _em_shares(rows, ranges, spans, buffers[6], buffers[7], dpi, unmarked, d, r0, r1)

    previous = None
    seen, last = set(), None  # the sets of words chosen
    for _ in range(_MAX_ITER):
        if weight.min() < 1.0:
            return None
        pi = weight / n_fit
        log_pi = np.log(pi)
        dpi = float(log_pi[0] - log_pi[1])
        if whole:
            part, blocks, r0, r1, shares = steps[0][:5]
            columns = (*part.columns, part.ones, blocks)
            status, s1, w0, w1, soft, undefined = _em_step(columns, r0, r1, T, c0, fit, prior_args, shares, dpi)
        else:
            made = _at_once(submit, counts_of, range(len(parts)))
            c0 = made[0]
            for more in made[1:]:
                c0 = c0 + more
            status, s1 = _em_m_step(c0, T, fit, prior_args)
            if status == 0:
                made = _at_once(submit, lambda p: shares_of(p, dpi), range(len(parts)))
                w0, w1, soft, undefined = (sum(m[k] for m in made) for k in range(4))
        if status != 0 or undefined:  # no word tells the senses apart, or a sentence is impossible in both
            return None
        if switch and not every_marker:
            chosen = buffers[5]
            key = chosen.tobytes()
            if key in seen and key != last:  # the choice cycles: these words, held from now on
                mk, mk_cmp, parent_mk = mk[chosen], mk_cmp[chosen], parent_mk[chosen]
                mk_off = np.searchsorted(mk, voc_off)
                markers = np.zeros(int(voc_off[-1]), dtype=np.bool_)
                markers[mk] = True
                every_marker = True
                fit = ((cmp_off, discount, parent_mk, parent_had, mk, mk_cmp, mk_off, True, marked, w_mk, T_mk, U),)
                fit += (buffers,)
            seen.add(key)
            last = key
        # Each sentence's log-probability is the second sense's plus log(exp(d) + 1); the sum of the second's is the
        # sum of what the words tell of it, each by its count in the sentences fitted on
        total = n_fit * float(log_pi[1]) + L + s1 + soft
        weight = np.array([w0, w1])
        if previous is not None and abs(total - previous) <= _TOL * abs(total):
            break
        previous = total
    if weight.min() < 1.0:
        return None
    totals, _, thresholds, p_mk, scales = buffers[:5]
    held_out = None
    if left_out is not None:  # the sentences left out under the fit: their markers' entries too, with the same rates
        evidence = np.log(p_mk.T) * w_mk[:, None]
        with np.errstate(divide="ignore"):
            log_scales = np.log(scales)
        rows, unmarked_ll = steps[0][4][0], steps[0][5]
        ranges = np.array([[Xs.first[left_out], Xs.first[left_out + 1]]], dtype=np.int64)
        ll = _em_marked_loglik(*rows, evidence.reshape(-1), log_pi, unmarked_ll, steps[0][4][3], log_scales, ranges)
        held_out = ll[lo:hi].copy()
        if held_only:
            return held_out
    r = []
    for part, _, r0, r1, _, _ in steps:
        rp = np.column_stack([r0[:-1], r1[:-1]])
        r.append(rp if part.perm is None else rp[part.inv])

    def senses():  # the last fit again, for every word: the senses' rates, and their counts
        counts_had = np.column_stack([c0[:n_cols], np.maximum(T[:n_cols] - c0[:n_cols], 0.0)])
        counts = np.zeros((int(voc_off[-1]), 2))
        counts[cols] = counts_had
        parent_p = np.concatenate([d.p for d in parent]).astype(np.float64)
        parent_rest = np.array([d.rest for d in parent], dtype=np.float64)
        p, rest, is_parent, _ = _em_fit(
            counts, totals, thresholds, parent_p, parent_rest, markers, discount, voc_off, every_marker
        )
        if len(prior_args[0]):  # the same rates for the profiled words as the last step's
            set_cmp, centre, strengths = prior_args[1], prior_args[6], prior_args[7]
            _prior_rates(p, counts_had, np.ascontiguousarray(cols[set_cmp]), set_cmp, centre, strengths, True)
            is_parent[0] = False
        dists = [
            [
                (
                    parent[c]
                    if is_parent[c, k]
                    else Dist(
                        p[k, voc_off[c] : voc_off[c + 1]].copy(),
                        float(rest[c, k]),
                        parent[c].markers,
                        parent[c].weights,
                        parent[c].discount,
                    )
                )
                for k in range(2)
            ]
            for c in range(len(parent))
        ]
        counts = [counts[voc_off[c] : voc_off[c + 1]].copy() for c in range(len(parent))]
        return dists, counts

    return _Fit(pi, r[0] if len(r) == 1 else np.concatenate(r), held_out, senses)


def fit_split(Xs, parent, r, search=False, sample=None, first=None, prior=None):
    """Fit a split of the sentences Xs (one matrix for each group of words) into two senses by expectation-maximization,
    from the probabilities r (n x 2) of each sentence being in either, each sense a sparse departure from the parent:
    both have their own rates for the words that mark either, and share the parent's for all others. A search (from a
    seed) first lets every one of the parent's markers tell the senses apart: few of the words of the seed's sense are
    marked yet on its side, and the sense would not grow. The words that mark either are chosen again at every step;
    when the choice comes back to a set it made before, after another, that set is held: such a fit cycles, and never
    settled (stopped by _MAX_ITER). A search's first pass only finds where the second starts: given a sample of the
    sentences (their indices, and their matrices: see search_sample), it runs on those alone, and the second, which
    chooses the words, on all of them. Returns (pi, dists, r, counts), or None when a sense ends up with less than a
    sentence or none of the words tells them apart. first, a list, is given the fit of a search's first pass (whose
    pi, dists are its senses). prior: see _em."""
    if search:
        if sample is None:
            fitted = _em(Xs, parent, r, every_marker=True, prior=prior)
        else:
            kept, Xkept = sample
            fitted = _em(Xkept, parent, r[kept], every_marker=True, prior=prior)
        if fitted is None:
            return None
        if first is not None:
            first.append(fitted)
        if sample is None:
            r = fitted[2]
        else:  # every sentence's probabilities under the first pass's senses
            ll = _mixture_loglik(Xs, fitted[0], fitted[1])
            r = np.exp(ll - _logsumexp2(ll)[:, None])
    return _em(Xs, parent, r, every_marker=False, prior=prior)


def search_sample(Xs, parent=None, marked=None):
    """The sentences a search's first pass runs on (fit_split), for a sense of at least _SAMPLE_EVERY times
    _SAMPLE_LEAST of them: every _SAMPLE_EVERY-th, so that each work keeps its share, and each sense its sentences in
    proportion (a sample of whole works would leave out the senses of a few); (indices, matrices, packed for the
    parent's markers when given, their entries from marked as _Mats takes it), or None for all."""
    n = Xs[0].shape[0]
    if n < _SAMPLE_EVERY * _SAMPLE_LEAST:
        return None
    kept = np.arange(0, n, _SAMPLE_EVERY)
    return kept, _Mats(
        [X[kept] for X in Xs], parent=parent, marked=None if marked is None else (marked[0], marked[1][kept])
    )


@numba.njit(cache=True, nogil=True)
def _seed_scores(indptr, indices, data, col_ptr, col_rows, row_tokens, node_counts, total_tokens, col_weights):
    """For each candidate word f, how much better its sentences and the others predict their other candidate words
    with rates of their own than at the sense's rates: for each word more frequent on a side, its count there times
    the log of its rate there, left out of each sentence (count less one: a word in a single one of a side's sentences
    predicts none of them), over its rate in the sense, less half the log of the side's total (the price of one more
    rate, by the Bayesian information criterion). Leaving each sentence out keeps a few sentences that share their words
    from looking like a sense of their own. The cost of saying which side each sentence is on is left to the test:
    expectation-maximization moves the sentences from the seed's side to their sense's core, which changes it. Sentences
    x candidates in CSR (indptr, indices, data; with no data, each entry counts one, a count of n as n entries); the
    sentences of each candidate in col_ptr, col_rows."""
    m = len(node_counts)
    n = len(indptr) - 1
    scores = np.full(m, -np.inf)
    side_a = np.zeros(m, dtype=data.dtype)
    touched = np.empty(m, dtype=np.uint32)
    rates = node_counts / total_tokens
    counted = len(data) == 0
    for f in range(m):
        n_a = col_ptr[f + 1] - col_ptr[f]
        if n_a == 0 or n_a == n:
            continue
        n_touched = 0
        tokens_a = 0.0
        for k in range(col_ptr[f], col_ptr[f + 1]):
            i = col_rows[k]
            tokens_a += row_tokens[i]
            if counted:
                for j in range(indptr[i], indptr[i + 1]):
                    w = indices[j]
                    if side_a[w] == 0:
                        touched[n_touched] = w
                        n_touched += 1
                    side_a[w] += 1
            else:
                for j in range(indptr[i], indptr[i + 1]):
                    w = indices[j]
                    if side_a[w] == 0.0:
                        touched[n_touched] = w
                        n_touched += 1
                    side_a[w] += data[j]
        tokens_b = total_tokens - tokens_a
        price_a = 0.5 * np.log(tokens_a) if tokens_a > 0 else 0.0
        price_b = 0.5 * np.log(tokens_b) if tokens_b > 0 else 0.0
        # A word none of f's sentences have (count c = n_w on the other side) gains at most c (x - 1), which is
        # (n_w tokens_a - total) / tokens_b (log x <= x - 1): nothing below this count (margins far above rounding's)
        least = (total_tokens + price_b * tokens_b) / (tokens_a + 1e-12 * total_tokens) * (1.0 - 1e-9)
        gain = 0.0
        for w in range(m):
            if w == f or (side_a[w] == 0.0 and node_counts[w] < least):
                continue
            rate = rates[w]
            for side in range(2):
                c = side_a[w] if side == 0 else node_counts[w] - side_a[w]
                big = tokens_a if side == 0 else tokens_b
                if c <= 1.0 or big <= 0:
                    continue
                x = (c - 1.0) / (big * rate)
                price = price_a if side == 0 else price_b
                if x <= 1.0 or c * (x - 1.0) * (1.0 + 1e-12) <= price:
                    continue  # log x <= x - 1 (and the margin, for rounding): g would not be above 0
                g = c * np.log(x) - price
                if g > 0:
                    gain += g * col_weights[w]
        scores[f] = gain
        for t in range(n_touched):
            side_a[touched[t]] = 0
    return scores


@numba.njit(cache=True, nogil=True)
def _seed_counts(indptr, indices, data, column, row_len, col_len, row_tokens):
    """For one group of words: each sentence's number of entries among the candidates (column: the candidate each word
    is, -1 for none), a count of n as n entries, added to row_len; each candidate's number of sentences, to col_len;
    each sentence's count of all words, to row_tokens. Whether the counts are all whole numbers, none negative (else
    the rest is not to be used)."""
    for i in range(len(indptr) - 1):
        tokens = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            v = data[j]
            if not (v >= 0 and v == np.rint(v)):
                return False
            tokens += v
            f = column[indices[j]]
            if f >= 0:
                row_len[i] += np.int64(v)
                col_len[f] += 1
        row_tokens[i] += tokens
    return True


@numba.njit(cache=True, nogil=True)
def _seed_fill(indptr, indices, data, column, row_at, col_at, idx, col_rows, node_counts):
    """For one group of words, after _seed_counts: each sentence's candidates (a count of n, n times) into idx from
    row_at, each candidate's sentences, in order, into col_rows from col_at, and the candidates' counts."""
    for i in range(len(indptr) - 1):
        for j in range(indptr[i], indptr[i + 1]):
            f = column[indices[j]]
            if f >= 0:
                for _ in range(np.int64(data[j])):
                    idx[row_at[i]] = f
                    row_at[i] += 1
                col_rows[col_at[f]] = i
                col_at[f] += 1
                node_counts[f] += data[j]


def _seed_matrix(Xs, candidates):
    """The sentences x candidates matrix _seed_scores takes, the sentences of each candidate, the candidates' counts
    and the sentences' counts of all words; made from the groups' matrices at once when their counts are whole numbers
    (as the sentences' are), each count of n as n entries, which add up to the same in any order, with no counts to
    read, and the candidates' indices as small as their number allows."""
    n = Xs[0].shape[0]
    m = sum(len(c) for c in candidates)
    row_len = np.zeros(n, dtype=np.int64)
    col_len = np.zeros(m, dtype=np.int64)
    row_tokens = np.zeros(n)
    columns, offset, whole = [], 0, True
    for X, cand in zip(Xs, candidates):
        column = np.full(X.shape[1], -1, dtype=np.int64)
        column[cand] = np.arange(offset, offset + len(cand))
        offset += len(cand)
        columns.append(column)
        whole = whole and _seed_counts(X.indptr, X.indices, X.data, column, row_len, col_len, row_tokens)
    if not whole or float(row_tokens.sum()) >= 2.0**53:
        A = csr_matrix(hstack([X[:, cand] for X, cand in zip(Xs, candidates)]))
        A.sum_duplicates()
        columns = A.tocsc()
        node_counts = np.asarray(A.sum(axis=0)).ravel()
        row_tokens = sum(np.asarray(X.sum(axis=1)).ravel() for X in Xs)
        # Counts that are whole numbers add up the same in integers, faster (and indices unsigned: see _WIDTH)
        whole = bool(np.all(A.data == np.rint(A.data))) and A.nnz and float(np.abs(A.data).sum()) < 2.0**53
        data = A.data.astype(np.int64) if whole else A.data.astype(np.float64)
        rows = (A.indptr.astype(np.uint64), A.indices.astype(np.uint32), data)
        return rows, columns.indptr.astype(np.uint64), columns.indices.astype(np.uint32), node_counts, row_tokens
    indptr = np.zeros(n + 1, dtype=np.uint64)
    np.cumsum(row_len, out=indptr[1:])
    col_ptr = np.zeros(m + 1, dtype=np.uint64)
    np.cumsum(col_len, out=col_ptr[1:])
    idx = np.empty(int(indptr[-1]), dtype=np.uint16 if m <= 2**16 else np.uint32)
    col_rows = np.empty(int(col_ptr[-1]), dtype=np.uint32)
    node_counts = np.zeros(m)
    row_at, col_at = indptr[:-1].astype(np.int64), col_ptr[:-1].astype(np.int64)
    for X, column in zip(Xs, columns):
        _seed_fill(X.indptr, X.indices, X.data, column, row_at, col_at, idx, col_rows, node_counts)
    return (indptr, idx, np.empty(0, dtype=np.int64)), col_ptr, col_rows, node_counts, row_tokens


def seed_splits(Xs, collocates, k=_SEEDS_TRIED, weights=None):
    """The halves splits may start from: the sentences with a seed word and the others. Seeds are among the query's
    collocates (words markedly more frequent around it than in the corpus, in any group of words) that the sense's
    sentences have, those whose sentences differ most from the others in their other collocates first: senses are
    anchored by words, as in HyperLex. (A sense's own marker words, against its parent, would not do: those of the
    larger side of a split are only the words the smaller one lacks.) Returns up to k (r, (group, local index)), the
    most promising first."""
    candidates = [
        cols[np.bincount(X.indices[: X.nnz], minlength=X.shape[1])[cols] > 0] for X, cols in zip(Xs, collocates)
    ]
    if sum(len(c) for c in candidates) < 2:
        return []
    (indptr, indices, data), col_ptr, col_rows, node_counts, row_tokens = _seed_matrix(Xs, candidates)
    weights = weights or [None] * len(Xs)
    col_weights = np.concatenate(
        [np.ones(len(cand)) if w is None else w[cand] for cand, w in zip(candidates, weights)]
    ).astype(np.float64)
    scores = _seed_scores(
        indptr,
        indices,
        data,
        col_ptr,
        col_rows,
        row_tokens.astype(np.float64),
        node_counts,
        float(row_tokens.sum()),
        col_weights,
    )
    offsets = np.cumsum([0] + [len(c) for c in candidates])
    seeds = []
    for best in np.argsort(-scores, kind="stable")[:k]:
        if not np.isfinite(scores[best]) or scores[best] <= 0:
            break
        has = np.zeros(Xs[0].shape[0], dtype=bool)
        has[col_rows[col_ptr[best] : col_ptr[best + 1]]] = True
        group = int(np.searchsorted(offsets, best, side="right") - 1)
        seeds.append(
            (np.column_stack([has, ~has]).astype(np.float64), (group, int(candidates[group][best - offsets[group]])))
        )
    return seeds


def work_folds(groups):
    """The fold of each sentence: the group of works it is from, works going to the fold with the fewest sentences so
    far, the largest first. (folds, n_folds), or (None, 0) for sentences all from one work."""
    groups = np.asarray(groups)
    if groups.ndim == 1:  # the same as by rows, without sorting them as records
        _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    else:
        _, inverse, counts = np.unique(groups, axis=0, return_inverse=True, return_counts=True)
    inverse = np.ravel(inverse)
    n_groups = len(counts)
    if n_groups < 2:
        return None, 0
    n_folds = min(N_FOLDS, n_groups)
    load = np.zeros(n_folds, dtype=np.int64)
    fold_of = np.empty(n_groups, dtype=np.int64)
    for g in np.lexsort((np.arange(n_groups), -counts)):
        f = int(np.argmin(load))
        fold_of[g] = f
        load[f] += counts[g]
    return fold_of[inverse], n_folds


def _folds(groups):
    """The fold of each sentence, for the test of a split: whole works held out (work_folds); a single work's
    sentences, in turn."""
    folds, _ = work_folds(groups)
    return np.arange(len(groups)) % N_FOLDS if folds is None else folds


def _fold(Xs, unsplit_fit, f, prior=None):
    """A fold a split is tested on, the same for every split of a sense, from the sense's sentences (Xs, a _Mats packed
    by fold, which the fits of every fold share): its index, its held-out sentences (a mask), Xs, the unsplit sense
    fitted on the other sentences and the held-out sentences' log-probability under it, and the prior of the split's
    fits on the fold, or None. prior, (profiles, the rates of the sense's parent) or None: the unsplit sense has the
    prior of its corpus profile, as strong as its training sentences find it (_with_prior), and so do the split's."""
    test = Xs.blocks == f
    train = [np.asarray(X.T @ (~test).astype(np.float64)).ravel() for X in Xs]
    unsplit = unsplit_fit(train)
    fit_prior = None
    if prior is not None:
        profiles, against = prior
        local = Xs.words[0][_prior_places(profiles, Xs, unsplit)[1]]
        unsplit, strength, _ = _with_prior(profiles, local, unsplit, train[0], against)
        fit_prior = (profiles, strength)
    return f, test, Xs, unsplit, Xs.block_loglik(unsplit, f), fit_prior


def _fold_data(Xs, unsplit_fit, pool_map=map, prior=None):
    """The folds a split is tested on (_fold), the same for every split of a sense: the blocks of Xs."""
    return list(pool_map(lambda f: _fold(Xs, unsplit_fit, f, prior), range(len(Xs.bounds) - 1)))


def _fold_gain(fold, r):
    """The split started from the probabilities r, fitted on a fold's training sentences: each held-out sentence's
    gain over the unsplit sense, and the gains of those more likely in either sense; None for no evidence."""
    f, _, Xs, unsplit, unsplit_ll, prior = fold
    ll = _em(Xs, unsplit, r, every_marker=False, left_out=f, prior=prior, held_only=True)  # as _mixture_loglik makes it
    if ll is None:  # without these sentences, nothing can be told apart: no better
        return None
    gain = _logsumexp2(ll) - unsplit_ll
    if not np.all(np.isfinite(gain)):  # a sentence impossible under either: no evidence for the split
        return None
    return gain, np.bincount(np.argmax(ll, axis=1), weights=gain, minlength=2)


def _test_result(folds, results, n):
    sentence_gains = np.zeros(n)
    sides = np.zeros(2)
    for fold, result in zip(folds, results):
        if result is None:
            continue
        gain, side_gains = result
        sentence_gains[fold[1]] = gain
        sides += side_gains
    mean = float(sentence_gains.mean())
    se = float(sentence_gains.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    return {
        "tested": True,
        "gain": float(sentence_gains.sum()),
        "gains": [float(g) for g in sides],
        "bits_per_sentence": float(mean / np.log(2)),
        "z": float(mean / se) if se > 0 else 0.0,
        "sentences_won": int((sentence_gains > 0).sum()),
        "sentences": int(n),
    }


def test_split(Xs, unsplit_fit, r, groups, folds=None, pool_map=map, prior=None):
    """Whether the split predicts the sentences of the works it was not fitted on better than its unsplit sense: in
    each of N_FOLDS folds of the works, both are fitted on the other folds and scored on its sentences, the split paying
    for saying which sense each is in. Holding out whole works keeps the sentences of a passage, which share their
    words, from predicting one another. Sentences are the units of evidence: each gains or loses, and the mean of their
    gains, in nats, is compared with its standard error. prior: see _fold."""
    if folds is None:
        folds = _fold_data(_Mats(Xs, _folds(groups)), unsplit_fit, pool_map, prior)
    return _test_result(folds, list(pool_map(lambda fold: _fold_gain(fold, r), folds)), Xs[0].shape[0])


def induce_senses(channels, groups, finest=FINEST, unseeded=None, sample=None, cooccurrence=None):
    """Grow the tree of senses of the sentences: channels are (X, corpus probabilities, corpus rest[, weights]) for each
    group of words, X a sentences x words matrix of counts, weights how much each word's evidence counts; groups is
    the work of each sentence; unseeded, for each group of words, None or a mask of its words that seed no split
    (they still mark senses). Returns the root Node; split senses have children, the others are the senses found.
    Each split has its importance: how much it improves the prediction of the words around the query, per use of the
    word (its sense's share of the sentences times its gain per sentence there). Splits less important than finest
    times the first one are not looked for. The splits are looked for and tested on the sentences of sample (their
    indices; by default growth_sample's, all of them for a query of no more than _GROWN): see _try_split. Given the
    corpus's cooccurrence, the senses' rates for the collocates that may seed a split, of the sentence's words, have
    the prior of their corpus profiles (see _with_prior and _em): cooccurrence(columns), for some of the sentence's
    words (columns of the first group's matrix), is the number of the corpus's sentences outside the query's with each
    pair of them (the diagonal: with each), a square array (corpus_cooccurrence's)."""
    Xall = [csr_matrix(c[0]) for c in channels]
    n = Xall[0].shape[0]
    corpus = [Dist(np.asarray(c[1], dtype=np.float64), float(c[2])) for c in channels]
    counts = [np.asarray(X.sum(axis=0)).ravel() for X in Xall]
    discounts = [discount(c) for c in counts]
    collocates = [markedly_more_frequent(c, dist.p)[0] for c, dist in zip(counts, corpus)]
    # Only the query's collocates (words markedly more frequent around it than in the corpus) may mark a sense: words
    # used around it at no particular rate (je, vous, les) tell no sense of it from another, whatever else they tell
    masks = [np.isin(np.arange(len(c)), cols) for c, cols in zip(counts, collocates)]
    seeds = [cols if u is None else cols[~u[cols]] for cols, u in zip(collocates, unseeded or [None] * len(counts))]
    weights = [c[3] if len(c) > 3 else None for c in channels]

    def root_fit(train_counts):  # the whole query's sentences: calibrated, from the corpus
        dists = [fit_dist(c, dist, d) for c, dist, d in zip(train_counts, corpus, discounts)]
        for dist, mask, w, d in zip(dists, masks, weights, discounts):
            dist.markers = mask
            dist.weights = w
            dist.discount = d
        return dists

    root = Node(np.arange(n), root_fit(counts), counts, corpus)
    profiles = None
    if cooccurrence is not None and len(seeds[0]):
        profiles = _Profiles(seeds[0], cooccurrence(seeds[0]), Xall[0].shape[1])
        local = profiles.slot[np.flatnonzero(counts[0] > 0)] >= 0
        local = np.flatnonzero(counts[0] > 0)[local]
        root.dists, strength, tokens = _with_prior(profiles, local, root.dists, counts[0], corpus[0])
        root.prior = {"strength": strength, "tokens": tokens}
    if sample is None:
        sample = growth_sample(n)
    grown = None
    if sample is not None:
        grown = np.zeros(n, dtype=np.bool_)
        grown[sample] = True

    floor = {"importance": 0.0}

    def grow(node, submit):
        if node is root:
            unsplit_fit = root_fit
        else:
            parents = node.parent_dists
            unsplit_fit = lambda train_counts, parents=parents: [
                fit_sparse(c, d) for c, d in zip(train_counts, parents)
            ]
        _try_split(node, Xall, n, unsplit_fit, floor, groups, seeds, grown, submit, marked[0], profiles)
        return node

    marked = [None]

    def grow_with(submit, senses=None):
        # The markers' entries of all the sentences, apart, for each sense's packing to take its own from: made while
        # the first split is looked for (on its sentences, which sort out their own rather than wait)
        marked[0] = submit(_Marked, Xall, root.dists)
        grow(root, submit)
        if root.children:
            floor["importance"] = finest * root.test["importance"]
        if senses is None:
            queue = list(root.children)
            while queue:
                queue.extend(grow(queue.pop(0), submit).children)
            return
        pending = {senses.submit(grow, child, submit) for child in root.children}
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending |= {senses.submit(grow, child, submit) for child in future.result().children}

    # The seeds of a split and the folds of each are fitted at once, and the senses below the first split grow at once
    # too (they are independent of one another once the first sets the floor): the compiled EM and numpy release the
    # GIL. As many threads work as numba may use (NUMBA_NUM_THREADS, which the web server caps for each of its workers,
    # as for the rest of the app): those that grow the senses only hand their work to them.
    workers = numba.config.NUMBA_NUM_THREADS
    if workers < 2:
        grow_with(_now)
        return root
    with ThreadPoolExecutor(max_workers=workers) as fits, ThreadPoolExecutor(max_workers=workers) as senses:
        grow_with(fits.submit, senses)
    return root


def growth_sample(n):
    """The sentences splits are looked for and tested on, of a query's n: all of them (None), or for more than _GROWN,
    one at random in each of _GROWN stretches of them of equal length, in their order (the corpus's), so that each work
    keeps its share, and each sense its sentences in proportion (a sample of whole works would leave out the senses of
    a few), and no sentence is left out for its place in a pattern of the text, as every k-th would be. The same sample
    every time: the same senses."""
    if n <= _GROWN:
        return None
    starts = (np.arange(_GROWN + 1, dtype=np.int64) * n) // _GROWN
    offsets = np.random.default_rng(0).random(_GROWN) * np.diff(starts)
    return starts[:-1] + offsets.astype(np.int64)


def elbow(relative):
    """How many distinctions to show at first, of their importances relative to the main one: down to the elbow of
    their curve, from the most important down, the distinction farthest from the straight line from the main one to
    the faintest found (the knee of a scree plot, as it tells a number of clusters: Thorndike 1953, Satopää and others
    2011). It reads the shape of the whole curve, where a single drop from one to the next would decide by chance:
    importances mostly decline smoothly, as they follow the size of the senses split."""
    y = np.sort(np.asarray(relative, dtype=float))[::-1]
    if len(y) <= 2:
        return len(y)
    x = np.arange(len(y), dtype=float)
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    return int(np.argmax(np.abs(dy * x - dx * y + x[-1] * y[0] - y[-1] * x[0]))) + 1


def revealed_by(nodes, detail):
    """The splits shown at a detail, of a report's nodes: those at least that important relative to the main one, and
    those they are within. Returns their ids."""
    out = set()
    for node in nodes:
        if node["children"] and node["split"]["relative"] >= detail:
            n = node
            while n is not None and n["id"] not in out:
                out.add(n["id"])
                n = None if n["parent"] is None else nodes[n["parent"]]
    return out


def leaves_at(nodes, detail):
    """The nodes the splits shown at a detail end in, of a report's nodes, as the tree has them."""
    revealed = revealed_by(nodes, detail)
    out = []

    def walk(node):
        if node["id"] in revealed:
            for child in node["children"]:
                walk(nodes[child])
        else:
            out.append(node)

    if nodes:
        walk(nodes[0])
    return out


def _has_own(node):
    """Whether a sense has something to show: a word markedly more frequent in it, in any group of words."""
    return bool(node["words"] or node["before"] or node["after"] or node["hidden"])


def sense_numbers(nodes, marked=_has_own):
    """The number of each sense of a report's nodes (as their ids), None for the rest of the sentences, which no sense
    takes; and the node each number starts at. marked tells whether a node has something of its own to show (the
    report: a word outside the collocation filter, which alone makes no sense). Each split carves one of its halves out
    of the other, which goes on as their parent: the half with nothing of its own to show, if only one has, else the
    larger. The carved half takes
    the next number: at the top (1, 2, …) out of the rest, else its parent's followed by the next of its own (2.1,
    2.2, …). Out of the rest, a half with something to show is a sense either way, and one with nothing stays the rest.
    The splits give their numbers as the detail reveals them, from the most important down (those they are within
    first), so that a sense keeps its number at any detail."""
    number, start, counts = {}, {}, {}
    if not nodes:
        return number, start
    number[nodes[0]["id"]] = None
    order = sorted((n for n in nodes if n["children"]), key=lambda n: (-n["split"]["relative"], n["id"]))
    done = set()
    for split in order:
        chain = []
        n = split
        while n is not None and n["id"] not in done:
            chain.append(n)
            n = None if n["parent"] is None else nodes[n["parent"]]
        for parent in reversed(chain):
            done.add(parent["id"])
            halves = [nodes[c] for c in parent["children"]]  # the larger first
            bare = [h for h in halves if not marked(h)]
            going_on = bare[0] if len(bare) == 1 else halves[0]
            mine = number[parent["id"]]
            for half in halves:
                if (half is going_on) if mine is not None else not marked(half):
                    number[half["id"]] = mine
                    continue
                counts[mine] = counts.get(mine, 0) + 1
                number[half["id"]] = str(counts[mine]) if mine is None else f"{mine}.{counts[mine]}"
                start[number[half["id"]]] = half["id"]
    return number, start


def default_detail(nodes, number):
    """The detail shown at first, of a report's nodes and their sense numbers: down to the elbow, and finer, a split at
    a time from the most important down, until the senses shown cover MOST of the sentences, the rest no longer most;
    when no detail does, the elbow."""
    relative = sorted((n["split"]["relative"] for n in nodes if n["children"]), reverse=True)
    k = elbow(relative)
    if not k:
        return 1.0
    for detail in relative[k - 1 :]:
        if sum(n["share"] for n in leaves_at(nodes, detail) if number[n["id"]] is not None) > MOST:
            return detail
    return relative[k - 1]


def marks(node, against, common=None, listed=None):
    """Whether a sense has something of its own to show: a word markedly more frequent in it than against (the dists
    of the sense it is told from) in any group of words, outside common (for each group, None or a mask of the words
    that alone make no sense). listed: for each group, those words as _described has them, if made already."""
    for ci, (c, dist) in enumerate(zip(node.counts, node.dists)):
        mask = None if common is None else common[ci]
        listed_c = _described(c, dist, against[ci].p)[0] if listed is None else listed[ci]
        if len(listed_c) and (mask is None or not mask[listed_c].all()):
            return True
    return False


def _race(racing, fits, folds, n):
    """The candidates out of the race after the folds given (Maron and Moore 1994: models compared by
    cross-validation, those clearly worse no longer tested): those whose held-out sentences so far gain less than the
    best candidate's, by more than EVIDENCE standard errors of their differences (each sentence with the gain of both).
    Clearly worse on the works held out so far, a candidate would not be the split kept: testing it on the other folds
    would only take time. The race is first decided on two folds: the works of one alone, whose sentences share their
    words, can favor a candidate that those of the others do not."""
    held = np.zeros(n, dtype=bool)
    for fold in folds:
        held |= fold[1]
    gains = {}
    for s in racing:
        g = np.zeros(n)
        for f, fold in enumerate(folds):
            result = fits[(s, f)].result()
            if result is not None:
                g[fold[1]] = result[0]
        gains[s] = g[held]
    best = max(racing, key=lambda s: (float(gains[s].sum()), -s))
    behind = set()
    for s in racing:
        if s == best:
            continue
        d = gains[best] - gains[s]
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        if se > 0 and float(d.mean()) / se > EVIDENCE:
            behind.add(s)
    return behind


def _try_split(
    node, Xall, n_all, unsplit_fit, floor, groups, collocates, grown=None, submit=_now, marked=None, profiles=None
):
    """Split a sense, if a split passes: looked for and tested on its sentences, or on those of them in the sample
    (grown, a mask of all the sentences, or None for all), then fitted again on all of them (_fit_all). marked, a
    future of the markers' entries of all the sentences (_Marked), which the packings take theirs from once made.
    profiles (_Profiles) or None: the senses of its splits have the prior of their corpus profiles, as strong as the
    sense's sentences find the corpus (_with_prior: in each fold, its training sentences)."""
    units = node.units
    tried = units if grown is None else units[grown[units]]
    if len(tried) < 2:
        node.reason = "inseparable"
        return
    whole = marked.result() if marked is not None and marked.done() else None
    of = None if whole is None else (whole, tried)

    # All of it on the pool's threads (submit), which alone do the work, each step as soon as what it needs is done,
    # so that no thread waits for the slowest of a step: the sentences' matrices packed for the EM (by the folds the
    # splits are tested on, so that the fits of every fold share them), the seeds and the sample of the searches' first
    # pass at once, the folds of the test once the packing is done (the folds, the same for every split, are wasted
    # only when no seed splits the sense), each seed's search once the first three are made, each candidate's fit on a
    # fold once its search and the fold are done and the race needs it (see _race: fitting ahead of the race, so that
    # no thread waits, takes longer, on candidates the race then drops). Each result is kept in its place, and the race
    # decides on whole folds: the results do not depend on the order the threads finish in.
    def sentences():
        return [X[tried] for X in Xall], _folds(groups[tried])

    Xs, fold_of = submit(sentences).result()
    packed = submit(_Mats, Xs, fold_of, node.dists, of)
    seeding = submit(seed_splits, Xs, collocates, _SEEDS_TRIED, [d.weights for d in node.dists])
    # The searches' first pass on a sample of the sense's sentences, unless they are a sample already (grown): sampled
    # again, it took no less time, and its splits were no better
    sampling = submit(search_sample, Xs, node.dists, of) if len(tried) == len(units) else None
    Xs = packed.result()
    n_folds = len(Xs.bounds) - 1
    fold_prior = None if profiles is None else (profiles, node.parent_dists[0])
    folds = [submit(_fold, Xs, unsplit_fit, f, fold_prior) for f in range(n_folds)]
    prior = None
    if profiles is not None:  # the strength of the corpus against the sense's sentences, for the searches
        c0 = np.asarray(Xs[0].sum(axis=0)).ravel()
        local = Xs.words[0][_prior_places(profiles, Xs, node.dists)[1]]
        _, strength, tokens = _with_prior(profiles, local, node.dists, c0, node.parent_dists[0])
        prior = (profiles, strength)
        node.prior = {"strength": strength, "tokens": tokens}
    seeds = seeding.result()
    sample = None if sampling is None else sampling.result()
    firsts = [[] for _ in seeds]  # each search's first pass (its fit), whose senses _fit_all starts from
    searches = [
        submit(fit_split, Xs, node.dists, r, True, sample, first, prior) for (r, _), first in zip(seeds, firsts)
    ]
    found = {}  # by the place of their seed: the candidate, or None
    fits = {}  # (candidate, fold): its fit on the fold
    out = set()  # the candidates out of the race
    raced = 0  # the last fold the race is decided on (never on the first alone: see _race)
    while True:
        for s, search in enumerate(searches):
            if s in found or not search.done():
                continue
            fitted = search.result()
            side = None if fitted is None else fitted[2][:, 0] >= fitted[2][:, 1]
            # One sense in every sentence, more or less: no halves
            found[s] = None if side is None or side.all() or not side.any() else (seeds[s][1], fitted, side, firsts[s])
        known = len(found) == len(searches)
        racing = [s for s in sorted(found) if found[s] is not None and s not in out]
        if known and not racing:
            for fold in folds:  # no halves: no test
                fold.cancel()
            break
        # The race decided on the next fold once every racing candidate's fits on it and on the folds before are
        # done: a fold's data may be made after the next one's, and its fits submitted after theirs
        while (
            known
            and len(racing) > 1
            and raced + 1 < n_folds - 1
            and all((s, f) in fits and fits[(s, f)].done() for s in racing for f in range(raced + 2))
        ):
            raced += 1
            behind = _race(racing, fits, [fold.result() for fold in folds[: raced + 1]], len(tried))
            for s in behind:
                for f in range(n_folds):
                    if (s, f) in fits:
                        fits[(s, f)].cancel()
            out |= behind
            racing = [s for s in racing if s not in behind]
        over = known and (len(racing) < 2 or raced + 1 >= n_folds - 1)
        for s in racing:
            for f in range(n_folds if over else raced + 2):
                if (s, f) not in fits and folds[f].done():
                    fits[(s, f)] = submit(_fold_gain, folds[f].result(), found[s][1][2])
        pending = {search for search in searches if not search.done()}
        pending |= {fold for fold in folds if not fold.done()}
        pending |= {fits[key] for key in fits if key[0] in racing and not fits[key].done()}
        if known and not pending and all((s, f) in fits for s in racing for f in range(n_folds)):
            break
        wait(pending, return_when=FIRST_COMPLETED)
    candidates = [found[s] for s in racing]
    if candidates:
        folds = [fold.result() for fold in folds]
        results = [fits[(s, f)].result() for s in racing for f in range(n_folds)]
    best = None
    for c, (seed, fitted, side, first) in enumerate(candidates):
        test = _test_result(folds, results[c * len(folds) : (c + 1) * len(folds)], len(tried))
        test["seed"] = seed
        test["importance"] = float(len(units) / n_all * max(0.0, test["bits_per_sentence"]))
        if node.prior is not None:
            test["prior"] = {
                "strength": round(node.prior["strength"], 2),
                "tokens": round(node.prior["tokens"], 2),
                "folds": [None if fold[5] is None else round(fold[5][1], 2) for fold in folds],
            }
        passes = test["z"] > EVIDENCE and test["importance"] >= floor["importance"]
        if best is None or (passes, test["importance"]) > (best[0], best[1]["importance"]):
            best = (passes, test, fitted, side, first)
    if best is None:
        node.reason = "inseparable"
        return
    passes, node.test, fitted, side, first = best
    if not passes:
        node.reason = "no_gain" if node.test["z"] <= EVIDENCE else "faint"
        return
    if len(tried) < len(units):
        whole = None if marked is None else marked.result()
        pi, dists, side, counts = submit(
            _fit_all, Xall, units, node.dists, first[0][:2], fitted, whole, submit, prior
        ).result()
    else:
        pi, dists, _, counts = fitted  # the senses' rates and counts of the split kept, made now (_Fit)
    node.model = (pi, dists)
    for k, members in ((0, side), (1, ~side)):
        child = Node(units[members], [d[k] for d in dists], [c[:, k] for c in counts], node.dists)
        child.k = k
        node.children.append(child)
    node.children.sort(key=lambda c: (-len(c.units), c.k))


def _fit_all(Xall, units, parent, first, fitted, marked=None, submit=_now, prior=None):
    """A split found on a sample of its sense's sentences, fitted again on all of them (units): its search's second
    pass, on all of them, from the senses of its first (pi, dists), as these share them out. The second pass on the
    sample alone chose the words that mark either sense from the sample's counts: fewer of them than all the sentences
    have, which would hold the split to a fit less good on them. Should that fail (no halves), the split as fitted
    (fitted: pi, dists, ...), its senses' rates fitted again on all the sentences, as they share them out. Returns (pi,
    dists, side: the sentences more likely in the first sense, counts). marked: the markers' entries of all the
    sentences (_Marked), or None. The sentences are in _PARTS parts of about as many entries, each packed and fitted
    at once on the pool's threads (submit), so that the refit of the first split, which nothing else can run beside,
    keeps both busy. prior: see _em (the fallback's rates have none)."""
    entries = np.cumsum(sum(np.diff(X.indptr)[units] for X in Xall))
    cuts = np.searchsorted(entries, np.arange(1, _PARTS) * entries[-1] / _PARTS) if len(units) else []
    pieces = [piece for piece in np.split(units, cuts) if len(piece)]

    def rows(piece):  # a part's matrices, and the words each group of them has
        Xs = [X[piece] for X in Xall]
        return Xs, [np.bincount(X.indices[: X.nnz], minlength=X.shape[1]) > 0 for X in Xs]

    made = _at_once(submit, rows, pieces)
    words = [np.flatnonzero(np.logical_or.reduce([had[c] for _, had in made])) for c in range(len(Xall))]

    def packed(p):  # a part packed, and its sentences' probabilities under the first pass's senses
        Xs = _Mats(made[p][0], marked=None if marked is None else (marked, pieces[p]), words=words, by_words=False)
        Xs.unmarked_loglik(parent)
        ll = _mixture_loglik(Xs, *first)
        return Xs, np.exp(ll - _logsumexp2(ll)[:, None])

    made = _at_once(submit, packed, range(len(pieces)))
    again = _em(
        _Parts(Xs for Xs, _ in made), parent, np.concatenate([r for _, r in made]), False, submit=submit, prior=prior
    )
    if again is not None:
        pi, dists, r, counts = again
        side = r[:, 0] >= r[:, 1]
        if side.any() and not side.all():
            return pi, dists, side, counts
    Xs = [X[units] for X in Xall]
    ll = _mixture_loglik(Xs, *fitted[:2])
    r = np.exp(ll - _logsumexp2(ll)[:, None])
    counts = [np.asarray(X.T @ r) for X in Xs]
    dists = [[fit_sparse(c[:, k], d, _split_words(c, d, False)) for k in range(2)] for c, d in zip(counts, parent)]
    pi = _weights(r) / len(units)
    ll = _mixture_loglik(Xs, pi, dists)
    return pi, dists, ll[:, 0] >= ll[:, 1], counts


@numba.njit(cache=True, nogil=True)
def _add_rows(indptr, indices, data, ones, evidence, ll):
    """ll += X @ evidence (n x 2), X in CSR: each row's sums of its entries in order, from 0, as scipy's product makes
    them (with ones, the counts are not read: one times a number is that number)."""
    for i in range(len(indptr) - 1):
        s0 = 0.0
        s1 = 0.0
        if ones:
            for j in range(indptr[i], indptr[i + 1]):
                w = np.uint64(indices[j])
                s0 += evidence[w, 0]
                s1 += evidence[w, 1]
        else:
            for j in range(indptr[i], indptr[i + 1]):
                w = np.uint64(indices[j])
                s0 += data[j] * evidence[w, 0]
                s1 += data[j] * evidence[w, 1]
        ll[i, 0] += s0
        ll[i, 1] += s1


def node_posteriors(root, Xall, submit=_now):
    """Every sense of the tree, in order (each before those it splits into), given its id and parent; and for every
    sentence the log-probability of each sense (n x senses), and its margin: the smallest log-likelihood ratio, along
    the way to the sense, of the sense's side over the other (positive when the sentence is on its side of every
    split, infinite for the whole query). Each sense's are stored together (the arrays are senses x n, turned). The
    sentences' log-likelihoods under each split are made at once on the pool's threads (submit)."""
    n = Xall[0].shape[0]
    nodes = []
    ones = [bool(np.all(X.data[: X.nnz] == 1)) for X in Xall]

    def size(node):
        return 1 + sum(size(child) for child in node.children)

    def splits(node):
        return [node] + [s for child in node.children for s in splits(child)] if node.children else []

    def loglik(node):
        pi, dists = node.model
        ll = np.tile(np.log(pi), (n, 1))  # _mixture_loglik, compiled
        for X, one, ds in zip(Xall, ones, dists):
            _add_rows(X.indptr, X.indices, X.data, one, np.column_stack([_evidence(ds[0]), _evidence(ds[1])]), ll)
        return ll

    split = splits(root)
    lls = dict(zip(map(id, split), _at_once(submit, loglik, split)))
    logp, margin = np.empty((size(root), n)), np.empty((size(root), n))

    def walk(node, lp, mg, parent):
        node.id = len(nodes)
        node.parent = parent
        nodes.append(node)
        logp[node.id] = lp
        margin[node.id] = mg
        if not node.children:
            return
        ll = lls.pop(id(node))
        cond = ll - _logsumexp2(ll)[:, None]
        for child in node.children:
            walk(child, lp + cond[:, child.k], np.minimum(mg, ll[:, child.k] - ll[:, 1 - child.k]), node)

    walk(root, np.zeros(n), np.full(n, np.inf), None)
    return nodes, logp.T, margin.T


def _path(root, leaf):
    """The (node, component) splits on the way from the root to a leaf."""
    found = []

    def walk(node, trail):
        if node is leaf:
            found.extend(trail)
            return True
        return any(walk(child, trail + [(node, child.k)]) for child in node.children)

    walk(root, [])
    return found


# ----------------------------- Smoothing over time ------------------------------


@numba.njit(cache=True)
def _ldl(w, lam):
    """LDL' factors of W + lam D'D, D the second differences, as its diagonal d and the two subdiagonals of L."""
    n = len(w)
    a0 = np.empty(n)
    a1 = np.zeros(n)
    a2 = np.zeros(n)
    for i in range(n):
        s = 0.0
        if i <= n - 3:
            s += 1.0
        if 1 <= i <= n - 2:
            s += 4.0
        if 2 <= i <= n - 1:
            s += 1.0
        a0[i] = w[i] + lam * s
        if i <= n - 2:
            t = 0.0
            if i <= n - 3:
                t -= 2.0
            if 1 <= i <= n - 2:
                t -= 2.0
            a1[i] = lam * t
        if i <= n - 3:
            a2[i] = lam
    d = np.empty(n)
    l1 = np.zeros(n)
    l2 = np.zeros(n)
    for i in range(n):
        di = a0[i]
        if i >= 1:
            di -= l1[i - 1] * l1[i - 1] * d[i - 1]
        if i >= 2:
            di -= l2[i - 2] * l2[i - 2] * d[i - 2]
        d[i] = di
        if i <= n - 2:
            v = a1[i]
            if i >= 1:
                v -= l2[i - 1] * d[i - 1] * l1[i - 1]
            l1[i] = v / di
        if i <= n - 3:
            l2[i] = a2[i] / di
    return d, l1, l2


@numba.njit(cache=True)
def _ldl_solve(d, l1, l2, b):
    n = len(d)
    y = np.empty(n)
    for i in range(n):
        v = b[i]
        if i >= 1:
            v -= l1[i - 1] * y[i - 1]
        if i >= 2:
            v -= l2[i - 2] * y[i - 2]
        y[i] = v
    f = np.empty(n)
    for i in range(n - 1, -1, -1):
        v = y[i] / d[i]
        if i <= n - 2:
            v -= l1[i] * f[i + 1]
        if i <= n - 3:
            v -= l2[i] * f[i + 2]
        f[i] = v
    return f


@numba.njit(cache=True)
def _inverse_diagonal(d, l1, l2):
    """The diagonal of the inverse of LDL', by Takahashi's recursion over its band."""
    n = len(d)
    z0 = np.zeros(n + 2)
    z1 = np.zeros(n + 2)
    z2 = np.zeros(n + 2)
    for i in range(n - 1, -1, -1):
        a = l1[i] if i <= n - 2 else 0.0
        b = l2[i] if i <= n - 3 else 0.0
        z2[i] = -(a * z1[i + 1] + b * z0[i + 2])
        z1[i] = -(a * z0[i + 1] + b * z1[i + 1])
        z0[i] = 1.0 / d[i] - (a * z1[i] + b * z2[i])
    return z0[:n]


@numba.njit(cache=True)
def _smooth_resamples(w, z, lam):
    """Smooth each resampling b of each series k: z is resamples x series x years, w resamples x years."""
    n_b, n_k, n = z.shape
    out = np.full(z.shape, np.nan)
    for b in range(n_b):
        if np.sum(w[b] > 0) < 2:  # all its works from one year: no course over time to draw
            continue
        d, l1, l2 = _ldl(w[b], lam)
        for k in range(n_k):
            out[b, k] = _ldl_solve(d, l1, l2, w[b] * z[b, k])
    return out


def whittaker(w, z, lam):
    """Whittaker-Henderson smoothing of the series z (series x years) with weights w (years): the curves minimizing
    sum w (z - f)^2 + lam sum (second differences of f)^2. Linear functions are left as they are, so shares that sum to
    one still do."""
    d, l1, l2 = _ldl(w, lam)
    return np.array([_ldl_solve(d, l1, l2, w * series) for series in z])


def choose_smoothness(w, z):
    """The smoothness generalized cross-validation prefers for the series z together (they share the weights w)."""
    observed = w > 0
    m = int(observed.sum())
    best, best_score = _LAMBDAS[0], np.inf
    for lam in _LAMBDAS:
        d, l1, l2 = _ldl(w, lam)
        trace = float(np.sum(w * _inverse_diagonal(d, l1, l2)))
        if trace >= m:
            continue
        rss = 0.0
        for series in z:
            f = _ldl_solve(d, l1, l2, w * series)
            rss += float(np.sum(w[observed] * (series[observed] - f[observed]) ** 2))
        score = rss / m / (1.0 - trace / m) ** 2
        if score < best_score:
            best, best_score = lam, score
    return float(best)


def _nanpercentile(smoothed, q):
    """np.nanpercentile(smoothed, q, axis=0) of smoothed resamples, those not drawn (a resample's years too few) all
    undefined: the percentiles of those drawn, at once (nanpercentile takes each place's series in turn)."""
    undefined = np.isnan(smoothed).reshape(len(smoothed), -1)
    drawn = ~undefined.any(axis=1)
    if not drawn.any() or not undefined[~drawn].all():  # else as nanpercentile has it
        return np.nanpercentile(smoothed, q, axis=0)
    return np.percentile(smoothed[drawn], q, axis=0)


@numba.njit(cache=True, nogil=True)
def _work_sums(work_of, P, n_works):
    """np.add.at(sums, work_of, P) into works x senses zeros: each work's sums of its sentences' probabilities, added
    in the order of the sentences."""
    sums = np.zeros((n_works, P.shape[1]))
    for i in range(P.shape[0]):
        for k in range(P.shape[1]):
            sums[work_of[i], k] += P[i, k]
    return sums


@numba.njit(cache=True, nogil=True)
def _year_totals(work_year, draws, work_n, work_s, n_years, finite):
    """For each draw of the works (draws x works: how many times each is drawn), each year's sentences, each sense's
    probabilities and works, added in the order of the works as np.bincount adds them (a work drawn no times, when the
    sums are finite, adds zero, which changes no sum)."""
    n_draws, n_works = draws.shape
    n_k = work_s.shape[1]
    n = np.zeros((n_draws, n_years))
    y = np.zeros((n_draws, n_k, n_years))
    works = np.zeros((n_draws, n_years))
    for b in range(n_draws):
        for w in range(n_works):
            m = draws[b, w]
            if m == 0 and finite:
                continue
            t = work_year[w]
            n[b, t] += m * work_n[w]
            for k in range(n_k):
                y[b, k, t] += m * work_s[w, k]
            works[b, t] += m
    return n, y, works


def sense_curves(posteriors, years, works, year_words=None):
    """The course over time of each sense: its share of the year's sentences and, with the words of each year
    (year_words), its uses per million words, smoothed, with bands from resampling the works. Years weigh by their
    number of works, the units of evidence: the sentences of a work are not independent of one another, so weighing
    by sentences would take the swings of a few works for a course to follow. None without at least two years with
    sentences and three years in all."""
    dated = years > 0
    if len(np.unique(years[dated])) < 2:
        return None
    start, end = int(years[dated].min()), int(years[dated].max())
    n_years = end - start + 1
    if n_years < 3:
        return None
    P = posteriors[dated]
    year_index = years[dated] - start
    work_ids, work_of = np.unique(works[dated], return_inverse=True)
    n_works = len(work_ids)
    # Each work has one year: its sentences, and their probabilities of each sense, summed by work
    work_year = np.zeros(n_works, dtype=np.int64)
    work_year[work_of] = year_index
    work_n = np.bincount(work_of, minlength=n_works).astype(np.float64)
    work_s = _work_sums(work_of, P, n_works)
    rng = np.random.default_rng(0)  # fixed: the same bands every time
    resamples = rng.multinomial(n_works, np.full(n_works, 1.0 / n_works), size=N_RESAMPLES).astype(np.float64)
    finite = bool(np.isfinite(work_s).all())
    all_n, all_y, all_works = _year_totals(
        work_year, np.vstack([np.ones(n_works), resamples]), work_n, work_s, n_years, finite
    )
    n, y, works_in = all_n[0], all_y[0], all_works[0]
    boot_n, boot_y, boot_works = all_n[1:], all_y[1:], all_works[1:]

    def curves(z, boot_z, weight, boot_weight, digits):
        scale = weight[weight > 0].mean()
        w = weight / scale
        lam = choose_smoothness(w, z)
        value = whittaker(w, z, lam)
        smoothed = _smooth_resamples(boot_weight / scale, boot_z, lam)
        low, high = _nanpercentile(smoothed, BAND)
        return {"smoothness": lam, "value": digits(value), "low": digits(low), "high": digits(high)}

    def decimals(a):  # shares: to a hundredth of a percent, more than a chart shows
        return np.round(np.nan_to_num(a), 4).tolist()

    def significant(a):  # rates per million: four significant digits
        return [[float(f"{v:.4g}") for v in row] for row in np.nan_to_num(a)]

    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(n > 0, y / n, 0.0)
        boot_share = np.where(boot_n[:, None, :] > 0, boot_y / boot_n[:, None, :], 0.0)
    result = {
        "start": start,
        "end": end,
        "sentences": n.astype(np.int64).tolist(),
        "works": works_in.astype(np.int64).tolist(),
        "share": curves(share, boot_share, works_in, boot_works, decimals),
        "rate": None,
    }
    if year_words:
        words = np.array([float(year_words.get(start + t, 0)) for t in range(n_years)])
        if (words > 0).sum() >= 2:
            per_million = np.where(words > 0, 1e6 / np.where(words > 0, words, 1.0), 0.0)
            rate = y * per_million
            boot_rate = boot_y * per_million  # the words of each year are the corpus's, not resampled
            has_words = words > 0
            result["rate"] = curves(rate, boot_rate, works_in * has_words, boot_works * has_words, significant)
    return result


# ----------------------------- The sentences of a query ------------------------------


@numba.njit(cache=True, nogil=True)
def _sentence_words(
    unit_bounds,
    hit_sent,
    hit_start,
    hit_end,
    sent_offsets,
    count_ids,
    excluded,
    filtered,
    prop_columns,
    prop_allowed,
    max_dist,
    n_vocab,
    bag,
    left,
    right,
    bag_at,
    side_at,
):
    """The words of each sentence (once each; within max_dist of a hit when max_dist >= 0; with a kept value of each
    word property of prop_columns, see word_properties.property_masks), and the words just before and just after each
    hit, into the flat arrays bag, left and right from bag_at and side_at (room enough: the words of the sentences,
    and their hits), and their bounds by sentence there. The query's own words are left out, and the filtered ones."""
    n_units = len(unit_bounds) - 1
    bag_ptr = np.zeros(n_units + 1, dtype=np.int64)
    left_ptr = np.zeros(n_units + 1, dtype=np.int64)
    right_ptr = np.zeros(n_units + 1, dtype=np.int64)
    seen = np.zeros(n_vocab, dtype=np.bool_)
    hashes = np.empty(n_units, dtype=np.uint64)
    nb = bag_at
    nl = side_at
    nr = side_at
    bag_ptr[0] = nb
    left_ptr[0] = nl
    right_ptr[0] = nr
    for u in range(n_units):
        h0 = unit_bounds[u]
        h1 = unit_bounds[u + 1]
        si = hit_sent[h0]
        s = sent_offsets[si]
        e = sent_offsets[si + 1]
        h = np.uint64(14695981039346656037)  # FNV-1a over the sentence's words, all of them
        for k in range(s, e):
            h = (h ^ np.uint64(count_ids[k])) * np.uint64(1099511628211)
        hashes[u] = h
        first = nb
        for k in range(s, e):
            t = count_ids[k]
            if excluded[t] or filtered[t] or seen[t]:
                continue
            if not _nb_word_kept(k, prop_columns, prop_allowed):
                continue
            if max_dist >= 0:
                near = False
                for h in range(h0, h1):
                    if _nb_span_with(np.uint32(k - s + 1), hit_start[h], hit_end[h]) <= max_dist:
                        near = True
                        break
                if not near:
                    continue
            seen[t] = True
            bag[nb] = t
            nb += 1
        for k in range(first, nb):
            seen[bag[k]] = False
        bag_ptr[u + 1] = nb
        for h in range(h0, h1):
            if hit_start[h] > 1:
                t = count_ids[s + hit_start[h] - 2]
                if not excluded[t] and not filtered[t]:
                    left[nl] = t
                    nl += 1
            if hit_end[h] < e - s:
                t = count_ids[s + hit_end[h]]
                if not excluded[t] and not filtered[t]:
                    right[nr] = t
                    nr += 1
        left_ptr[u + 1] = nl
        right_ptr[u + 1] = nr
    return bag_ptr, left_ptr, right_ptr, hashes


@numba.njit(cache=True, nogil=True)
def _first_of_each(values):
    """The indices of the first of each of the values, in order (np.sort(np.unique(values, return_index=True)[1])), by
    a table of the values seen rather than sorting them."""
    n = len(values)
    size = 2
    while size < 2 * n:
        size *= 2
    mask = np.uint64(size - 1)
    table = np.empty(size, dtype=np.uint64)
    used = np.zeros(size, dtype=np.bool_)
    keep = np.empty(n, dtype=np.int64)
    m = 0
    for i in range(n):
        v = np.uint64(values[i])
        j = (v ^ (v >> np.uint64(29))) & mask
        seen = False
        while used[j]:
            if table[j] == v:
                seen = True
                break
            j = (j + np.uint64(1)) & mask
        if not seen:
            used[j] = True
            table[j] = v
            keep[m] = i
            m += 1
    return keep[:m]


@numba.njit(cache=True, nogil=True)
def _hit_sentences(sent_keys, rows):
    """The place of each hit's sentence among the sorted sentence keys (sent_keys_native.npy), by the first 6 fields of
    its row, or where it would go (the first key not before it), and whether it is there: looked for from the hit
    before's, by steps doubling away from it, then halving (hits in the order of the corpus are near one another)."""
    n_sents = sent_keys.shape[0]
    si = np.empty(rows.shape[0], dtype=np.int64)
    found = np.zeros(rows.shape[0], dtype=np.bool_)
    at = 0
    for i in range(rows.shape[0]):
        key = rows[i]
        step = 1
        if at < n_sents and _nb_compare_keys(sent_keys[at], key) < 0:  # after the key at: the place is past it
            below, above = at, at + 1  # sent_keys[below] before the key
            while above < n_sents and _nb_compare_keys(sent_keys[above], key) < 0:
                below = above
                step *= 2
                above = below + step
            lo, hi = below + 1, min(above, n_sents)
        else:  # not after the key at: the place is at it or before
            below, above = at - 1, at  # sent_keys[above] not before the key (or the end)
            while below >= 0 and _nb_compare_keys(sent_keys[below], key) >= 0:
                above = below
                step *= 2
                below = above - step
            lo, hi = max(below + 1, 0), above
        while lo < hi:
            mid = (lo + hi) // 2
            if _nb_compare_keys(sent_keys[mid], key) < 0:
                lo = mid + 1
            else:
                hi = mid
        at = lo
        si[i] = lo
        found[i] = lo < n_sents and _nb_compare_keys(sent_keys[lo], key) == 0
    return si, found


def _channel(flat, starts, ends, rows, n_vocab, submit=_now):
    """A sentences x words count matrix of local word indices, of the sentences rows (their words flat[starts[r] :
    ends[r]] for each r, vocabulary ids below n_vocab), and the vocabulary ids of its words: those of these sentences
    (the local index of a word, its rank among them). Each row's words in order, once each with its count, as scipy's
    sum_duplicates leaves them; made in two parts of about as many entries at once on the pool's threads (submit)."""
    lengths = ends[rows] - starts[rows]
    at = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(lengths, out=at[1:])
    half = int(np.searchsorted(at, at[-1] / 2))
    parts = [(lo, hi) for lo, hi in ((0, half), (half, len(rows))) if hi > lo]

    def marked(part):
        had = np.zeros(n_vocab, dtype=np.bool_)
        _mark_words(flat, starts, ends, rows, part[0], part[1], had)
        return had

    had = np.logical_or.reduce(_at_once(submit, marked, parts)) if parts else np.zeros(n_vocab, dtype=np.bool_)
    ids = np.flatnonzero(had).astype(flat.dtype)
    local = np.zeros(n_vocab, dtype=np.int32)
    local[ids] = np.arange(len(ids), dtype=np.int32)
    indices = np.empty(at[-1], dtype=np.int32)
    data = np.empty(at[-1])
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    end = _at_once(
        submit, lambda p: _sorted_rows(flat, starts, ends, rows, *p, local, indices, data, indptr, at), parts
    )
    if len(end) > 1 and end[0] < at[half]:  # words counted once in the first part: the second's moved up to it
        gap = at[half] - end[0]
        indices[end[0] : end[1] - gap] = indices[at[half] : end[1]].copy()
        data[end[0] : end[1] - gap] = data[at[half] : end[1]].copy()
        indptr[half + 1 :] -= gap
    X = csr_matrix((data[: indptr[-1]], indices[: indptr[-1]], indptr), shape=(len(rows), len(ids)))
    X.has_canonical_format = True
    return X, ids


@numba.njit(cache=True, nogil=True)
def _mark_words(flat, starts, ends, rows, lo, hi, had):
    """had[w] for each word w of the rows lo to hi of rows (flat[starts[r] : ends[r]] for each r)."""
    for q in range(lo, hi):
        r = rows[q]
        for t in range(starts[r], ends[r]):
            had[flat[t]] = True


@numba.njit(cache=True, nogil=True)
def _sorted_rows(flat, starts, ends, rows, lo, hi, local, indices, data, indptr, at):
    """The rows lo to hi of rows: each's words (flat[starts[r] : ends[r]] for each r) by their local index, in order,
    once each with its count (indices, data), from at[lo], and where each ends (indptr). Returns where the last
    ends."""
    w = at[lo]
    for q in range(lo, hi):
        r = rows[q]
        a = starts[r]
        n = ends[r] - a
        if n <= 64:  # by insertion, as they are copied
            for t in range(n):
                v = local[flat[a + t]]
                k = w + t
                while k > w and indices[k - 1] > v:
                    indices[k] = indices[k - 1]
                    k -= 1
                indices[k] = v
        else:
            for t in range(n):
                indices[w + t] = local[flat[a + t]]
            indices[w : w + n].sort()
        m = w
        for t in range(w, w + n):
            v = indices[t]
            if m > w and indices[m - 1] == v:
                data[m - 1] += 1.0
            else:
                indices[m] = v
                data[m] = 1.0
                m += 1
        w = m
        indptr[q + 1] = w
    return w


def _filter_mask(colloc_dir, prefix, filter_words, n_vocab):
    """The words of filter_words (as the collocation report's filter list has them), as a mask of the vocabulary."""
    if not filter_words:
        return np.zeros(n_vocab, dtype=np.bool_)
    hashes = np.load(os.path.join(colloc_dir, f"{prefix}vocab_hashes.npy"), mmap_mode="r")
    return _nb_build_filter(hashes, _filter_hashes(filter_words))


def _vocabulary(data_path, count_lemmas):
    """The strings of the words (or lemmas) of the collocation vocabulary, and where each starts."""
    colloc_dir = os.path.join(data_path, "collocations")
    prefix = "attr_lemma_" if count_lemmas else ""
    v_offsets = np.asarray(np.load(os.path.join(colloc_dir, f"{prefix}vocab_offsets.npy"), mmap_mode="r"))
    with open(os.path.join(colloc_dir, f"{prefix}vocab_strings.bin"), "rb") as f:
        return f.read(), v_offsets


def query_sentences(db, q, count_lemmas, properties=(), metadata=None, distance=None, filter_words=None):
    """The sentences the query is in, one for each sentence, with their words: a dict, or None without any. The words
    of filter_words (as the collocation report's filter list has them) are left out, of the words of the sentence and
    of those next to the query alike; the words of the sentence without a kept value of each word property of
    properties ((name, values) pairs, as the collocation report's filters have them) too."""
    metadata = metadata or {}
    workers = numba.config.NUMBA_NUM_THREADS
    with ThreadPoolExecutor(max_workers=max(1, workers - 1)) as pool:
        submit = pool.submit if workers > 1 else _now
        return _query_sentences(db, q, count_lemmas, properties, metadata, distance, filter_words, submit)


def _query_sentences(db, q, count_lemmas, properties, metadata, distance, filter_words, submit):
    """query_sentences, its compiled steps in two parts at once, on this thread and the pool's (submit)."""
    data_path = os.path.normpath(db.path)
    method, method_arg = collocation_search_method(q, db.locals.query_patterns, distance)
    hits = db.query(q, method, method_arg, raw_results=True, raw_bytes=True, **metadata)
    hits.finish()
    n_hits = len(hits)
    if n_hits == 0:
        return None
    with hits.open_raw() as f:
        rows = np.frombuffer(f.read(), dtype=np.uint32).reshape(-1, hits.length)

    colloc_dir = os.path.join(data_path, "collocations")
    sent_keys = np.load(os.path.join(colloc_dir, "sent_keys_native.npy"), mmap_mode="r")
    sent_offsets = np.asarray(np.load(os.path.join(colloc_dir, "sent_offsets.npy"), mmap_mode="r"))
    prefix = "attr_lemma_" if count_lemmas else ""
    count_ids = np.load(
        os.path.join(colloc_dir, "attr_lemma_ids.npy" if count_lemmas else "token_ids.npy"), mmap_mode="r"
    )
    v_blob, v_offsets = _vocabulary(data_path, count_lemmas)
    n_vocab = len(v_offsets) - 1

    halves = [(lo, hi) for lo, hi in ((0, len(rows) // 2), (len(rows) // 2, len(rows))) if hi > lo]
    made = _at_once(submit, lambda part: _hit_sentences(sent_keys, rows[part[0] : part[1]]), halves)
    si, matched = (np.concatenate(got) for got in zip(*made))
    hit_index = np.flatnonzero(matched)
    if len(hit_index) == 0:
        return None
    si = si[hit_index].astype(np.int64)
    positions = rows[hit_index][:, 7::2].astype(np.int64)  # the word position of each of the hit's query words
    # The query's own words: whatever form or lemma is at its hits' positions, wherever it is in the sentence
    at = (sent_offsets[si].astype(np.int64)[:, None] + positions - 1)[positions > 0]
    excluded = np.zeros(n_vocab, dtype=np.bool_)
    excluded[np.asarray(count_ids[at])] = True
    filtered = _filter_mask(colloc_dir, prefix, filter_words, n_vocab)

    order = np.argsort(si, kind="stable")
    hit_index, si, positions = hit_index[order], si[order], positions[order]
    unit_bounds = np.append(np.flatnonzero(np.r_[True, si[1:] != si[:-1]]), len(si))
    prop_columns, prop_allowed = property_masks(data_path, properties)
    max_dist = distance + phrase_words(q, db.locals.query_patterns) if distance is not None else -1
    n_units = len(unit_bounds) - 1
    # The sentences in two parts of about as many words, each's words into its own stretch of the arrays
    first_si = si[unit_bounds[:-1]]
    words_at = np.zeros(n_units + 1, dtype=np.int64)
    np.cumsum(sent_offsets[first_si + 1].astype(np.int64) - sent_offsets[first_si], out=words_at[1:])
    half = int(np.searchsorted(words_at, words_at[-1] / 2))
    parts = [(lo, hi) for lo, hi in ((0, half), (half, n_units)) if hi > lo]
    bag = np.empty(words_at[-1], dtype=np.int64)
    left = np.empty(len(si), dtype=np.int64)
    right = np.empty(len(si), dtype=np.int64)
    hit_start = positions.min(axis=1).astype(np.uint32)
    hit_end = positions.max(axis=1).astype(np.uint32)

    def words_of(part):
        lo, hi = part
        return _sentence_words(
            unit_bounds[lo : hi + 1],
            si,
            hit_start,
            hit_end,
            sent_offsets,
            count_ids,
            excluded,
            filtered,
            prop_columns,
            prop_allowed,
            np.int32(max_dist),
            n_vocab,
            bag,
            left,
            right,
            words_at[lo],
            unit_bounds[lo],
        )

    made = _at_once(submit, words_of, parts)
    bounds = [np.concatenate([ptr[:-1] for ptr in got]) for got in zip(*(m[:3] for m in made))]
    ends = [np.concatenate([ptr[1:] for ptr in got]) for got in zip(*(m[:3] for m in made))]
    hashes = np.concatenate([m[3] for m in made])
    first = hit_index[unit_bounds[:-1]]  # the first hit of each sentence, in the hitlist
    # Identical sentences count once (repeated titles, running heads, reprints): they are one use, not many
    keep = _first_of_each(hashes)
    channels = [_channel(flat, a, b, keep, n_vocab, submit) for flat, a, b in zip((bag, left, right), bounds, ends)]
    first = first[keep]
    # The words just before and after each hit as the text has them: their forms, and where the hit is in its text
    forms = np.load(os.path.join(colloc_dir, "token_ids.npy"), mmap_mode="r") if count_lemmas else count_ids
    s0 = sent_offsets[si].astype(np.int64)
    length = sent_offsets[si + 1].astype(np.int64) - s0
    lo, hi = positions.min(axis=1), positions.max(axis=1)
    kept_of = np.full(n_units, -1, dtype=np.int64)
    kept_of[keep] = np.arange(len(keep))
    unit_of = kept_of[np.repeat(np.arange(n_units), np.diff(unit_bounds))]
    hit_bytes = rows[hit_index][:, 8::2].astype(np.int64)
    query_form = np.asarray(forms[s0 + positions[:, 0] - 1]).astype(np.int64)
    neighbors = [None]
    for at, ok in ((s0 + lo - 2, lo > 1), (s0 + hi, hi < length)):  # as _sentence_words takes them
        at = np.where(ok, at, 0)
        near = np.asarray(count_ids[at]).astype(np.int64)
        ok = ok & (unit_of >= 0) & ~excluded[near] & ~filtered[near]
        neighbors.append(
            {
                "unit": unit_of[ok],
                "word": near[ok],
                "form": np.asarray(forms[at[ok]]).astype(np.int64),
                "query_form": query_form[ok],
                "hit": hit_index[ok],
                "doc": rows[hit_index[ok], 0].astype(np.int64),
                "first_byte": hit_bytes[ok].min(axis=1),
                "last_byte": hit_bytes[ok].max(axis=1),
            }
        )
    return {
        "n_hits": n_hits,
        "n_units": len(keep),
        "duplicates": n_units - len(keep),
        "first_hit": first,
        "docs": rows[first, 0].astype(np.int64),
        "divisions": rows[first, :3].astype(np.int64),
        "channels": channels,
        "v_blob": v_blob,
        "v_offsets": v_offsets,
        "count_lemmas": count_lemmas,
        "filtered": filtered,  # the words of filter_words, by vocabulary id
        "data_path": data_path,
        "neighbors": neighbors,  # for each group of words next to the query (1, before; 2, after), as above
        "query_words": int(positions.shape[1]),
        "sentence_index": first_si,  # each sentence's place in the collocation index, identical ones each
    }


# ----------------------------- The report ------------------------------


def _cache_path(db, *parts):
    h = hashlib.sha1(_CACHE_TAG)
    for part in parts:
        h.update(json.dumps(part, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"\0")
    return os.path.join(db.hitlist_dir, f"{h.hexdigest()}.senses.json")


def word_senses(
    db,
    q,
    count_lemmas,
    properties=(),
    metadata=None,
    distance=None,
    year_field=None,
    filter_words=None,
):
    """The senses of the query and their course over time, as JSON-ready data (cached in the hitlist directory). The
    words of filter_words (the collocation report's filter) are not shown but next to the query, and seed no split; the
    analysis reads them, as every word."""
    metadata = {k: v for k, v in (metadata or {}).items() if v}
    filter_words = sorted(set(filter_words or []))
    path = _cache_path(db, q, bool(count_lemmas), properties, metadata, distance, year_field, filter_words)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                report = json.load(f)
            _touch(path)
            return report
        except (OSError, ValueError):
            pass
    began = time.perf_counter()
    result = _word_senses(db, q, count_lemmas, properties, metadata, distance, year_field, filter_words)
    result["seconds"] = round(time.perf_counter() - began, 2)
    result["filter_list"] = filter_words
    fd, tmp = tempfile.mkstemp(dir=db.hitlist_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(result, f)
        os.replace(tmp, path)
    except OSError:  # a cache that can't be written is no cache
        if os.path.exists(tmp):
            os.unlink(tmp)
    return result


def _touch(path):
    """Marks a cached result as just used: the hitlist cleanup keeps the senses reports until they take too much
    room, and then removes those used least recently first (see www/middleware.py)."""
    try:
        os.utime(path)
    except OSError:
        pass


def _word_senses(db, q, count_lemmas, properties, metadata, distance, year_field, filter_words=None):
    # The analysis reads every word, rare ones weighing more: left out of it, the filter's words that tell a rare sense
    # (dieu for prière) would be too (see the pseudoword test). They only seed no split
    sentences = query_sentences(db, q, count_lemmas, properties, metadata, distance)
    if sentences is None:
        return {"n_hits": 0, "n_sentences": 0, "n_works": 0, "nodes": [], "time": None}
    channels, names = _senses_channels(sentences, count_lemmas)
    common = _filtered(sentences, filter_words)

    def cooccurrence(columns):  # of the corpus's sentences outside the query's
        return corpus_cooccurrence(sentences["data_path"], count_lemmas, names[0][columns], sentences["sentence_index"])

    root = induce_senses(
        channels, sentences["docs"], unseeded=[common[ids] for ids in names], cooccurrence=cooccurrence
    )
    return _senses_report(db, sentences, channels, names, root, filter_words, metadata, year_field, common)


def _senses_channels(sentences, count_lemmas):
    """The groups of words of the sentences, as induce_senses takes them, and the vocabulary ids of their words. Each
    word's evidence weighs by its inverse document frequency: of the corpus's sentences for the words of the
    sentence, of its running words for those just before and after."""
    data_path = sentences["data_path"]
    df, n_corpus_sentences = _load_sentence_df(data_path, count_lemmas)
    tf = _load_token_counts(data_path, count_lemmas)
    channels = []
    names = []
    for (X, ids), background, units in zip(sentences["channels"], (df, tf, tf), (n_corpus_sentences, None, None)):
        total = float(background.sum())
        # The corpus's rates among the words left in (filtered words are in no sentence, so none is expected)
        p = background[ids] / (total - float(background[sentences["filtered"]].sum()))
        weights = np.log((units if units else total) / background[ids])
        channels.append((X, p, max(0.0, 1.0 - float(p.sum())), weights))
        names.append(ids)
    return channels, names


_TAGS = regex.compile(rb"<[^>]*>|<[^>]*$|^[^<]*>")
_SPACES = regex.compile(r"\s+")
_JOINING = regex.compile(r"[^\s\p{L}\p{M}\p{N}]+(?=[\p{L}\p{M}\p{N}])")  # a mark joining two words (d'un)


@numba.njit(cache=True, nogil=True)
def _most_common_forms(order, unit, word, form, query_form, inside):
    """For each word of the rows (by word, form, query form and hit: order, and each of these in that order) in the
    sentences inside, the first of its rows with its most common pair of forms, the first such pair among equals, in
    the order of the words."""
    words = np.empty(len(order), dtype=np.int64)
    rows = np.empty(len(order), dtype=np.int64)
    n = 0
    best = 0  # the count of the current word's most common forms so far
    count = 0  # of the current forms
    start = -1  # their first row
    last = -1  # the place of the last row inside
    for t in range(len(order)):
        if not inside[unit[t]]:
            continue
        new_word = last < 0 or word[t] != word[last]
        if new_word or form[t] != form[last] or query_form[t] != query_form[last]:
            if start >= 0 and count > best:  # the forms just counted, more common than those before
                rows[n - 1] = order[start]
                best = count
            if new_word:
                words[n] = word[t]
                n += 1
                best = 0
            start = t
            count = 0
        count += 1
        last = t
    if start >= 0 and count > best:
        rows[n - 1] = order[start]
    return words[:n], rows[:n]


class _Phrases:
    """The words next to the query as the text has them: "sa volonté" for the lemma son before volonté, "l'esprit"
    for le (the words of the index lose their apostrophes and capitals). For a sense, a word's most common forms next
    to the query (and the query's), read from the first hit that has them; None when the text does not read so."""

    def __init__(self, db, sentences):
        self.db = db
        self.neighbors = sentences.get("neighbors")
        self.n_words = sentences.get("query_words", 1)
        self.n_units = sentences["n_units"]
        self.text_dir = os.path.join(sentences["data_path"], "TEXT")
        self.word_re = regex.compile(db.locals["token_regex"])
        if sentences["count_lemmas"]:
            self.f_blob, self.f_offsets = _vocabulary(sentences["data_path"], False)
        else:
            self.f_blob, self.f_offsets = sentences["v_blob"], sentences["v_offsets"]
        self.filenames = {}
        self.cache = {}
        self.sorted = {}

    def _sorted(self, side):
        """The rows of the hits next to the query on a side by word, its form, the query's form and hit, and their
        sentences: sorted once, those of any sentences are then in order (each hit is in one row)."""
        if side not in self.sorted:
            nb = self.neighbors[side]
            order = np.lexsort((nb["hit"], nb["query_form"], nb["form"], nb["word"]))
            self.sorted[side] = order, *(nb[k][order] for k in ("unit", "word", "form", "query_form"))
        return self.sorted[side]

    def near(self, units):
        """For the sentences units: {group: {vocabulary id of a word next to the query: the row of its first hit with
        its most common forms}}."""
        out = {}
        if self.neighbors is None:
            return out
        inside = np.zeros(self.n_units, dtype=np.bool_)
        inside[units] = True
        for side in (1, 2):
            words, rows = _most_common_forms(*self._sorted(side), inside)
            out[side] = dict(zip(words.tolist(), rows.tolist()))
        return out

    def _form(self, i):
        return self.f_blob[int(self.f_offsets[i]) : int(self.f_offsets[i + 1])].decode("utf-8", errors="replace")

    def _text(self, raw):
        return html.unescape(_TAGS.sub(b"", raw).decode("utf-8", errors="ignore"))

    def text(self, side, row):
        """The phrase of a word next to the query, at a row of the hits next to it (None: as the words read)."""
        if row is None or self.neighbors is None:
            return None
        nb = self.neighbors[side]
        key = (side, int(nb["form"][row]), int(nb["query_form"][row]))
        if key not in self.cache:
            try:
                self.cache[key] = self._read(side, row)
            except (OSError, ValueError, KeyError, IndexError, TypeError):  # no text to read: as the words read
                self.cache[key] = None
        return self.cache[key]

    def _read(self, side, row):
        nb = self.neighbors[side]
        doc = int(nb["doc"][row])
        if doc not in self.filenames:
            self.filenames[doc] = self.db[(doc,)].filename
        mm = _get_mmap(os.path.join(self.text_dir, self.filenames[doc]))
        b0, b1 = int(nb["first_byte"][row]), int(nb["last_byte"][row])
        span = self._text(mm[b0 : b1 + _TEXT_WINDOW])
        words = list(self.word_re.finditer(span))
        k = self.n_words
        if len(words) < k or words[0].group().lower() != self._form(int(nb["query_form"][row])):
            return None
        query = span[: words[k - 1].end()]
        if side == 1:
            before = self._text(mm[max(0, b0 - _TEXT_WINDOW) : b0])
            last = None
            for last in self.word_re.finditer(before):
                pass
            if last is None:
                return None
            word, phrase = last.group(), last.group() + before[last.end() :] + query
        else:
            if len(words) <= k:
                return None
            nxt = words[k]
            word = nxt.group()
            phrase = query + span[words[k - 1].end() : nxt.start()] + word
            joined = _JOINING.match(span, nxt.end())  # a word joined to the next reads with it: d'un, cardinal's
            if joined and len(words) > k + 1 and words[k + 1].start() == joined.end():
                phrase += joined.group() + words[k + 1].group()
        if word.lower() != self._form(int(nb["form"][row])):
            return None
        return _SPACES.sub(" ", phrase).strip().lower()


def _filtered(sentences, filter_words):
    """The words of filter_words (as the collocation report's filter list has them), as a mask of the vocabulary of
    the sentences."""
    prefix = "attr_lemma_" if sentences["count_lemmas"] else ""
    colloc_dir = os.path.join(sentences["data_path"], "collocations")
    return _filter_mask(colloc_dir, prefix, filter_words, len(sentences["v_offsets"]) - 1)


def _senses_report(db, sentences, channels, names, root, filter_words, metadata, year_field, unshown=None):
    """The report of the tree of senses root, as JSON-ready data. unshown: the words of filter_words, as _filtered
    has them (made again without). Its parts that do not depend on one another are made at once, on this thread and
    those of a pool (as many in all as numba may use, as for induce_senses), but for the texts read for the phrases
    next to the query, here (the database is this thread's), once all are known."""
    workers = numba.config.NUMBA_NUM_THREADS
    with ThreadPoolExecutor(max_workers=max(1, workers - 1)) as pool:
        submit = pool.submit if workers > 1 else _now
        return _senses_report_with(
            db, sentences, channels, names, root, filter_words, metadata, year_field, unshown, submit
        )


def _senses_report_with(db, sentences, channels, names, root, filter_words, metadata, year_field, unshown, submit):
    unshown = _filtered(sentences, filter_words) if unshown is None else unshown
    n = sentences["n_units"]
    docs = sentences["docs"]
    Xall = [csr_matrix(c[0]) for c in channels]
    phrases = _Phrases(db, sentences)
    columns = {}

    def columns_of(ci):  # the sentences of each word of a group (_pattern_columns), made once
        if ci not in columns:
            columns[ci] = _pattern_columns(Xall[ci].indptr, Xall[ci].indices, Xall[ci].shape[1])
        return columns[ci]

    def first(part):  # what the parts below take from: the senses' posteriors, the words' sentences, the phrases' rows
        if part == "posteriors":
            return node_posteriors(root, Xall, submit)
        if part == "sorted":
            return [phrases._sorted(side) for side in (1, 2) if phrases.neighbors is not None]
        return columns_of(part)

    (nodes, logp, margin), *_ = _at_once(submit, first, ["posteriors", 0, "sorted", 1, 2])
    posteriors = np.exp(logp)
    weights = [c[3] for c in channels]
    corpus_p = [c[1] for c in channels]

    v_blob, v_offsets = sentences["v_blob"], sentences["v_offsets"]

    # The filter's words: not among a sense's words, but next to the query, where common words make the construction
    # (homme d'esprit, mon esprit). For each group of words, None or the mask of those not shown
    not_shown = [unshown[names[0]]] + [None] * (len(names) - 1)

    def word(ci, local):
        tid = int(names[ci][local])
        s = v_blob[int(v_offsets[tid]) : int(v_offsets[tid + 1])].decode("utf-8", errors="replace")
        return s[6:] if s.startswith("lemma:") else s

    top_importance = max((nd.test["importance"] for nd in nodes if nd.children), default=0.0)
    first_hit = sentences["first_hit"]
    listed_words = {}
    near_by_node = {}
    mine_of = {}

    def sentences_of(node):  # its sentences: those on its side of every split on the way to it
        if node.id not in mine_of:
            mine_of[node.id] = np.flatnonzero(margin[:, node.id] > 0)
        return mine_of[node.id]

    def near(node):  # the words next to the query in a sense's sentences, as the text has them (see _Phrases)
        if node.id not in near_by_node:
            near_by_node[node.id] = phrases.near(sentences_of(node))
        return near_by_node[node.id]

    asked = []  # the words of the senses, to which their sentences are added once all are known (with_hits)

    def with_hits(item, node, ci, i):
        asked.append((item, node.id, ci, i))
        return item

    def hits_of(items):
        """To each word of a sense, the sentences of the sense with it, the most clearly in the sense first (their
        first hits): from the sentences of each word, for the words of a sense."""
        nid = items[0][1]
        inside = margin[:, nid] > 0
        clear_of = posteriors[:, nid]
        for item, _, ci, i in items:
            col_ptr, rows = columns_of(ci)
            rows = rows[col_ptr[i] : col_ptr[i + 1]]
            rows = rows[inside[rows]]
            clear = clear_of[rows]
            if len(rows) > _WORD_EXAMPLES:  # the clearest only, without sorting them all
                keep = np.argpartition(-clear, _WORD_EXAMPLES)[:_WORD_EXAMPLES]
                rows, clear = rows[keep], clear[keep]
            item["hits"] = first_hit[rows[np.argsort(-clear, kind="stable")]].astype(np.int64).tolist()

    def with_phrase(item, node, ci, i, read):  # its phrase, read once all are known: (item, group, row) to read
        if ci > 0:
            read.append((item, ci, near(node).get(ci, {}).get(int(names[ci][i]))))
        return item

    def add_phrases(read):  # in the order they were asked for: the first read of a phrase is the one kept
        for item, ci, row in read:
            phrase = phrases.text(ci, row)
            if phrase:
                item["phrase"] = phrase

    listed_of = {}

    def listed_words_of(node, ci, against):  # _described's words, made once for a sense and those it is told from
        key = (node.id, ci, id(against))
        if key not in listed_of:
            listed_of[key] = _described(node.counts[ci], node.dists[ci], against[ci].p)[0]
        return listed_of[key]

    def described_words(node, ci, against, listed):  # describe's items of the words listed
        c, p = node.counts[ci], node.dists[ci].p
        counts = c[listed].astype(np.float64).tolist()
        ratios = (p[listed] / against[ci].p[listed]).tolist()
        lifts = (p[listed] / corpus_p[ci][listed]).tolist()
        return list(zip(listed.tolist(), counts, ratios, lifts))

    def rarer_words(node, ci, against):  # describe_rarer's items of the words shown, only as many as are sent
        listed, g2 = _described_rarer(node.counts[ci], node.dists[ci], against[ci].p)
        if not_shown[ci] is not None:
            keep = ~not_shown[ci][listed]
            listed, g2 = listed[keep], g2[keep]
        listed, g2 = listed[:_MAX_WORDS], g2[:_MAX_WORDS]
        c, p = node.counts[ci], node.dists[ci].p
        counts = c[listed].astype(np.float64).tolist()
        return list(zip(listed.tolist(), counts, (p[listed] / against[ci].p[listed]).tolist(), g2.tolist()))

    def described(node, against, read):
        """A sense's words, against the dists of the sense it is told from (as a rule, its parent): those markedly more
        frequent in it, and what it lacks. read: the phrases to read, (item, group, row)."""
        groups_of_words = []
        hidden = []
        for ci in range(len(channels)):
            # As describe lists them, those of its words shown and those not (the filter's), only as many as are sent
            listed = listed_words_of(node, ci, against)
            if not_shown[ci] is not None:
                hidden = described_words(node, ci, against, listed[not_shown[ci][listed]][:_MAX_WORDS])
                listed = listed[~not_shown[ci][listed]]
            listed = described_words(node, ci, against, listed[:_MAX_WORDS])
            if ci == 0:
                listed_words[node.id] = np.array([i for i, *_ in listed], dtype=np.int64)

            def entry(i, c, r, f):  # a word of the sense, as the page shows it
                item = {"word": word(ci, i), "n": round(c, 2), "ratio": round(r, 3), "lift": round(f, 3)}
                return with_hits(with_phrase(item, node, ci, i, read), node, ci, i)

            groups_of_words.append([entry(i, c, r, f) for i, c, r, f in listed])
        # A sense with nothing else to show: the words of the filter that mark it
        if not any(groups_of_words):
            hidden = [
                {"word": word(0, i), "n": round(c, 2), "ratio": round(r, 3), "lift": round(lift, 3)}
                for i, c, r, lift in hidden
            ]
        else:
            hidden = []
        # Its words that go together in its sentences, across the works, around hub words (as in HyperLex)
        groups = _word_groups(sentences_of(node), columns_of(0), docs, listed_words.get(node.id))
        # What it lacks, against the sense it was split from, in any group of words (the strongest evidence first)
        rarer = []
        if node.parent is not None:
            rarer = sorted(
                (
                    (g, {"word": word(ci, i), "group": ci, "n": round(c, 2), "ratio": round(r, 4), "at": (ci, i)})
                    for ci in range(len(channels))
                    for i, c, r, g in rarer_words(node, ci, against)
                ),
                key=lambda item: -item[0],
            )
            rarer = [with_phrase(item, node.parent, *item.pop("at"), read) for _, item in rarer[:_MAX_WORDS]]
        return {
            "words": groups_of_words[0],
            "hidden": hidden,
            "before": groups_of_words[1],
            "after": groups_of_words[2],
            "groups": groups,
            "rarer": rarer,
        }

    # Something of its own to show, for a sense: a word markedly more frequent in it, in any group, outside the
    # collocation filter (whose words, mostly function words, tell the style of works more than a sense: alone, they
    # make none)
    common = [unshown[ids] for ids in names]
    marked = {
        node.id: marks(
            node,
            node.parent_dists,
            common,
            [listed_words_of(node, ci, node.parent_dists) for ci in range(len(channels))],
        )
        for node in nodes
    }

    def sense(node):  # a sense as the page shows it, and the phrases it asks for
        read = []
        split = None
        if node.test is not None:
            split = {k: v for k, v in node.test.items() if k != "gains"}
            seed = split.pop("seed", None)
            split["seed"] = (
                None if seed is None else with_phrase({"word": word(*seed), "group": seed[0]}, node, *seed, read)
            )
            split["relative"] = round(float(split["importance"] / top_importance), 4) if top_importance > 0 else 0.0
        mine = sentences_of(node)
        works_of = docs[mine]
        work_counts = np.unique(works_of, return_counts=True)[1] if len(mine) else np.zeros(1)
        item = {
            "id": node.id,
            "parent": None if node.parent is None else node.parent.id,
            "children": [child.id for child in node.children],
            "share": round(float(posteriors[:, node.id].mean()), 6),
            "sentences": int(len(mine)),
            **described(node, node.parent_dists, read),
            "split": split,
            "reason": node.reason,
            "works": int(len(work_counts)) if len(mine) else 0,
            "top_work_share": round(float(work_counts.max() / max(1, len(mine))), 4),
            "examples": _examples(node, root, Xall, weights, posteriors, margin, mine, first_hit, word, not_shown),
        }
        return item, read

    timeline = None
    years = None
    if year_field:
        try:
            year_array, year_words, _, _, _ = _get_doc_year_data(db, year_field)
            docs_ok = np.clip(docs, 0, len(year_array) - 1)
            years = np.where(docs < len(year_array), year_array[docs_ok], 0).astype(np.int64)
            if metadata:
                year_words, _ = _corpus_year_counts(db, year_array, metadata)
        except Exception:  # no year field the database can read: senses without their course over time
            years = None

    def part(task):  # the senses and their course over time, at once
        if task is None:
            try:
                return sense_curves(posteriors, years, docs, year_words)
            except Exception:  # no course the years can tell
                return None
        return sense(task)

    made = _at_once(submit, part, ([None] if years is not None else []) + nodes)
    if years is not None:
        timeline, made = made[0], made[1:]
    out = [item for item, _ in made]
    reads = [read for _, read in made]

    # The senses' numbers; a sense that goes on as its parent is told from the sense its number started in (the
    # rest of a sense, after one is carved out of it, has the words of that sense, not those of none)
    number, start = sense_numbers(out, lambda item: marked[item["id"]])
    rank = {n: k + 1 for k, n in enumerate(sorted(start, key=lambda n: [int(x) for x in n.split(".")]))}
    again = [node for node in nodes if number[node.id] is not None and start[number[node.id]] != node.id]

    def told_again(node):
        read = []
        return described(node, nodes[start[number[node.id]]].parent_dists, read), read

    for node, (words, read) in zip(again, _at_once(submit, told_again, again) if again else []):
        out[node.id].update(words)
        reads.append(read)
    for item in out:
        item["sense"] = number[item["id"]]
        item["rank"] = None if item["sense"] is None else rank[item["sense"]]
    for read in reads:
        add_phrases(read)
    by_node = {}
    for a in asked:
        by_node.setdefault(a[1], []).append(a)
    if by_node:
        _at_once(submit, hits_of, list(by_node.values()))
    return {
        "n_hits": int(sentences["n_hits"]),
        "n_sentences": int(n),
        "n_works": int(len(np.unique(docs))),
        "duplicates": int(sentences["duplicates"]),
        "folds": N_FOLDS,
        "evidence": EVIDENCE,
        "resamples": N_RESAMPLES,
        "band": list(BAND),
        "detail": default_detail(out, number),
        "finest": FINEST,
        "nodes": out,
        "time": timeline,
    }


# ----------------------------- Word groups: HyperLex inside a sense ------------------------------


def _works_of(works, n):
    """The work of each sentence numbered from 0, the sentences in order of their works, where each work's begin, and
    the number of works (None when the sentences are all from one work: nothing can be replicated across works)."""
    if works is None:
        return None
    _, work, sizes = np.unique(np.asarray(works), return_inverse=True, return_counts=True)
    work = np.ravel(work).astype(np.int64)
    if len(sizes) < 2:
        return None
    order = np.argsort(work, kind="stable")
    bounds = np.r_[0, np.cumsum(sizes)].astype(np.int64)
    return work, order, bounds, len(sizes)


def _by_works(X, by_works):
    """Each work's count of sentences with each word (works x words), and its number of sentences."""
    work, _, bounds, n_works = by_works
    W = csr_matrix((np.ones(X.shape[0]), (work, np.arange(X.shape[0]))), shape=(n_works, X.shape[0]))
    return (W @ X).tocsr(), np.diff(bounds).astype(np.float64)


def _replicated(excess_sum, excess_squares, n_works):
    """Whether a mean excess over chance per work is more than EVIDENCE standard errors over the works."""
    mean = excess_sum / n_works
    variance = np.maximum(excess_squares - n_works * mean**2, 0.0) / (n_works - 1)
    se = np.sqrt(variance / n_works)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(se > 0, mean / se, 0.0) > EVIDENCE


@numba.njit(cache=True, nogil=True)
def _pair_moments(indptr, indices, order, bounds, pair_id, pair_a, pair_b, n_words):
    """For each pair of words (pair_id: words x words, -1 for none), the sums over works of its count of sentences
    with both squared, and of that count times its expected count (the product of the work's counts of sentences with
    either, over its sentences)."""
    n_pairs = len(pair_a)
    squares = np.zeros(n_pairs)
    cross = np.zeros(n_pairs)
    count = np.zeros(n_words)
    both = np.zeros(n_pairs)
    touched = np.empty(n_pairs, dtype=np.int64)
    for d in range(len(bounds) - 1):
        n_touched = 0
        for t in range(bounds[d], bounds[d + 1]):
            s = order[t]
            for x in range(indptr[s], indptr[s + 1]):
                a = indices[x]
                count[a] += 1.0
                for y in range(x + 1, indptr[s + 1]):
                    b = indices[y]
                    p = pair_id[a, b] if a < b else pair_id[b, a]
                    if p >= 0:
                        if both[p] == 0.0:
                            touched[n_touched] = p
                            n_touched += 1
                        both[p] += 1.0
        n_d = bounds[d + 1] - bounds[d]
        for t in range(n_touched):
            p = touched[t]
            expected = count[pair_a[p]] * count[pair_b[p]] / n_d
            squares[p] += both[p] * both[p]
            cross[p] += both[p] * expected
            both[p] = 0.0
        for t in range(bounds[d], bounds[d + 1]):
            s = order[t]
            for x in range(indptr[s], indptr[s + 1]):
                count[indices[x]] = 0.0
    return squares, cross


def links_of(X, works=None):
    """The links between the words of X (sentences x words, 1 where a sentence has a word): pairs (i < j) whose
    pointwise mutual information is above zero even at the lower end of its uncertainty, and, given the work of each
    sentence (works), whose excess of sentences with both over chance holds across the works: its mean over the works
    is more than EVIDENCE standard errors, so that a passage repeating its words in one work is no link. Returns i, j,
    their weight (the lower end of the PMI, normalized by -log of the share of sentences with both: NPMI) and their
    count of sentences."""
    X = csr_matrix(X, dtype=np.float64)
    X.data[:] = 1.0
    X.sort_indices()
    n = X.shape[0]
    n_words = X.shape[1]
    counts = np.asarray(X.sum(axis=0)).ravel()
    both = (X.T @ X).tocoo()
    upper = both.row < both.col
    i, j, c = both.row[upper].astype(np.int64), both.col[upper].astype(np.int64), both.data[upper]
    # The exact (Garwood) lower bound of a Poisson count, EVIDENCE standard errors' worth below
    low = gammaincinv(c, ndtr(-EVIDENCE))
    expected = counts[i] * counts[j] / n
    with np.errstate(divide="ignore"):
        pmi_low = np.log(low / expected)
    linked = pmi_low > 0
    by_works = _works_of(works, n)
    if by_works is not None and linked.any():
        i, j, c, pmi_low = i[linked], j[linked], c[linked], pmi_low[linked]
        work, order, bounds, n_works = by_works
        per_work, sizes = _by_works(X, by_works)
        inverse = csr_matrix((1.0 / sizes, (np.arange(n_works), np.arange(n_works))), shape=(n_works, n_works))
        expected_sum = (per_work.T @ inverse @ per_work).toarray()[i, j]
        squared = per_work.multiply(per_work).tocsr()
        expected_squares = (squared.T @ (inverse @ inverse) @ squared).toarray()[i, j]
        pair_id = np.full((n_words, n_words), -1, dtype=np.int64)
        pair_id[i, j] = np.arange(len(i))
        squares, cross = _pair_moments(
            X.indptr.astype(np.int64), X.indices.astype(np.int64), order, bounds, pair_id, i, j, n_words
        )
        linked = _replicated(c - expected_sum, squares - 2.0 * cross + expected_squares, n_works)
    i, j, c, pmi_low = i[linked], j[linked], c[linked], pmi_low[linked]
    share = c / n
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(share < 1.0, pmi_low / -np.log(share), 1.0)
    return i, j, np.minimum(weight, 1.0), c


def hub_groups(X, works=None):
    """Group the words of X (sentences x words, 1 where a sentence has a word) as HyperLex does, without its
    thresholds: words are linked as links_of links them; the word with the most weight of links is a hub, and takes the
    words it is linked to; the next among the others, and so on, a word going to the hub it is linked to most. Returns
    a list of (hub, its words, strongest first), the most linked hub first; words linked to none are in none."""
    n_words = X.shape[1]
    li, lj, w, _ = links_of(X, works)
    if len(li) == 0:
        return []
    weight = np.zeros((n_words, n_words))
    weight[li, lj] = weight[lj, li] = w
    group_of = np.full(n_words, -1, dtype=np.int64)
    hubs = []
    while True:
        free = group_of < 0
        strength = (weight[:, free] * free[:, None]).sum(axis=1) * free
        if strength.max() <= 0:
            break
        hub = int(np.argmax(strength))
        group_of[hub] = len(hubs)
        group_of[np.flatnonzero(free & (weight[hub] > 0))] = len(hubs)  # the free words it is linked to
        hubs.append(hub)
    # Each grouped word goes to the hub it is linked to most
    for word in np.flatnonzero(group_of >= 0):
        if word in hubs:
            continue
        to_hubs = weight[word, hubs]
        group_of[word] = int(np.argmax(to_hubs))
    out = []
    for g, hub in enumerate(hubs):
        members = [int(x) for x in np.flatnonzero(group_of == g) if x != hub]
        members.sort(key=lambda x: -weight[hub, x])
        out.append((hub, members))
    return out


@numba.njit(cache=True, nogil=True)
def _pattern_columns(indptr, indices, n_cols):
    """The sentences of each word of a sentences x words matrix in CSR, in order (its columns' row indices, without
    the counts), and where each word's begin."""
    n = len(indptr) - 1
    col_ptr = np.zeros(n_cols + 1, dtype=np.int64)
    for j in range(indptr[0], indptr[n]):
        col_ptr[indices[j] + 1] += 1
    for c in range(n_cols):
        col_ptr[c + 1] += col_ptr[c]
    at = col_ptr[:-1].copy()
    rows = np.empty(col_ptr[n_cols], dtype=np.int32)
    for i in range(n):
        for j in range(indptr[i], indptr[i + 1]):
            rows[at[indices[j]]] = i
            at[indices[j]] += 1
    return col_ptr, rows


def _word_groups(mine, columns, docs, listed):
    """The groups of a sense's listed words (positions in its words) around hub words: [{"hub": position, "members":
    [positions]}], words linked in its sentences mine across the works (hub_groups). columns: the sentences of each
    word of the group (_pattern_columns), of which the matrix of its sentences and listed words is made."""
    if listed is None or len(listed) < 2 or len(mine) < 2:
        return []
    col_ptr, rows = columns
    parts = [rows[col_ptr[j] : col_ptr[j + 1]] for j in listed]
    at = np.concatenate(parts)
    word = np.repeat(np.arange(len(listed)), [len(part) for part in parts])
    inside = np.zeros(len(docs), dtype=np.bool_)
    inside[mine] = True
    rank = np.cumsum(inside) - 1
    keep = inside[at]
    sub = csr_matrix((np.ones(int(keep.sum())), (rank[at[keep]], word[keep])), shape=(len(mine), len(listed)))
    return [{"hub": int(hub), "members": [int(m) for m in members]} for hub, members in hub_groups(sub, docs[mine])]


def _first(key, among, k):
    """The k of among first by key, then by among: among[np.lexsort((among, key))][:k], without sorting them all."""
    if len(key) > k and not np.isnan(key).any():
        near = key <= np.partition(key, k - 1)[k - 1]
        key, among = key[near], among[near]
    return among[np.lexsort((among, key))][:k]


def _examples(node, root, Xall, weights, posteriors, margin, mine, first_hit, word, hidden=None):
    """The sentences most clearly in a sense, with the words that decided it: for a split sense, those with the
    largest margin, and for each the words whose (weighted) evidence most favoured its side along the way; for the
    whole query, those with the most evidence of its collocates. hidden: for each group of words, None or a mask of
    its words not shown."""
    path = _path(root, node)
    if path:
        top = _first(-margin[mine, node.id], mine, _MAX_EXAMPLES)
    else:  # the whole query: the sentences with the most evidence of its collocates against the corpus
        score = sum(
            X @ ((1.0 if d.weights is None else d.weights) * np.log(d.p / parent.p))
            for X, d, parent in zip(Xall, node.dists, node.parent_dists)
        )
        top = _first(-score, np.arange(len(score)), _MAX_EXAMPLES)
    # The evidence of every word of the sentences, all at once: of each, in order, the groups' words in order
    sentence, group, local, evidence = [], [], [], []
    for ci, X in enumerate(Xall):
        lo, hi = X.indptr[top].astype(np.int64), X.indptr[top + 1].astype(np.int64)
        lengths = hi - lo
        at = np.arange(int(lengths.sum())) + np.repeat(lo - np.cumsum(lengths) + lengths, lengths)
        indices, data = X.indices[at], X.data[at]
        w = 1.0 if weights[ci] is None else weights[ci][indices]
        contribution = np.zeros(len(at))
        if path:
            for split_node, k in path:
                dists = split_node.model[1][ci]
                contribution += data * w * (np.log(dists[k].p[indices]) - np.log(dists[1 - k].p[indices]))
        else:
            contribution = data * w * np.log(node.dists[ci].p[indices] / node.parent_dists[ci].p[indices])
        keep = contribution > 0
        if hidden is not None and hidden[ci] is not None:
            keep &= ~hidden[ci][indices]
        sentence.append(np.repeat(np.arange(len(top)), lengths)[keep])
        group.append(np.full(int(keep.sum()), ci))
        local.append(indices[keep])
        evidence.append(contribution[keep])
    sentence, group, local, evidence = (np.concatenate(a) for a in (sentence, group, local, evidence))
    order = np.lexsort((np.arange(len(sentence)), -evidence, sentence))  # the strongest first, as found among equals
    starts = np.searchsorted(sentence[order], np.arange(len(top) + 1))
    examples = []
    for s, u in enumerate(top):
        strongest = order[starts[s] : starts[s + 1]][:8]
        examples.append(
            {
                "hit": int(first_hit[u]),
                "p": round(float(posteriors[u, node.id]), 4),
                "evidence": [
                    {"word": word(int(ci), int(i)), "group": int(ci)}
                    for ci, i in zip(group[strongest], local[strongest])
                ],
            }
        )
    return examples
