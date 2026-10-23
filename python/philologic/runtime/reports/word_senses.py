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
Essen and Kneser 1994) to the corpus's. No list of words is filtered: in telling which sense a
sentence is in, each word's evidence weighs by its inverse document frequency, so that frequent words (je, vous, les,
de) count for less than rarer ones.

A split starts from the sentences with one of the query's collocates, the seed: of those whose sentences differ most
from the others in their other collocates, the most promising are tried (none of the collocation filter's words,
mostly function words: par, leur, mon tell the register of works more than a sense), and expectation-maximization
moves the
sentences to the sense they fit, each with a probability of being in either (first letting every collocate tell the
senses apart, since few words of the seed's sense are marked on its side yet, then only those that mark either,
chosen at every step, and held if that choice cycles). A split is kept if, fitted without each of N_FOLDS groups of
works in turn, it predicts the sentences of the works left out better than its unsplit sense, paying for saying which
sense each is in, by more than EVIDENCE standard errors over the sentences. Holding out whole works keeps the
sentences of a passage, which share their words, from predicting one another.

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
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numba
import numpy as np
import regex
from scipy.sparse import csr_matrix, hstack
from scipy.special import gammaincinv, ndtr

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
from philologic.runtime.word_properties import _nb_word_kept, property_masks
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
_LAMBDAS = 10.0 ** np.arange(-3.0, 9.01, 0.25)  # smoothness values generalized cross-validation chooses among
_MAX_WORDS = 40  # words sent for each sense and group of words (the response would be large otherwise)
_MAX_EXAMPLES = 50  # example sentences sent for each sense
_WORD_EXAMPLES = 10  # example sentences sent for each word of a sense: those of the sense with the word
_TEXT_WINDOW = 400  # bytes of a text read on either side of a hit, to reach the word next to it
_CACHE_TAG = b"senses27\0"  # bump when the result changes


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
    listed, g2 = markedly_more_frequent(c, parent_p)
    if dist.markers is not None:
        keep = dist.markers[listed]
        listed, g2 = listed[keep], g2[keep]
    if dist.weights is not None:  # the strongest evidence first, as the words weigh in telling senses apart
        order = np.argsort(-g2 * dist.weights[listed], kind="stable")
        listed, g2 = listed[order], g2[order]
    return [
        (int(i), float(c[i]), float(dist.p[i] / parent_p[i]), float(dist.p[i] / corpus_p[i]), float(g))
        for i, g in zip(listed, g2)
    ]


def describe_rarer(c, dist, parent_p):
    """The words markedly less frequent in a sense than in its parent, among those that may mark a sense: what a sense
    that is the rest of its parent lacks. (local index, count in the sense, ratio of its probability in it to its
    parent's, weighted G2), the strongest evidence first."""
    listed, g2 = markedly_more_frequent(c, parent_p, or_less=True)
    less = c[listed] < float(c.sum()) * parent_p[listed]
    listed, g2 = listed[less], g2[less]
    if dist.markers is not None:
        keep = dist.markers[listed]
        listed, g2 = listed[keep], g2[keep]
    if dist.weights is not None:
        g2 = g2 * dist.weights[listed]
    order = np.argsort(-g2, kind="stable")
    listed, g2 = listed[order], g2[order]
    return [(int(i), float(c[i]), float(dist.p[i] / parent_p[i]), float(g)) for i, g in zip(listed, g2)]


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
    total = float(c.sum())
    if total <= 0:
        return parent
    if own is None:
        own = marked(c, parent)
    if len(own) == 0:
        return parent
    kept = np.maximum(c[own] - parent.discount, 0.0) / total
    freed = float(np.minimum(c, parent.discount).sum()) / total  # the discount of every word the sense has
    p_own = kept + freed * parent.p[own]
    others = 1.0 - float(parent.p[own].sum())  # the parent's probability of every other word, the rest included
    scale = (1.0 - float(p_own.sum())) / others if others > 0 else 0.0
    p = parent.p * scale
    p[own] = p_own
    return Dist(p, parent.rest * scale, parent.markers, parent.weights, parent.discount)


def _split_words(c, parent, every_marker):
    """The words both senses of a split have their own rates for, from their counts c (words x 2): those that mark
    either, or every one of the parent's markers they have."""
    if every_marker:
        present = np.flatnonzero(c.sum(axis=1) > 0)
        return present if parent.markers is None else present[parent.markers[present]]
    return np.union1d(marked(c[:, 0], parent), marked(c[:, 1], parent))


# The EM of a split, compiled: the words of all groups in one vocabulary (each group's words after the last group's),
# the sentences' matrices in CSR. It computes what fit_sparse, _split_words and _mixture_loglik do, in the same order
# (numpy's pairwise sums included), so that the results are the same to the last bit; logarithms and exponentials are
# left to numpy, whose own implementations differ from the C library's in the last bit. Its iterations only follow the
# words the sentences have: those of a sense's sentences are a fraction of the query's.


class _Mats(list):
    """The matrices of the groups of words of the same sentences, packed once for the compiled EM: the words the
    sentences have, in one vocabulary (cols, their index in the vocabulary of all words; pos, among their group's
    words; each group's from cmp_off), indices into it, and whether each group's counts are all ones (whose counts are
    then not read: one times a number is that number)."""

    def __init__(self, Xs):
        super().__init__(csr_matrix(X) for X in Xs)
        n = self[0].shape[0]
        n_groups = len(self)
        voc_off = np.zeros(n_groups + 1, dtype=np.int64)
        cmp_off = np.zeros(n_groups + 1, dtype=np.int64)
        nnz_off = np.zeros(n_groups + 1, dtype=np.int64)
        had = []
        for c, X in enumerate(self):
            had.append(np.flatnonzero(np.bincount(X.indices[: X.nnz], minlength=X.shape[1])))
            voc_off[c + 1] = voc_off[c] + X.shape[1]
            cmp_off[c + 1] = cmp_off[c] + len(had[c])
            nnz_off[c + 1] = nnz_off[c] + X.nnz
        indptr = np.empty((n_groups, n + 1), dtype=np.int64)
        indices = np.empty(nnz_off[-1], dtype=np.int32)
        data = np.empty(nnz_off[-1], dtype=np.float64)
        ones = np.empty(n_groups, dtype=np.bool_)
        for c, X in enumerate(self):
            to_cmp = np.zeros(X.shape[1], dtype=np.int32)
            to_cmp[had[c]] = np.arange(cmp_off[c], cmp_off[c + 1], dtype=np.int32)
            indptr[c] = X.indptr + nnz_off[c]
            indices[nnz_off[c] : nnz_off[c + 1]] = to_cmp[X.indices[: X.nnz]]
            ones[c] = bool(np.all(X.data[: X.nnz] == 1))
            if not ones[c]:
                data[nnz_off[c] : nnz_off[c + 1]] = X.data[: X.nnz]
        pos = np.concatenate(had).astype(np.int64)
        cols = pos + np.repeat(voc_off[:-1], np.diff(cmp_off))
        self.ones = ones
        self.rows = _row_slices(indptr, indices, data, ones, len(cols))
        self.columns = _column_slices(indptr, indices, data, ones, cmp_off)
        self.words = (pos, cols, cmp_off, voc_off)
        self._marking = []

    def marking(self, parent):
        """The markers of the groups of words (all of them; their indices; each group's first; their places among the
        words the sentences have, -1 for none) and the weights of the words the sentences have (and the padding's): the
        same for every sense of a tree, its root's, so made once."""
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
        weights_had = np.repeat(np.append(weights[cols], 1.0), 2).reshape(-1, 2)  # both senses', for numpy's fast loop
        made = (markers, mk, np.searchsorted(mk, voc_off), had[mk], weights_had)
        self._marking.append((key, made))
        return made


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
def _row_slices(indptr, indices, data, ones, pad):
    """The entries of each group of the sentences in slices of sentences: the sentences of each slice (order, groups x
    n, by number of entries within each _WINDOW), the start of each slice's entries (groups x slices + 1), the entries t
    of its sentences stored together (idx, the word pad for padding, and their counts, sdata, for the groups not all
    ones)."""
    n_groups = indptr.shape[0]
    n = indptr.shape[1] - 1
    n_slices = (n + _WIDTH - 1) // _WIDTH
    order = np.empty((n_groups, n), dtype=np.uint64)
    start = np.empty((n_groups, n_slices + 1), dtype=np.uint64)
    lengths = np.empty((n_groups, n), dtype=np.int64)
    total = 0
    for c in range(n_groups):
        for i in range(n):
            lengths[c, i] = indptr[c, i + 1] - indptr[c, i]
        for w in range(0, n, _WINDOW):
            by_length = _by_length(lengths[c, w : w + _WINDOW])
            for q in range(len(by_length)):
                order[c, w + q] = w + by_length[q]
        for sl in range(n_slices):
            start[c, sl] = total
            longest = 0
            for q in range(sl * _WIDTH, min(n, (sl + 1) * _WIDTH)):
                longest = max(longest, lengths[c, order[c, q]])
            total += _WIDTH * longest
        start[c, n_slices] = total
    idx = np.full(total, pad, dtype=np.uint32)
    sdata = np.zeros(total)
    for c in range(n_groups):
        for q in range(n):
            i = order[c, q]
            base = start[c, q // _WIDTH] + q % _WIDTH
            for t in range(lengths[c, i]):
                idx[base + t * _WIDTH] = indices[indptr[c, i] + t]
                if not ones[c]:
                    sdata[base + t * _WIDTH] = data[indptr[c, i] + t]
    return order, start, idx, sdata


@numba.njit(cache=True, nogil=True)
def _column_slices(indptr, indices, data, ones, cmp_off):
    """The entries of the words the sentences have in slices of words, each group's apart: the words of each slice
    (order, _WIDTH a slice, the word after the last for none), the start of each slice's entries, the sentences of
    each word's entries, in order, stored together (idx, the sentence after the last for padding, and the entries'
    counts, cdata, for the groups not all ones), and each group's first slice (g_off)."""
    n_groups = indptr.shape[0]
    n = indptr.shape[1] - 1
    n_cols = cmp_off[n_groups]
    lengths = np.zeros(n_cols, dtype=np.int64)
    for c in range(n_groups):
        for jj in range(indptr[c, 0], indptr[c, n]):
            lengths[indices[jj]] += 1
    g_off = np.zeros(n_groups + 1, dtype=np.int64)
    for c in range(n_groups):
        g_off[c + 1] = g_off[c] + (cmp_off[c + 1] - cmp_off[c] + _WIDTH - 1) // _WIDTH
    order = np.full(_WIDTH * g_off[n_groups], n_cols, dtype=np.uint64)
    slot = np.empty(n_cols, dtype=np.int64)  # each word's place among the slices' words
    for c in range(n_groups):
        by_length = _by_length(lengths[cmp_off[c] : cmp_off[c + 1]])
        for q in range(len(by_length)):
            j = cmp_off[c] + by_length[q]
            order[_WIDTH * g_off[c] + q] = j
            slot[j] = _WIDTH * g_off[c] + q
    start = np.empty(g_off[n_groups] + 1, dtype=np.uint64)
    total = 0
    for sl in range(g_off[n_groups]):
        start[sl] = total
        first = order[_WIDTH * sl]
        total += _WIDTH * (lengths[first] if first < n_cols else 0)
    start[g_off[n_groups]] = total
    idx = np.full(total, n, dtype=np.uint32)
    cdata = np.zeros(total)
    at = np.empty(n_cols, dtype=np.int64)  # where each word's next entry goes
    for j in range(n_cols):
        at[j] = start[slot[j] // _WIDTH] + slot[j] % _WIDTH
    for c in range(n_groups):
        for i in range(n):
            for jj in range(indptr[c, i], indptr[c, i + 1]):
                j = indices[jj]
                idx[at[j]] = i
                if not ones[c]:
                    cdata[at[j]] = data[jj]
                at[j] += _WIDTH
    return order, start, idx, cdata, g_off


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


@numba.njit(cache=True, nogil=True)
def _block_sums_at(counts, d, pos, e, lo, n, acc, out):
    """_block_sum of the block [lo, lo + n) of the counts of either sense and of their least with d (out, 4 sums), the
    counts zero but at the positions pos, from entry e; returns the first entry after the block. acc: 4 x 8."""
    end = len(pos)
    if n < 8:
        out[:] = 0.0
        while e < end and pos[e] < lo + n:
            c0 = counts[e, 0]
            c1 = counts[e, 1]
            out[0] += c0
            out[1] += c1
            out[2] += min(c0, d)
            out[3] += min(c1, d)
            e += 1
        return e
    acc[:, :] = 0.0
    m = lo + n - (n % 8)
    while e < end and pos[e] < m:
        u = (pos[e] - lo) & 7
        c0 = counts[e, 0]
        c1 = counts[e, 1]
        acc[0, u] += c0
        acc[1, u] += c1
        acc[2, u] += min(c0, d)
        acc[3, u] += min(c1, d)
        e += 1
    for v in range(4):
        first = (acc[v, 0] + acc[v, 1]) + (acc[v, 2] + acc[v, 3])
        out[v] = first + ((acc[v, 4] + acc[v, 5]) + (acc[v, 6] + acc[v, 7]))
    while e < end and pos[e] < lo + n:
        c0 = counts[e, 0]
        c1 = counts[e, 1]
        out[0] += c0
        out[1] += c1
        out[2] += min(c0, d)
        out[3] += min(c1, d)
        e += 1
    return e


@numba.njit(cache=True, nogil=True)
def _sums_at(counts, d, pos, n, out):
    """_pairwise_sum, to the last bit, of the n counts of either sense (out[0], out[1]) and of their least with d
    (out[2], out[3]), the counts (none negative) zero but at the positions pos (increasing): in the time of their
    entries, as adding zero changes no sum, so that a part with no entry, zero, is not summed."""
    end = len(pos)
    acc = np.empty((4, 8))
    val = np.empty(4)
    if n <= 128:
        _block_sums_at(counts, d, pos, 0, 0, n, acc, val)
        for v in range(4):
            out[v] = 0.0 + val[v]
        return
    st_lo = np.empty(64, dtype=np.int64)
    st_n = np.empty(64, dtype=np.int64)
    st_stage = np.zeros(64, dtype=np.int64)  # as in _pairwise_sum
    st_first = np.empty((64, 4))
    top = 0
    st_lo[0] = 0
    st_n[0] = n
    e = 0
    while True:
        empty = e == end or pos[e] >= st_lo[top] + st_n[top]
        if st_n[top] <= 128 or empty:
            if empty:
                val[:] = 0.0
            else:
                e = _block_sums_at(counts, d, pos, e, st_lo[top], st_n[top], acc, val)
            top -= 1
            while top >= 0:
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
                for v in range(4):
                    val[v] = st_first[top, v] + val[v]
                top -= 1
            if top < 0:
                for v in range(4):
                    out[v] = 0.0 + val[v]
                return
        else:
            st_stage[top] = 1
            half = st_n[top] // 2
            half -= half % 8
            st_lo[top + 1] = st_lo[top]
            st_n[top + 1] = half
            st_stage[top + 1] = 0
            top += 1


@numba.njit(cache=True, nogil=True)
def _em_counts(order, start, idx, cdata, g_off, ones, r, n_cols):
    """Each word's (soft) count in either sense (X.T @ r for each group), for the n_cols words the sentences have (and
    one more, for the slices' empty places), from the slices of the words' entries: each word's sentences in order,
    from 0. r: n + 1 x 2, flat (the padding's last, 0). Returns n_cols + 1 x 2."""
    counts = np.zeros(2 * (n_cols + 1))
    one, two, three, four = np.uint64(1), np.uint64(2), np.uint64(3), np.uint64(4)
    for c in range(len(ones)):
        for sl in range(g_off[c], g_off[c + 1]):
            a0 = 0.0
            b0 = 0.0
            a1 = 0.0
            b1 = 0.0
            a2 = 0.0
            b2 = 0.0
            a3 = 0.0
            b3 = 0.0
            e = start[sl]
            end = start[sl + 1]
            if ones[c]:
                while e < end:
                    i = np.uint64(idx[e]) * two
                    a0 += r[i]
                    b0 += r[i + one]
                    i = np.uint64(idx[e + one]) * two
                    a1 += r[i]
                    b1 += r[i + one]
                    i = np.uint64(idx[e + two]) * two
                    a2 += r[i]
                    b2 += r[i + one]
                    i = np.uint64(idx[e + three]) * two
                    a3 += r[i]
                    b3 += r[i + one]
                    e += four
            else:
                while e < end:
                    i = np.uint64(idx[e]) * two
                    a0 += cdata[e] * r[i]
                    b0 += cdata[e] * r[i + one]
                    i = np.uint64(idx[e + one]) * two
                    a1 += cdata[e + one] * r[i]
                    b1 += cdata[e + one] * r[i + one]
                    i = np.uint64(idx[e + two]) * two
                    a2 += cdata[e + two] * r[i]
                    b2 += cdata[e + two] * r[i + one]
                    i = np.uint64(idx[e + three]) * two
                    a3 += cdata[e + three] * r[i]
                    b3 += cdata[e + three] * r[i + one]
                    e += four
            q = _WIDTH * sl
            j = order[q] * two
            counts[j] = a0
            counts[j + one] = b0
            j = order[q + 1] * two
            counts[j] = a1
            counts[j + one] = b1
            j = order[q + 2] * two
            counts[j] = a2
            counts[j + one] = b2
            j = order[q + 3] * two
            counts[j] = a3
            counts[j + one] = b3
    return counts.reshape(n_cols + 1, 2)


@numba.njit(cache=True, nogil=True)
def _em_sums(counts, pos, cmp_off, voc_off, discount):
    """Each group's total count in either sense, and total of the counts' least with its discount, over all its words,
    as numpy sums them (groups x 2 each)."""
    n_groups = len(cmp_off) - 1
    totals = np.empty((n_groups, 2))
    least = np.empty((n_groups, 2))
    out = np.empty(4)
    for c in range(n_groups):
        a, b = cmp_off[c], cmp_off[c + 1]
        _sums_at(counts[a:b], discount[c], pos[a:b], voc_off[c + 1] - voc_off[c], out)
        totals[c, 0] = out[0]
        totals[c, 1] = out[1]
        least[c, 0] = out[2]
        least[c, 1] = out[3]
    return totals, least


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
def _em_fit_had(counts, totals, least, thresholds, parent_p, parent_had, cmp_off, mk, mk_cmp, mk_off, discount, every):
    """_em_fit for the words the sentences have, from their counts (counts), the groups' totals and sums of the counts'
    least with the discount over all their words (least), and their parent's probabilities (parent_had): their
    probabilities (words x 2, and 1 for the padding of the slices), whether any word tells the senses apart, and
    which of the markers have rates of their own (chosen). mk
    are the markers of each group (from mk_off), mk_cmp their place among the words the sentences have (-1 for none:
    such a word may still mark a sense, as rarer in both than in the parent)."""
    n_groups = len(cmp_off) - 1
    n_had = len(parent_had)
    p = np.empty((n_had + 1, 2))
    p[n_had] = 1.0  # the padding of the slices of the sentences' entries: its evidence is 0
    any_own = False
    chosen = np.zeros(len(mk), dtype=np.bool_)
    own = np.empty(len(mk), dtype=np.int64)
    own_cmp = np.empty(len(mk), dtype=np.int64)
    for c in range(n_groups):
        n_own = 0
        for m in range(mk_off[c], mk_off[c + 1]):
            j = mk[m]
            q = mk_cmp[m]
            c0 = counts[q, 0] if q >= 0 else 0.0
            c1 = counts[q, 1] if q >= 0 else 0.0
            is_own = False
            if every:
                is_own = c0 + c1 > 0
            else:
                for k in range(2):
                    t = totals[c, k]
                    if t <= 0:
                        continue
                    co = c0 if k == 0 else c1
                    eo = t * parent_p[j]
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
                own[n_own] = j
                own_cmp[n_own] = q
                n_own += 1
        any_own = any_own or n_own > 0
        a, b = cmp_off[c], cmp_off[c + 1]
        d = discount[c]
        for k in range(2):
            t = totals[c, k]
            if t <= 0 or n_own == 0:
                for q in range(a, b):
                    p[q, k] = parent_had[q]
                continue
            freed = least[c, k] / t
            p_own = np.empty(n_own)
            parent_own = np.empty(n_own)
            for m in range(n_own):
                j = own[m]
                co = counts[own_cmp[m], k] if own_cmp[m] >= 0 else 0.0
                p_own[m] = max(co - d, 0.0) / t + freed * parent_p[j]
                parent_own[m] = parent_p[j]
            others = 1.0 - _pairwise_sum(parent_own, 0, n_own)
            scale = (1.0 - _pairwise_sum(p_own, 0, n_own)) / others if others > 0 else 0.0
            for q in range(a, b):
                p[q, k] = parent_had[q] * scale
            for m in range(n_own):
                if own_cmp[m] >= 0:
                    p[own_cmp[m], k] = p_own[m]
    return p, any_own, chosen


@numba.njit(cache=True, nogil=True)
def _em_loglik(order, start, idx, sdata, ones, evidence, log_pi):
    """Each sentence's (weighted) log-probability under either sense of a split, with its share: n x 2, from the slices
    of the sentences' entries. evidence: words x 2, flat, the padding's last (0). Each sentence's sum of each group
    still adds its entries in order, from 0, and the groups' sums in order: _WIDTH sentences' at once."""
    n_groups, n = order.shape
    ll = np.empty(2 * n)
    for i in range(n):
        ll[2 * i] = log_pi[0]
        ll[2 * i + 1] = log_pi[1]
    one, two, three, four = np.uint64(1), np.uint64(2), np.uint64(3), np.uint64(4)
    for c in range(n_groups):
        for sl in range(start.shape[1] - 1):
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
            if q + 1 < n:
                i = order[c, q + 1] * two
                ll[i] += a1
                ll[i + one] += b1
            if q + 2 < n:
                i = order[c, q + 2] * two
                ll[i] += a2
                ll[i + one] += b2
            if q + 3 < n:
                i = order[c, q + 3] * two
                ll[i] += a3
                ll[i + one] += b3
    return ll.reshape(n, 2)


@numba.njit(cache=True, nogil=True)
def _lse_join_less(ll, s, hi, log2, r):
    """_lse_join, and r = ll - s[:, None] and the total of s (as numpy sums it), in one pass: (whether any of s is not
    finite, the total; r and the total then to be made again)."""
    bad = False
    for i in range(ll.shape[0]):
        if ll[i, 0] == ll[i, 1]:
            s[i] = log2 + hi[i]
        else:
            s[i] = s[i] + hi[i]
        bad = bad or not np.isfinite(s[i])
        r[i, 0] = ll[i, 0] - s[i]
        r[i, 1] = ll[i, 1] - s[i]
    return bad, (0.0 if bad else _pairwise_sum(s, 0, len(s)))


@numba.njit(cache=True, nogil=True)
def _less(ll, total, out):
    """ll - total[:, None], into out."""
    for i in range(ll.shape[0]):
        out[i, 0] = ll[i, 0] - total[i]
        out[i, 1] = ll[i, 1] - total[i]


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


def _em(Xs, parent, r, every_marker, switch=True):
    """The EM of a split (see fit_split). The words chosen at every step (every_marker false), once the set chosen comes
    back to one chosen before, after another, are held from then on (switch): the choice cycles, and the fit would not
    settle."""
    if not isinstance(Xs, _Mats):
        Xs = _Mats(Xs)
    ones = Xs.ones
    order, start, idx, sdata = Xs.rows
    c_order, c_start, c_idx, cdata, g_off = Xs.columns
    pos, cols, cmp_off, voc_off = Xs.words
    n = Xs[0].shape[0]
    n_words = int(voc_off[-1])
    parent_p = np.concatenate([d.p for d in parent]).astype(np.float64)
    parent_rest = np.array([d.rest for d in parent], dtype=np.float64)
    discount = np.array([d.discount for d in parent], dtype=np.float64)
    markers, mk, mk_off, mk_cmp, weights_had = Xs.marking(parent)
    parent_had = parent_p[cols]
    weight = _weights(r)
    padded = np.zeros((n + 1, 2))  # r, and the padding's (0)
    padded[:n] = r
    r = padded[:n]
    lse, hi, log2 = np.empty(n), np.empty(n), 0.0 + math.log(2.0)
    previous = None
    seen, last = set(), None  # the sets of words chosen
    for _ in range(_MAX_ITER):
        if weight.min() < 1.0:
            return None
        pi = weight / n
        counts = _em_counts(c_order, c_start, c_idx, cdata, g_off, ones, padded.reshape(-1), len(cols))
        totals, least = _em_sums(counts, pos, cmp_off, voc_off, discount)
        with np.errstate(divide="ignore"):
            thresholds = np.log(totals)
        p, any_own, chosen = _em_fit_had(
            counts, totals, least, thresholds, parent_p, parent_had, cmp_off, mk, mk_cmp, mk_off, discount, every_marker
        )
        if not any_own:  # no word tells the senses apart: both are the parent, no split
            return None
        if switch and not every_marker:
            key = chosen.tobytes()
            if key in seen and key != last:  # the choice cycles: these words, held from now on
                mk, mk_cmp = mk[chosen], mk_cmp[chosen]
                mk_off = np.searchsorted(mk, voc_off)
                markers = np.zeros(n_words, dtype=np.bool_)
                markers[mk] = True
                every_marker = True
            seen.add(key)
            last = key
        evidence = np.log(p, out=p)
        np.multiply(evidence, weights_had, out=evidence)
        ll = _em_loglik(order, start, idx, sdata, ones, evidence.reshape(-1), np.log(pi))
        # r = exp(ll - logsumexp(ll)): _logsumexp2's arithmetic, its passes joined
        _lse_apart(ll, lse, hi)
        with np.errstate(invalid="ignore", over="ignore"):
            np.exp(lse, out=lse)
        np.log1p(lse, out=lse)
        bad, total = _lse_join_less(ll, lse, hi, log2, r)
        if bad:
            lse = _logsumexp2(ll)
            _less(ll, lse, r)
            total = _pairwise_sum(lse, 0, n)
        np.exp(r, out=r)
        if previous is not None and abs(total - previous) <= _TOL * abs(total):
            break
        previous = total
        weight = _weights(r)
    if _weights(r).min() < 1.0:
        return None
    # The last fit again, for every word
    counts_had, counts = counts, np.zeros((n_words, 2))
    counts[cols] = counts_had[: len(cols)]
    p, rest, is_parent, _ = _em_fit(
        counts, totals, thresholds, parent_p, parent_rest, markers, discount, voc_off, every_marker
    )
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
    return pi, dists, r, [counts[voc_off[c] : voc_off[c + 1]].copy() for c in range(len(parent))]


def fit_split(Xs, parent, r, search=False):
    """Fit a split of the sentences Xs (one matrix for each group of words) into two senses by expectation-maximization,
    from the probabilities r (n x 2) of each sentence being in either, each sense a sparse departure from the parent:
    both have their own rates for the words that mark either, and share the parent's for all others. A search (from a
    seed) first lets every one of the parent's markers tell the senses apart: few of the words of the seed's sense are
    marked yet on its side, and the sense would not grow. The words that mark either are chosen again at every step;
    when the choice comes back to a set it made before, after another, that set is held: such a fit cycles, and never
    settled (stopped by _MAX_ITER). Returns (pi, dists, r, counts), or None when a sense ends up with less than a
    sentence or none of the words tells them apart."""
    if search:
        first = _em(Xs, parent, r, every_marker=True)
        if first is None:
            return None
        r = first[2]
    return _em(Xs, parent, r, every_marker=False)


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


def _fold_data(Xs, unsplit_fit, groups, pool_map=map):
    """The folds a split is tested on, the same for every split of a sense: for each, its training and held-out
    sentences (masks and matrices), the unsplit sense fitted on the training ones and the held-out sentences'
    log-probability under it. Whole works are held out; a single work's sentences, in turn."""
    n = Xs[0].shape[0]
    folds, n_folds = work_folds(groups)
    if folds is None:  # a single work: its sentences then
        folds, n_folds = np.arange(n) % N_FOLDS, min(N_FOLDS, n)

    def fold(f):
        train = folds != f
        test = ~train
        Xtrain = _Mats([X[train] for X in Xs])
        Xtest = [X[test] for X in Xs]
        unsplit = unsplit_fit([np.asarray(X.sum(axis=0)).ravel() for X in Xtrain])
        return train, test, Xtrain, Xtest, unsplit, _loglik(Xtest, unsplit)

    return list(pool_map(fold, range(n_folds)))


def _fold_gain(fold, r):
    """The split started from the probabilities r, fitted on a fold's training sentences: each held-out sentence's
    gain over the unsplit sense, and the gains of those more likely in either sense; None for no evidence."""
    train, _, Xtrain, Xtest, unsplit, unsplit_ll = fold
    fitted = fit_split(Xtrain, unsplit, r[train])
    if fitted is None:  # without these sentences, nothing can be told apart: no better
        return None
    pi, dists, _, _ = fitted
    ll = _mixture_loglik(Xtest, pi, dists)
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


def test_split(Xs, unsplit_fit, r, groups, folds=None, pool_map=map):
    """Whether the split predicts the sentences of the works it was not fitted on better than its unsplit sense: in
    each of N_FOLDS folds of the works, both are fitted on the other folds and scored on its sentences, the split paying
    for saying which sense each is in. Holding out whole works keeps the sentences of a passage, which share their
    words, from predicting one another. Sentences are the units of evidence: each gains or loses, and the mean of their
    gains, in nats, is compared with its standard error."""
    if folds is None:
        folds = _fold_data(Xs, unsplit_fit, groups, pool_map)
    return _test_result(folds, list(pool_map(lambda fold: _fold_gain(fold, r), folds)), Xs[0].shape[0])


def induce_senses(channels, groups, finest=FINEST, unseeded=None):
    """Grow the tree of senses of the sentences: channels are (X, corpus probabilities, corpus rest[, weights]) for each
    group of words, X a sentences x words matrix of counts, weights how much each word's evidence counts; groups is
    the work of each sentence; unseeded, for each group of words, None or a mask of its words that seed no split
    (they still mark senses). Returns the root Node; split senses have children, the others are the senses found.
    Each split has its importance: how much it improves the prediction of the words around the query, per use of the
    word (its sense's share of the sentences times its gain per sentence there). Splits less important than finest
    times the first one are not looked for."""
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

    floor = {"importance": 0.0}

    def grow(node, pool_map):
        if node is root:
            unsplit_fit = root_fit
        else:
            parents = node.parent_dists
            unsplit_fit = lambda train_counts, parents=parents: [
                fit_sparse(c, d) for c, d in zip(train_counts, parents)
            ]
        _try_split(node, Xall, n, unsplit_fit, floor, groups, seeds, pool_map)
        return node

    def grow_with(pool_map, senses=None):
        grow(root, pool_map)
        if root.children:
            floor["importance"] = finest * root.test["importance"]
        if senses is None:
            queue = list(root.children)
            while queue:
                queue.extend(grow(queue.pop(0), pool_map).children)
            return
        pending = {senses.submit(grow, child, pool_map) for child in root.children}
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending |= {senses.submit(grow, child, pool_map) for child in future.result().children}

    # The seeds of a split and the folds of each are fitted at once, and the senses below the first split grow at once
    # too (they are independent of one another once the first sets the floor): the compiled EM and numpy release the
    # GIL. As many threads work as numba may use (NUMBA_NUM_THREADS, which the web server caps for each of its workers,
    # as for the rest of the app): those that grow the senses only hand their work to them.
    workers = numba.config.NUMBA_NUM_THREADS
    if workers < 2:
        grow_with(map)
        return root
    with ThreadPoolExecutor(max_workers=workers) as fits, ThreadPoolExecutor(max_workers=workers) as senses:
        grow_with(fits.map, senses)
    return root


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


def marks(node, against, common=None):
    """Whether a sense has something of its own to show: a word markedly more frequent in it than against (the dists
    of the sense it is told from) in any group of words, outside common (for each group, None or a mask of the words
    that alone make no sense)."""
    for ci, (c, dist) in enumerate(zip(node.counts, node.dists)):
        mask = None if common is None else common[ci]
        for i, *_ in describe(c, dist, against[ci].p, against[ci].p):
            if mask is None or not mask[i]:
                return True
    return False


def _try_split(node, Xall, n_all, unsplit_fit, floor, groups, collocates, pool_map=map):
    units = node.units
    if len(units) < 2:
        node.reason = "inseparable"
        return

    def prepare(_):
        Xs = _Mats([X[units] for X in Xall])
        return Xs, seed_splits(Xs, collocates, weights=[d.weights for d in node.dists])

    ((Xs, seeds),) = pool_map(prepare, (None,))  # on the pool's threads too, which alone do the work
    candidates = []
    for (_, seed), fitted in zip(seeds, pool_map(lambda s: fit_split(Xs, node.dists, s[0], search=True), seeds)):
        if fitted is None:
            continue
        side = fitted[2][:, 0] >= fitted[2][:, 1]
        if side.all() or not side.any():  # one sense is in every sentence, more or less: no halves
            continue
        candidates.append((seed, fitted, side))
    best = None
    if candidates:
        folds = _fold_data(Xs, unsplit_fit, groups[units], pool_map)
        tasks = [(fitted[2], fold) for _, fitted, _ in candidates for fold in folds]
        results = list(pool_map(lambda task: _fold_gain(task[1], task[0]), tasks))
    for c, (seed, fitted, side) in enumerate(candidates):
        pi, dists, r, counts = fitted
        test = _test_result(folds, results[c * len(folds) : (c + 1) * len(folds)], len(units))
        test["seed"] = seed
        test["importance"] = float(len(units) / n_all * max(0.0, test["bits_per_sentence"]))
        passes = test["z"] > EVIDENCE and test["importance"] >= floor["importance"]
        if best is None or (passes, test["importance"]) > (best[0], best[1]["importance"]):
            best = (passes, test, pi, dists, side, counts)
    if best is None:
        node.reason = "inseparable"
        return
    passes, node.test, pi, dists, side, counts = best
    if not passes:
        node.reason = "no_gain" if node.test["z"] <= EVIDENCE else "faint"
        return
    node.model = (pi, dists)
    for k, members in ((0, side), (1, ~side)):
        child = Node(units[members], [d[k] for d in dists], [c[:, k] for c in counts], node.dists)
        child.k = k
        node.children.append(child)
    node.children.sort(key=lambda c: (-len(c.units), c.k))


def node_posteriors(root, Xall):
    """Every sense of the tree, in order (each before those it splits into), given its id and parent; and for every
    sentence the log-probability of each sense (n x senses), and its margin: the smallest log-likelihood ratio, along
    the way to the sense, of the sense's side over the other (positive when the sentence is on its side of every
    split, infinite for the whole query)."""
    n = Xall[0].shape[0]
    nodes, logp, margin = [], [], []

    def walk(node, lp, mg, parent):
        node.id = len(nodes)
        node.parent = parent
        nodes.append(node)
        logp.append(lp)
        margin.append(mg)
        if not node.children:
            return
        pi, dists = node.model
        ll = _mixture_loglik(Xall, pi, dists)
        cond = ll - _logsumexp2(ll)[:, None]
        for child in node.children:
            walk(child, lp + cond[:, child.k], np.minimum(mg, ll[:, child.k] - ll[:, 1 - child.k]), node)

    walk(root, np.zeros(n), np.full(n, np.inf), None)
    return nodes, np.column_stack(logp), np.column_stack(margin)


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
    work_s = np.zeros((n_works, P.shape[1]))
    np.add.at(work_s, work_of, P)

    def totals(m):
        n = np.bincount(work_year, weights=m * work_n, minlength=n_years)
        y = np.array([np.bincount(work_year, weights=m * work_s[:, k], minlength=n_years) for k in range(P.shape[1])])
        return n, y, np.bincount(work_year, weights=m, minlength=n_years)

    n, y, works_in = totals(np.ones(n_works))
    rng = np.random.default_rng(0)  # fixed: the same bands every time
    resamples = rng.multinomial(n_works, np.full(n_works, 1.0 / n_works), size=N_RESAMPLES).astype(np.float64)
    boot = [totals(m) for m in resamples]
    boot_n = np.array([b[0] for b in boot])
    boot_y = np.array([b[1] for b in boot])
    boot_works = np.array([b[2] for b in boot])

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
):
    """The words of each sentence (once each; within max_dist of a hit when max_dist >= 0; with a kept value of each
    word property of prop_columns, see word_properties.property_masks), and the words just before and just after each
    hit, as flat arrays and their bounds by sentence. The query's own words are left out, and the filtered ones."""
    n_units = len(unit_bounds) - 1
    total = 0
    for u in range(n_units):
        si = hit_sent[unit_bounds[u]]
        total += sent_offsets[si + 1] - sent_offsets[si]
    n_hits = len(hit_sent)
    bag = np.empty(total, dtype=np.int64)
    left = np.empty(n_hits, dtype=np.int64)
    right = np.empty(n_hits, dtype=np.int64)
    bag_ptr = np.zeros(n_units + 1, dtype=np.int64)
    left_ptr = np.zeros(n_units + 1, dtype=np.int64)
    right_ptr = np.zeros(n_units + 1, dtype=np.int64)
    seen = np.zeros(n_vocab, dtype=np.bool_)
    hashes = np.empty(n_units, dtype=np.uint64)
    nb = 0
    nl = 0
    nr = 0
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
    return bag[:nb], bag_ptr, left[:nl], left_ptr, right[:nr], right_ptr, hashes


def _channel(flat, ptr, n_units):
    """A sentences x words count matrix of local word indices, and the vocabulary ids of its words."""
    # np.unique(flat, return_inverse=True), by marking the ids rather than sorting the entries
    had = np.zeros(int(flat.max()) + 1 if len(flat) else 0, dtype=np.bool_)
    had[flat] = True
    ids = np.flatnonzero(had).astype(flat.dtype)
    local = np.zeros(len(had), dtype=np.int64)
    local[ids] = np.arange(len(ids))
    cols = local[flat]
    X = csr_matrix((np.ones(len(flat)), cols, ptr), shape=(n_units, len(ids)))
    X.sum_duplicates()
    return X, ids


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
    sent_keys = np.load(os.path.join(colloc_dir, "sent_keys_s24.npy"), mmap_mode="r")
    sent_offsets = np.asarray(np.load(os.path.join(colloc_dir, "sent_offsets.npy"), mmap_mode="r"))
    prefix = "attr_lemma_" if count_lemmas else ""
    count_ids = np.load(
        os.path.join(colloc_dir, "attr_lemma_ids.npy" if count_lemmas else "token_ids.npy"), mmap_mode="r"
    )
    v_blob, v_offsets = _vocabulary(data_path, count_lemmas)
    n_vocab = len(v_offsets) - 1

    keys = np.ascontiguousarray(rows[:, :6].astype(">u4")).view("S24").ravel()
    si = np.searchsorted(sent_keys, keys)
    n_sents = len(sent_keys)
    matched = (si < n_sents) & (sent_keys[np.clip(si, 0, n_sents - 1)] == keys)
    hit_index = np.flatnonzero(matched)
    if len(hit_index) == 0:
        return None
    si = si[hit_index].astype(np.int64)
    positions = rows[hit_index][:, 7::2].astype(np.int64)  # the word position of each of the hit's query words
    # The query's own words: whatever form or lemma is at its hits' positions, wherever it is in the sentence
    at = (sent_offsets[si].astype(np.int64)[:, None] + positions - 1)[positions > 0]
    excluded = np.zeros(n_vocab, dtype=np.bool_)
    excluded[np.unique(np.asarray(count_ids[at]))] = True
    filtered = _filter_mask(colloc_dir, prefix, filter_words, n_vocab)

    order = np.argsort(si, kind="stable")
    hit_index, si, positions = hit_index[order], si[order], positions[order]
    unit_bounds = np.append(np.flatnonzero(np.r_[True, si[1:] != si[:-1]]), len(si))
    prop_columns, prop_allowed = property_masks(data_path, properties)
    max_dist = distance + phrase_words(q, db.locals.query_patterns) if distance is not None else -1
    bag, bag_ptr, left, left_ptr, right, right_ptr, hashes = _sentence_words(
        unit_bounds,
        si,
        positions.min(axis=1).astype(np.uint32),
        positions.max(axis=1).astype(np.uint32),
        sent_offsets,
        count_ids,
        excluded,
        filtered,
        prop_columns,
        prop_allowed,
        np.int32(max_dist),
        n_vocab,
    )
    n_units = len(unit_bounds) - 1
    first = hit_index[unit_bounds[:-1]]  # the first hit of each sentence, in the hitlist
    # Identical sentences count once (repeated titles, running heads, reprints): they are one use, not many
    keep = np.sort(np.unique(hashes, return_index=True)[1])
    channels = []
    for flat, ptr in ((bag, bag_ptr), (left, left_ptr), (right, right_ptr)):
        X, ids = _channel(flat, ptr, n_units)
        X = X[keep]
        used = np.flatnonzero(X.getnnz(axis=0))  # the words the kept sentences still have
        channels.append((X[:, used], ids[used]))
    first = first[keep]
    # The words just before and after each hit as the text has them: their forms, and where the hit is in its text
    forms = np.load(os.path.join(colloc_dir, "token_ids.npy"), mmap_mode="r") if count_lemmas else count_ids
    s0 = sent_offsets[si].astype(np.int64)
    length = sent_offsets[si + 1].astype(np.int64) - s0
    lo, hi = positions.min(axis=1), positions.max(axis=1)
    kept_of = np.full(n_units, -1, dtype=np.int64)
    kept_of[keep] = np.arange(len(keep))
    unit_of = kept_of[np.searchsorted(unit_bounds, np.arange(len(si)), side="right") - 1]
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
                return json.load(f)
        except (OSError, ValueError):
            pass
    began = time.perf_counter()
    result = _word_senses(db, q, count_lemmas, properties, metadata, distance, year_field, filter_words)
    membership = result.pop("_membership", None)
    if membership is not None:
        _save_membership(_membership_path(path), membership)
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


def senses_with_membership(
    db,
    q,
    count_lemmas,
    properties=(),
    metadata=None,
    distance=None,
    year_field=None,
    filter_words=None,
):
    """The senses report (word_senses) and which of the query's sentences are clearly in each of its senses (on its
    side of every split): a sentences x senses array of booleans, kept next to the cached report, else found again."""
    report = word_senses(db, q, count_lemmas, properties, metadata, distance, year_field, filter_words)
    metadata = {k: v for k, v in (metadata or {}).items() if v}
    filter_words = sorted(set(filter_words or []))
    path = _membership_path(
        _cache_path(db, q, bool(count_lemmas), properties, metadata, distance, year_field, filter_words)
    )
    membership = _load_membership(path)
    if membership is None or membership.shape[1] != len(report["nodes"]):  # a report cached before its membership
        fresh = _word_senses(db, q, count_lemmas, properties, metadata, distance, year_field, filter_words)
        membership = fresh["_membership"]
        _save_membership(path, membership)
    return report, membership


def _membership_path(report_path):
    return report_path[: -len(".senses.json")] + ".membership.npz"


def _save_membership(path, membership):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, bits=np.packbits(membership, axis=0), shape=np.array(membership.shape))
        os.replace(tmp, path)
    except OSError:  # a cache that can't be written is no cache
        if os.path.exists(tmp):
            os.unlink(tmp)


def _load_membership(path):
    try:
        with np.load(path) as data:
            n, k = (int(x) for x in data["shape"])
            return np.unpackbits(data["bits"], axis=0, count=n).astype(np.bool_)[:, :k]
    except (OSError, ValueError, KeyError):
        return None


def _word_senses(db, q, count_lemmas, properties, metadata, distance, year_field, filter_words=None):
    # The analysis reads every word, rare ones weighing more: left out of it, the filter's words that tell a rare sense
    # (dieu for prière) would be too (see the pseudoword test). They only seed no split
    sentences = query_sentences(db, q, count_lemmas, properties, metadata, distance)
    if sentences is None:
        return {"n_hits": 0, "n_sentences": 0, "n_works": 0, "nodes": [], "time": None}
    channels, names = _senses_channels(sentences, count_lemmas)
    common = _filtered(sentences, filter_words)
    root = induce_senses(channels, sentences["docs"], unseeded=[common[ids] for ids in names])
    return _senses_report(db, sentences, channels, names, root, filter_words, metadata, year_field)


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

    def near(self, units):
        """For the sentences units: {group: {vocabulary id of a word next to the query: the row of its first hit with
        its most common forms}}."""
        out = {}
        if self.neighbors is None:
            return out
        inside = np.zeros(self.n_units, dtype=np.bool_)
        inside[units] = True
        for side in (1, 2):
            nb = self.neighbors[side]
            sel = np.flatnonzero(inside[nb["unit"]])
            if not len(sel):  # no word on this side of the query in these sentences
                out[side] = {}
                continue
            w, f, q, h = nb["word"][sel], nb["form"][sel], nb["query_form"][sel], nb["hit"][sel]
            order = np.lexsort((h, q, f, w))
            w, f, q, sel = w[order], f[order], q[order], sel[order]
            starts = np.flatnonzero(np.r_[True, (w[1:] != w[:-1]) | (f[1:] != f[:-1]) | (q[1:] != q[:-1])])
            counts = np.diff(np.r_[starts, len(w)])
            pick = np.lexsort((starts, -counts, w[starts]))  # for each word, its most common forms first
            words = w[starts][pick]
            first = np.r_[True, words[1:] != words[:-1]] if len(words) else np.zeros(0, dtype=np.bool_)
            out[side] = dict(zip(words[first].tolist(), sel[starts[pick][first]].tolist()))
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


def _senses_report(db, sentences, channels, names, root, filter_words, metadata, year_field):
    """The report of the tree of senses root, as JSON-ready data."""
    unshown = _filtered(sentences, filter_words)
    n = sentences["n_units"]
    docs = sentences["docs"]
    Xall = [csr_matrix(c[0]) for c in channels]
    nodes, logp, margin = node_posteriors(root, Xall)
    posteriors = np.exp(logp)
    weights = [c[3] for c in channels]
    corpus_p = [c[1] for c in channels]

    v_blob, v_offsets = sentences["v_blob"], sentences["v_offsets"]

    def shown(ci, local):  # the filter's words: not among a sense's words, but next to the query, where common words
        return ci > 0 or not unshown[int(names[ci][local])]  # make the construction (homme d'esprit, mon esprit)

    def word(ci, local):
        tid = int(names[ci][local])
        s = v_blob[int(v_offsets[tid]) : int(v_offsets[tid + 1])].decode("utf-8", errors="replace")
        return s[6:] if s.startswith("lemma:") else s

    top_importance = max((nd.test["importance"] for nd in nodes if nd.children), default=0.0)
    first_hit = sentences["first_hit"]
    listed_words = {}
    phrases = _Phrases(db, sentences)
    near_by_node = {}
    mine_of = {}
    columns = {}

    def sentences_of(node):  # its sentences: those on its side of every split on the way to it
        if node.id not in mine_of:
            mine_of[node.id] = np.flatnonzero(margin[:, node.id] > 0)
        return mine_of[node.id]

    def columns_of(ci):  # the sentences of each word of a group (_pattern_columns), made once
        if ci not in columns:
            columns[ci] = _pattern_columns(Xall[ci].indptr, Xall[ci].indices, Xall[ci].shape[1])
        return columns[ci]

    def near(node):  # the words next to the query in a sense's sentences, as the text has them (see _Phrases)
        if node.id not in near_by_node:
            near_by_node[node.id] = phrases.near(sentences_of(node))
        return near_by_node[node.id]

    asked = []  # the words of the senses, to which their sentences are added once all are known (with_hits)

    def with_hits(item, node, ci, i):
        asked.append((item, node.id, ci, i))
        return item

    def add_hits():
        """To each word of a sense, the sentences of the sense with it, the most clearly in the sense first (their
        first hits): from the sentences of each word, a sense at a time."""
        by_node = {}
        for a in asked:
            by_node.setdefault(a[1], []).append(a)
        for nid, items in by_node.items():
            inside = margin[:, nid] > 0
            clear_of = np.ascontiguousarray(posteriors[:, nid])
            for item, _, ci, i in items:
                col_ptr, rows = columns_of(ci)
                rows = rows[col_ptr[i] : col_ptr[i + 1]]
                rows = rows[inside[rows]]
                clear = clear_of[rows]
                if len(rows) > _WORD_EXAMPLES:  # the clearest only, without sorting them all
                    keep = np.argpartition(-clear, _WORD_EXAMPLES)[:_WORD_EXAMPLES]
                    rows, clear = rows[keep], clear[keep]
                item["hits"] = first_hit[rows[np.argsort(-clear, kind="stable")]].astype(np.int64).tolist()

    def with_phrase(item, node, ci, i):
        if ci > 0:
            phrase = phrases.text(ci, near(node).get(ci, {}).get(int(names[ci][i])))
            if phrase:
                item["phrase"] = phrase
        return item

    def described(node, against):
        """A sense's words, against the dists of the sense it is told from (as a rule, its parent): those markedly more
        frequent in it, and what it lacks."""
        groups_of_words = []
        hidden = []
        for ci in range(len(channels)):
            listed = describe(node.counts[ci], node.dists[ci], against[ci].p, corpus_p[ci])
            if ci == 0:
                hidden = [item for item in listed if not shown(ci, item[0])]
            listed = [item for item in listed if shown(ci, item[0])]
            if ci == 0:
                listed_words[node.id] = np.array([i for i, *_ in listed[:_MAX_WORDS]], dtype=np.int64)

            def entry(i, c, r, f):  # a word of the sense, as the page shows it
                item = {"word": word(ci, i), "n": round(c, 2), "ratio": round(r, 3), "lift": round(f, 3)}
                return with_hits(with_phrase(item, node, ci, i), node, ci, i)

            groups_of_words.append([entry(i, c, r, f) for i, c, r, f, _ in listed[:_MAX_WORDS]])
        # A sense with nothing else to show: the words of the filter that mark it
        if not any(groups_of_words):
            hidden = [
                {"word": word(0, i), "n": round(c, 2), "ratio": round(r, 3), "lift": round(lift, 3)}
                for i, c, r, lift, _ in hidden[:_MAX_WORDS]
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
                    for i, c, r, g in describe_rarer(node.counts[ci], node.dists[ci], against[ci].p)
                    if shown(ci, i)
                ),
                key=lambda item: -item[0],
            )
            rarer = [with_phrase(item, node.parent, *item.pop("at")) for _, item in rarer[:_MAX_WORDS]]
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
    marked = {node.id: marks(node, node.parent_dists, common) for node in nodes}
    out = []
    for node in nodes:
        split = None
        if node.test is not None:
            split = {k: v for k, v in node.test.items() if k != "gains"}
            seed = split.pop("seed", None)
            split["seed"] = None if seed is None else with_phrase({"word": word(*seed), "group": seed[0]}, node, *seed)
            split["relative"] = round(float(split["importance"] / top_importance), 4) if top_importance > 0 else 0.0
        mine = sentences_of(node)
        works_of = docs[mine]
        work_counts = np.unique(works_of, return_counts=True)[1] if len(mine) else np.zeros(1)
        out.append(
            {
                "id": node.id,
                "parent": None if node.parent is None else node.parent.id,
                "children": [child.id for child in node.children],
                "share": round(float(posteriors[:, node.id].mean()), 6),
                "sentences": int(len(mine)),
                **described(node, node.parent_dists),
                "split": split,
                "reason": node.reason,
                "works": int(len(work_counts)) if len(mine) else 0,
                "top_work_share": round(float(work_counts.max() / max(1, len(mine))), 4),
                "examples": _examples(node, root, Xall, weights, posteriors, margin, mine, first_hit, word, shown),
            }
        )

    timeline = None
    if year_field:
        try:
            year_array, year_words, _, _, _ = _get_doc_year_data(db, year_field)
            docs_ok = np.clip(docs, 0, len(year_array) - 1)
            years = np.where(docs < len(year_array), year_array[docs_ok], 0).astype(np.int64)
            if metadata:
                year_words, _ = _corpus_year_counts(db, year_array, metadata)
            timeline = sense_curves(posteriors, years, docs, year_words)
        except Exception:  # no year field the database can read: senses without their course over time
            timeline = None

    # The senses' numbers; a sense that goes on as its parent is told from the sense its number started in (the
    # rest of a sense, after one is carved out of it, has the words of that sense, not those of none)
    number, start = sense_numbers(out, lambda item: marked[item["id"]])
    rank = {n: k + 1 for k, n in enumerate(sorted(start, key=lambda n: [int(x) for x in n.split(".")]))}
    for node in nodes:
        sense = number[node.id]
        if sense is not None and start[sense] != node.id:
            out[node.id].update(described(node, nodes[start[sense]].parent_dists))
    for item in out:
        item["sense"] = number[item["id"]]
        item["rank"] = None if item["sense"] is None else rank[item["sense"]]
    add_hits()
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
        "_membership": margin > 0,  # which sentences are clearly in each sense: kept apart, not sent (word_senses)
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


def _examples(node, root, Xall, weights, posteriors, margin, mine, first_hit, word, shown=lambda ci, i: True):
    """The sentences most clearly in a sense, with the words that decided it: for a split sense, those with the
    largest margin, and for each the words whose (weighted) evidence most favoured its side along the way; for the
    whole query, those with the most evidence of its collocates."""
    path = _path(root, node)
    if path:
        top = mine[np.lexsort((mine, -margin[mine, node.id]))][:_MAX_EXAMPLES]
    else:  # the whole query: the sentences with the most evidence of its collocates against the corpus
        score = sum(
            X @ ((1.0 if d.weights is None else d.weights) * np.log(d.p / parent.p))
            for X, d, parent in zip(Xall, node.dists, node.parent_dists)
        )
        top = np.argsort(-score, kind="stable")[:_MAX_EXAMPLES]
    examples = []
    for u in top:
        evidence = {}
        for ci, X in enumerate(Xall):
            lo, hi = X.indptr[u], X.indptr[u + 1]  # the sentence's row, as getrow has it
            if hi == lo:
                continue
            indices, data = X.indices[lo:hi], X.data[lo:hi]
            w = 1.0 if weights[ci] is None else weights[ci][indices]
            contribution = np.zeros(hi - lo)
            if path:
                for split_node, k in path:
                    dists = split_node.model[1][ci]
                    contribution += data * w * (np.log(dists[k].p[indices]) - np.log(dists[1 - k].p[indices]))
            else:
                contribution = data * w * np.log(node.dists[ci].p[indices] / node.parent_dists[ci].p[indices])
            for i, c in zip(indices, contribution):
                if c > 0 and shown(ci, int(i)):
                    evidence[(ci, int(i))] = evidence.get((ci, int(i)), 0.0) + float(c)
        strongest = sorted(evidence.items(), key=lambda kv: -kv[1])[:8]
        examples.append(
            {
                "hit": int(first_hit[u]),
                "p": round(float(posteriors[u, node.id]), 4),
                "evidence": [{"word": word(ci, i), "group": ci} for (ci, i), _ in strongest],
            }
        )
    return examples
