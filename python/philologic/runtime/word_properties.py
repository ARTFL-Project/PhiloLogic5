"""Words with several properties at once: aimer:pos:VERB:tag:VINF, lemma:aimer:pos:VERB:tag:VINF.

The index has the hits of a word (or lemma) with one of its properties (aimer:pos:VERB), the collocation arrays the
properties of each word of the text, in text order. The hits of a word with several properties are those of its key
with one property that has the fewest hits, whose word has the other properties too."""

import os
from functools import lru_cache

import numba
import numpy as np

NO_HITS = np.empty((0, 9), dtype=np.uint32)


def property_names(db_path):
    """The word properties of a database (its data directory): those the collocation arrays have, the lemma aside"""
    colloc_dir = os.path.join(db_path, "collocations")
    try:
        mtime = os.stat(colloc_dir).st_mtime_ns
    except FileNotFoundError:
        return frozenset()
    return _property_names(colloc_dir, mtime)


@lru_cache(maxsize=32)
def _property_names(colloc_dir, mtime):
    files = os.listdir(colloc_dir)
    names = {
        name[len("attr_") : -len("_ids.npy")]
        for name in files
        if name.startswith("attr_") and name.endswith("_ids.npy")
    }
    return frozenset(names - {"lemma"})


def split_properties(key, names):
    """The word of an index key (with its lemma: prefix, for a lemma) and the (property, value) pairs it ends with,
    read from its end as long as their name is one of names"""
    parts = key.split(":")
    first = 2 if parts[0] == "lemma" and len(parts) > 1 else 1  # the word is never split
    pairs = []
    while len(parts) - 2 >= first and parts[-2] in names:
        pairs.append((parts[-2], parts[-1]))
        del parts[-2:]
    pairs.reverse()
    return ":".join(parts), pairs


def value_ids(db_path, name):
    """The id of each value of a word property in the collocation arrays, by value"""
    path = os.path.join(db_path, "collocations", f"attr_{name}_vocab.npy")
    return _value_ids(path, os.stat(path).st_mtime_ns)


@lru_cache(maxsize=64)
def _value_ids(path, mtime):
    return {str(value): i for i, value in enumerate(np.load(path, allow_pickle=True).tolist())}


def hits_with_properties(db_path, key, stored_hits):
    """The hits of a key with several properties of the word, or None if key has fewer than two. stored_hits(key) is
    what the index has under a key: an (n, 9) uint32 array of hits, or None."""
    if key.count(":") < 4:  # word:name:value:name:value at least
        return None
    word, pairs = split_properties(key, property_names(db_path))
    if len(pairs) < 2:
        return None
    values = {}
    for name, value in pairs:
        if values.setdefault(name, value) != value:  # a word has one value of a property
            return NO_HITS
    keys = []
    for name, value in values.items():
        hits = stored_hits(f"{word}:{name}:{value}")
        if hits is None or len(hits) == 0:
            return NO_HITS
        keys.append((len(hits), name, hits))
    _, kept, hits = min(keys, key=lambda k: k[0])
    if len(keys) == 1:
        return hits
    colloc_dir = os.path.join(db_path, "collocations")
    columns, targets = [], []
    for name, value in values.items():
        if name != kept:
            value_id = value_ids(db_path, name).get(value)
            if value_id is None:
                return NO_HITS
            columns.append(np.load(os.path.join(colloc_dir, f"attr_{name}_ids.npy"), mmap_mode="r"))
            targets.append(value_id)
    keep = _nb_have_values(
        hits,
        np.load(os.path.join(colloc_dir, "sent_keys_native.npy"), mmap_mode="r"),
        np.load(os.path.join(colloc_dir, "sent_offsets.npy"), mmap_mode="r"),
        tuple(columns),
        np.array(targets, dtype=np.uint32),
    )
    return hits[keep]


def property_masks(db_path, properties):
    """The collocation arrays of word properties, and for each which of its values are kept, of (name, values) pairs:
    (columns, allowed) tuples for _nb_word_kept, or (None, None) without any"""
    if not properties:
        return None, None
    colloc_dir = os.path.join(db_path, "collocations")
    columns, allowed = [], []
    for name, values in properties:
        ids = value_ids(db_path, name)
        mask = np.zeros(max(ids.values(), default=-1) + 1, dtype=np.bool_)
        for value in values:
            if value in ids:
                mask[ids[value]] = True
        columns.append(np.load(os.path.join(colloc_dir, f"attr_{name}_ids.npy"), mmap_mode="r"))
        allowed.append(mask)
    return tuple(columns), tuple(allowed)


@numba.njit(cache=True)
def _nb_word_kept(word, columns, allowed):
    """Whether the word at index word of the collocation arrays has a kept value of each property (columns and
    allowed as property_masks gives them): always, without properties (None)"""
    if columns is not None:
        for c in range(len(columns)):
            if not allowed[c][columns[c][word]]:
                return False
    return True


@numba.njit(cache=True)
def _nb_key_order(sent_key, hit):
    """Order of a sentence key (6 uint32) and of the sentence of a hit (its first 6 fields)"""
    for k in range(6):
        if sent_key[k] < hit[k]:
            return -1
        if sent_key[k] > hit[k]:
            return 1
    return 0


@numba.njit(parallel=True, cache=True)
def _nb_have_values(hits, sent_keys, sent_offsets, columns, targets):
    """Which hits are of a word whose value in each column is its target: the word is found in the arrays by the
    sentence of the hit and its position in it (field 7, from 1)"""
    n = hits.shape[0]
    n_sents = sent_keys.shape[0]
    keep = np.zeros(n, dtype=np.bool_)
    for i in numba.prange(n):
        lo, hi = 0, n_sents
        while lo < hi:
            mid = (lo + hi) // 2
            if _nb_key_order(sent_keys[mid], hits[i]) < 0:
                lo = mid + 1
            else:
                hi = mid
        if lo == n_sents or _nb_key_order(sent_keys[lo], hits[i]) != 0:
            continue
        word = np.uint64(sent_offsets[lo]) + np.uint64(hits[i, 7]) - np.uint64(1)
        ok = True
        for c in range(len(columns)):
            if columns[c][word] != targets[c]:
                ok = False
                break
        keep[i] = ok
    return keep
