"""Integration tests: the loader writes the corpus counts and the co-occurrence tables next to the collocation arrays."""

import os

import numpy as np
import pytest

from philologic.loadtime.PostFilters import _COUNTED, cooccurrence_paths


@pytest.mark.integration
@pytest.mark.parametrize("fixture, kinds", [("eltec_db", {"word", "lemma"}), ("shakespeare_db", {"word"})])
def test_the_loader_writes_the_corpus_counts(request, fixture, kinds):
    db = request.getfixturevalue(fixture)
    colloc_dir = os.path.join(db.path, "collocations")
    sent_offsets = np.load(os.path.join(colloc_dir, "sent_offsets.npy"), mmap_mode="r")
    found = set()
    for ids_file, voff_file, kind in _COUNTED.values():
        if not os.path.exists(os.path.join(colloc_dir, ids_file)):
            continue
        found.add(kind)
        count_ids = np.load(os.path.join(colloc_dir, ids_file), mmap_mode="r")
        n_vocab = len(np.load(os.path.join(colloc_dir, voff_file), mmap_mode="r")) - 1
        df = np.load(os.path.join(colloc_dir, f"sentence_df_{kind}.npy"))
        counts = np.load(os.path.join(colloc_dir, f"token_counts_{kind}.npy"))
        # Counted again another way: each (sentence, id) pair once
        sentence_of = np.repeat(np.arange(len(sent_offsets) - 1), np.diff(sent_offsets).astype(np.int64))
        pairs = np.unique(sentence_of * n_vocab + count_ids.astype(np.int64))
        assert np.array_equal(df, np.bincount(pairs % n_vocab, minlength=n_vocab))
        assert np.array_equal(counts, np.bincount(count_ids, minlength=n_vocab))
        assert counts.sum() == len(count_ids) and (df <= counts).all()
    assert found == kinds


@pytest.mark.integration
@pytest.mark.parametrize("fixture, kinds", [("eltec_db", {"word", "lemma"}), ("shakespeare_db", {"word"})])
def test_the_loader_writes_the_cooccurrence_table(request, fixture, kinds):
    db = request.getfixturevalue(fixture)
    colloc_dir = os.path.join(db.path, "collocations")
    sent_offsets = np.load(os.path.join(colloc_dir, "sent_offsets.npy"), mmap_mode="r")
    sentence_of = np.repeat(np.arange(len(sent_offsets) - 1), np.diff(sent_offsets).astype(np.int64))
    rng = np.random.default_rng(0)
    found = set()
    for ids_file, voff_file, kind in _COUNTED.values():
        if not os.path.exists(os.path.join(colloc_dir, ids_file)):
            continue
        found.add(kind)
        count_ids = np.load(os.path.join(colloc_dir, ids_file)).astype(np.int64)
        n_vocab = len(np.load(os.path.join(colloc_dir, voff_file), mmap_mode="r")) - 1
        paths = cooccurrence_paths(colloc_dir, kind)
        indptr, partners, counts = (np.load(paths[part]) for part in ("indptr", "partners", "counts"))
        assert len(indptr) == n_vocab + 1 and indptr[-1] == len(partners) == len(counts)
        row_of = np.repeat(np.arange(n_vocab), np.diff(indptr))
        df = np.bincount(np.unique(sentence_of * n_vocab + count_ids) % n_vocab, minlength=n_vocab)
        # Some words, from the commonest to the rarest: their partners in both directions, counted again from the text
        by_df = np.argsort(-df, kind="stable")
        for word in np.unique(np.concatenate([by_df[:5], rng.choice(by_df[df[by_df] > 0], 20)])):
            in_row = slice(indptr[word], indptr[word + 1])
            mirrored = partners == word
            got = np.zeros(n_vocab, dtype=np.int64)
            got[partners[in_row]] = counts[in_row]
            assert not got[row_of[mirrored]].any()  # each pair in one row only
            got[row_of[mirrored]] = counts[mirrored]
            with_word = np.zeros(len(sent_offsets) - 1, dtype=np.bool_)
            with_word[sentence_of[count_ids == word]] = True
            pairs = np.unique(sentence_of[with_word[sentence_of]] * n_vocab + count_ids[with_word[sentence_of]])
            expected = np.bincount(pairs % n_vocab, minlength=n_vocab)
            expected[word] = 0
            assert np.array_equal(got, expected)
    assert found == kinds
