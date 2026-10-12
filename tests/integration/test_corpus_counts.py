"""Integration test: the loader writes the corpus counts next to the collocation arrays."""

import os

import numpy as np
import pytest

from philologic.loadtime.PostFilters import _COUNTED


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
