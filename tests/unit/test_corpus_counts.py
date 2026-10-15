"""Unit tests for the corpus counts written next to the collocation arrays (write_corpus_counts), their readers, and
the script that writes them for databases loaded before."""

import os

import numpy as np
import pytest

from philologic.loadtime.PostFilters import write_corpus_counts
from philologic.runtime.reports.collocation import _load_sentence_df, _load_token_counts
from philologic.utils import write_corpus_counts as script

# Three sentences: words 0 1 1 2 | 1 3 | 0, of a vocabulary of 5 (word 4 never occurs); their lemmas 0 0 1 1 | 0 2 | 0
WORDS = np.array([0, 1, 1, 2, 1, 3, 0], dtype=np.uint32)
LEMMAS = np.array([0, 0, 1, 1, 0, 2, 0], dtype=np.uint32)
SENT_OFFSETS = np.array([0, 4, 6, 7], dtype=np.uint64)


def database(root, lemmas=True):
    """A database's data directory with the collocation arrays the counts are counted from."""
    colloc_dir = root / "data" / "collocations"
    colloc_dir.mkdir(parents=True)
    np.save(colloc_dir / "token_ids.npy", WORDS)
    np.save(colloc_dir / "vocab_offsets.npy", np.arange(6, dtype=np.uint64))  # 5 words
    np.save(colloc_dir / "sent_offsets.npy", SENT_OFFSETS)
    if lemmas:
        np.save(colloc_dir / "attr_lemma_ids.npy", LEMMAS)
        np.save(colloc_dir / "attr_lemma_vocab_offsets.npy", np.arange(4, dtype=np.uint64))  # 3 lemmas
    return root / "data", colloc_dir


class TestWriteCorpusCounts:
    def test_counts_the_sentences_and_occurrences_of_words_and_lemmas(self, tmp_path):
        _, colloc_dir = database(tmp_path)
        written = write_corpus_counts(str(colloc_dir))
        assert sorted(written) == [
            "sentence_df_lemma.npy",
            "sentence_df_word.npy",
            "token_counts_lemma.npy",
            "token_counts_word.npy",
        ]
        assert np.load(colloc_dir / "sentence_df_word.npy").tolist() == [2, 2, 1, 1, 0]
        assert np.load(colloc_dir / "token_counts_word.npy").tolist() == [2, 3, 1, 1, 0]
        assert np.load(colloc_dir / "sentence_df_lemma.npy").tolist() == [3, 1, 1]
        assert np.load(colloc_dir / "token_counts_lemma.npy").tolist() == [4, 2, 1]
        assert not [name for name in os.listdir(colloc_dir) if name.endswith(".tmp")]

    def test_a_database_without_lemmas_gets_the_counts_of_its_words(self, tmp_path):
        _, colloc_dir = database(tmp_path, lemmas=False)
        assert sorted(write_corpus_counts(str(colloc_dir))) == ["sentence_df_word.npy", "token_counts_word.npy"]


class TestReaders:
    def test_the_reports_read_the_written_counts(self, tmp_path):
        data, colloc_dir = database(tmp_path)
        write_corpus_counts(str(colloc_dir))
        assert _load_token_counts(str(data), False).tolist() == [2, 3, 1, 1, 0]
        assert _load_token_counts(str(data), True).tolist() == [4, 2, 1]
        df, n_sentences = _load_sentence_df(str(data), True)
        assert df.tolist() == [3, 1, 1] and n_sentences == 3

    def test_a_reloaded_database_s_counts_are_read(self, tmp_path):
        """Nothing read for a database is kept: a reload at the same path, with another vocabulary, is read anew."""
        data, colloc_dir = database(tmp_path)
        write_corpus_counts(str(colloc_dir))
        assert _load_token_counts(str(data), False).tolist() == [2, 3, 1, 1, 0]
        assert _load_sentence_df(str(data), False)[0].tolist() == [2, 2, 1, 1, 0]
        reloaded = tmp_path / "reloaded"
        _, reloaded_colloc_dir = database(reloaded, lemmas=False)
        np.save(reloaded_colloc_dir / "token_ids.npy", np.array([0, 0, 1], dtype=np.uint32))
        np.save(reloaded_colloc_dir / "vocab_offsets.npy", np.arange(3, dtype=np.uint64))  # 2 words
        np.save(reloaded_colloc_dir / "sent_offsets.npy", np.array([0, 3], dtype=np.uint64))
        write_corpus_counts(str(reloaded_colloc_dir))
        os.rename(data, tmp_path / "old_data")
        os.rename(reloaded / "data", data)
        assert _load_token_counts(str(data), False).tolist() == [2, 1]
        df, n_sentences = _load_sentence_df(str(data), False)
        assert df.tolist() == [1, 1] and n_sentences == 1

    def test_a_database_without_them_names_the_script_that_writes_them(self, tmp_path):
        data, _ = database(tmp_path)
        with pytest.raises(FileNotFoundError, match="philologic.utils.write_corpus_counts"):
            _load_token_counts(str(data), False)
        with pytest.raises(FileNotFoundError, match="philologic.utils.write_corpus_counts"):
            _load_sentence_df(str(data), False)


class TestScript:
    def test_writes_the_counts_once_unless_forced(self, tmp_path, capsys):
        _, colloc_dir = database(tmp_path / "db")
        script.main([str(tmp_path / "db")])
        assert "token_counts_word.npy" in capsys.readouterr().out
        assert script.has_counts(str(colloc_dir))
        script.main([str(tmp_path / "db")])
        assert "already has its corpus counts" in capsys.readouterr().out
        script.main([str(tmp_path / "db")], force=True)
        assert "token_counts_word.npy" in capsys.readouterr().out

    def test_counts_of_another_vocabulary_are_written_again(self, tmp_path):
        _, colloc_dir = database(tmp_path / "db")
        write_corpus_counts(str(colloc_dir))
        np.save(colloc_dir / "sentence_df_lemma.npy", np.zeros(2, dtype=np.int64))  # a vocabulary since reloaded
        assert not script.has_counts(str(colloc_dir))
        script.main([str(tmp_path / "db")])
        assert np.load(colloc_dir / "sentence_df_lemma.npy").tolist() == [3, 1, 1]

    def test_a_directory_without_collocation_arrays_is_named_and_skipped(self, tmp_path, capsys):
        (tmp_path / "nodb" / "data").mkdir(parents=True)
        script.main([str(tmp_path / "nodb")])
        assert "no collocation arrays" in capsys.readouterr().err
