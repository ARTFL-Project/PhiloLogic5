"""Unit tests for the co-occurrence table written next to the collocation arrays (write_cooccurrence), the word senses'
reading of it (corpus_cooccurrence), and the script that writes it for databases loaded before."""

import itertools
import os

import numpy as np
import pytest

from philologic.loadtime.PostFilters import cooccurrence_paths, write_cooccurrence, write_corpus_counts
from philologic.runtime.reports import word_senses as ws
from philologic.utils import write_cooccurrence as script


def database(root, sentences, n_vocab, lemmas=None, n_lemmas=0):
    """A database's data directory with the collocation arrays and corpus counts of the sentences (lists of word ids),
    and of their lemmas (as many lists of lemma ids), if given."""
    colloc_dir = root / "data" / "collocations"
    colloc_dir.mkdir(parents=True)
    offsets = np.concatenate([[0], np.cumsum([len(s) for s in sentences])]).astype(np.uint64)
    np.save(colloc_dir / "sent_offsets.npy", offsets)
    np.save(colloc_dir / "token_ids.npy", np.array(list(itertools.chain(*sentences)), dtype=np.uint32))
    np.save(colloc_dir / "vocab_offsets.npy", np.arange(n_vocab + 1, dtype=np.uint64))
    if lemmas is not None:
        np.save(colloc_dir / "attr_lemma_ids.npy", np.array(list(itertools.chain(*lemmas)), dtype=np.uint32))
        np.save(colloc_dir / "attr_lemma_vocab_offsets.npy", np.arange(n_lemmas + 1, dtype=np.uint64))
    write_corpus_counts(str(colloc_dir))
    return root / "data", colloc_dir


def brute_force(sentences, n_vocab, skipped=()):
    """The number of sentences (but those skipped) with each pair of ids (the diagonal: with each)."""
    counts = np.zeros((n_vocab, n_vocab), dtype=np.int64)
    for s, sentence in enumerate(sentences):
        if s not in skipped:
            for a in set(sentence):
                for b in set(sentence):
                    counts[a, b] += 1
    return counts


def table(colloc_dir, kind, n_vocab):
    """The table as a square array, each pair once in its row: checks that no pair is in both of its rows."""
    paths = cooccurrence_paths(str(colloc_dir), kind)
    indptr, partners, counts = (np.load(paths[part]) for part in ("indptr", "partners", "counts"))
    assert len(indptr) == n_vocab + 1 and indptr[-1] == len(partners) == len(counts)
    square = np.zeros((n_vocab, n_vocab), dtype=np.int64)
    for a in range(n_vocab):
        row = partners[indptr[a] : indptr[a + 1]]
        assert np.all(np.diff(row.astype(np.int64)) > 0)  # ascending, each once
        for b, c in zip(row, counts[indptr[a] : indptr[a + 1]]):
            assert square[a, b] == 0 and square[b, a] == 0 and b != a
            square[a, b] = square[b, a] = c
    return square


def random_corpus(rng, n_vocab, n_sentences):
    p = rng.dirichlet(np.full(n_vocab, 0.3))
    return [rng.choice(n_vocab, size=int(rng.integers(0, 15)), p=p).tolist() for _ in range(n_sentences)]


@pytest.mark.unit
class TestTable:
    def test_the_table_counts_every_pair_once_a_sentence(self, tmp_path):
        rng = np.random.default_rng(0)
        for trial in range(12):
            n_vocab = int(rng.integers(1, 40))
            sentences = random_corpus(rng, n_vocab, int(rng.integers(1, 80)))
            _, colloc_dir = database(tmp_path / str(trial), sentences, n_vocab)
            assert write_cooccurrence(str(colloc_dir), workers=int(rng.integers(1, 5))) == [
                f"cooccurrence_word_{part}.npy" for part in ("indptr", "partners", "counts")
            ]
            expected = brute_force(sentences, n_vocab)
            np.fill_diagonal(expected, 0)
            assert np.array_equal(table(colloc_dir, "word", n_vocab), expected)
            assert not [name for name in os.listdir(colloc_dir) if name.endswith(".tmp")]

    def test_a_pair_is_in_the_row_of_the_rarer_word(self, tmp_path):
        # 0 is in 3 sentences, 1 in 2, 2 in 1; 3 and 4 in as many (1): the pair is in the row of the later
        _, colloc_dir = database(tmp_path, [[0, 1, 2], [0, 1], [0, 3, 4]], 5)
        write_cooccurrence(str(colloc_dir))
        paths = cooccurrence_paths(str(colloc_dir), "word")
        indptr, partners = np.load(paths["indptr"]), np.load(paths["partners"])
        rows = [partners[indptr[a] : indptr[a + 1]].tolist() for a in range(5)]
        assert rows == [[], [0], [0, 1], [0], [0, 3]]

    def test_the_table_is_the_same_in_any_number_of_threads(self, tmp_path):
        rng = np.random.default_rng(1)
        sentences = random_corpus(rng, 60, 400)
        written = []
        for workers in (1, 3, 8):
            _, colloc_dir = database(tmp_path / str(workers), sentences, 60)
            write_cooccurrence(str(colloc_dir), workers)
            written.append([open(p, "rb").read() for p in cooccurrence_paths(str(colloc_dir), "word").values()])
        assert written[0] == written[1] == written[2]

    def test_words_and_lemmas_each_have_their_table(self, tmp_path):
        words = [[0, 1, 2], [3, 1], [4, 0]]
        lemmas = [[0, 1, 1], [0, 1], [2, 0]]
        _, colloc_dir = database(tmp_path, words, 5, lemmas, 3)
        assert len(write_cooccurrence(str(colloc_dir))) == 6
        expected = brute_force(lemmas, 3)
        np.fill_diagonal(expected, 0)
        assert np.array_equal(table(colloc_dir, "lemma", 3), expected)


@pytest.mark.unit
class TestRuntime:
    def test_the_corpus_outside_the_query_is_the_table_less_its_sentences(self, tmp_path):
        rng = np.random.default_rng(2)
        for trial in range(10):
            n_vocab = int(rng.integers(2, 50))
            sentences = random_corpus(rng, n_vocab, int(rng.integers(1, 120)))
            data, colloc_dir = database(tmp_path / str(trial), sentences, n_vocab)
            write_cooccurrence(str(colloc_dir), workers=2)
            words = rng.permutation(n_vocab)[: int(rng.integers(1, n_vocab + 1))]
            skipped = np.sort(rng.permutation(len(sentences))[: int(rng.integers(0, len(sentences) + 1))])
            cooc = ws.corpus_cooccurrence(str(data), False, words, skipped)
            expected = brute_force(sentences, n_vocab, set(skipped.tolist()))[np.ix_(words, words)]
            assert cooc.dtype == np.int32 and np.array_equal(cooc, expected)

    def test_a_database_without_it_names_the_script_that_writes_it(self, tmp_path):
        data, _ = database(tmp_path, [[0, 1], [1, 2]], 3)
        with pytest.raises(FileNotFoundError, match="philologic.utils.write_cooccurrence"):
            ws.corpus_cooccurrence(str(data), False, [0, 1], [])

    def test_a_table_partly_written_is_not_read(self, tmp_path):
        data, colloc_dir = database(tmp_path, [[0, 1], [1, 2]], 3)
        write_cooccurrence(str(colloc_dir))
        np.save(cooccurrence_paths(str(colloc_dir), "word")["counts"], np.zeros(1, dtype=np.uint32))
        with pytest.raises(ValueError, match="write_cooccurrence --force"):
            ws.corpus_cooccurrence(str(data), False, [0, 1], [])


@pytest.mark.unit
class TestScript:
    def test_writes_the_table_once_unless_forced(self, tmp_path, capsys):
        _, colloc_dir = database(tmp_path / "db", [[0, 1, 2], [1, 2]], 3)
        script.main([str(tmp_path / "db")])
        assert "cooccurrence_word_partners.npy" in capsys.readouterr().out
        assert script.has_table(str(colloc_dir))
        script.main([str(tmp_path / "db")])
        assert "already has its co-occurrence tables" in capsys.readouterr().out
        script.main([str(tmp_path / "db")], force=True)
        assert "cooccurrence_word_partners.npy" in capsys.readouterr().out

    def test_a_table_of_another_vocabulary_is_written_again(self, tmp_path):
        _, colloc_dir = database(tmp_path / "db", [[0, 1, 2], [1, 2]], 3)
        write_cooccurrence(str(colloc_dir))
        np.save(cooccurrence_paths(str(colloc_dir), "word")["indptr"], np.zeros(3, dtype=np.int64))  # since reloaded
        assert not script.has_table(str(colloc_dir))
        script.main([str(tmp_path / "db")])
        assert script.has_table(str(colloc_dir))

    def test_the_corpus_counts_are_written_first_if_missing(self, tmp_path, capsys):
        _, colloc_dir = database(tmp_path / "db", [[0, 1], [1]], 2)
        for name in ("sentence_df_word.npy", "token_counts_word.npy"):
            os.unlink(colloc_dir / name)
        script.main([str(tmp_path / "db")])
        assert "written first" in capsys.readouterr().out
        assert script.has_table(str(colloc_dir))

    def test_a_directory_without_collocation_arrays_is_named_and_skipped(self, tmp_path, capsys):
        (tmp_path / "nodb" / "data").mkdir(parents=True)
        script.main([str(tmp_path / "nodb")])
        assert "no collocation arrays" in capsys.readouterr().err
