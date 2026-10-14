"""Integration tests for the collocate filters: lemmas or word forms, common words, word properties (any of the
values of a property, all the properties), and searches for words with several properties, which the links of
collocates filtered by several properties are."""

import os
import re
import sys
import urllib.parse
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime import WebConfig, WSGIHandler  # noqa: E402
from philologic.runtime.exceptions import BadRequest  # noqa: E402
from philologic.runtime.reports.collocation import (  # noqa: E402
    CollocateFilters,
    collocate_filters,
    collocate_keys,
    collocation_results,
    safe_pickle_load,
)

pytestmark = pytest.mark.integration


def request_for(db, **params):
    root = os.path.dirname(os.path.normpath(db.path))
    config = WebConfig(root)
    return WSGIHandler({"QUERY_STRING": urllib.parse.urlencode(params), "PHILOLOGIC_DBPATH": root}, config), config


def report(db, **params):
    request, config = request_for(db, **params)
    return collocation_results(request, config)


def all_collocates(result):
    return safe_pickle_load(result["file_path"])


def total_hits(db, q, method, method_arg="0"):
    hits = db.query(q, method, method_arg)
    hits.finish()
    return len(hits)


class Arrays:
    """The collocation arrays of a database, read as they are: the reference the counts are checked against"""

    def __init__(self, db):
        col = os.path.join(db.path, "collocations")
        self.sent_keys = np.load(os.path.join(col, "sent_keys_s24.npy"), mmap_mode="r")
        self.sent_offsets = np.load(os.path.join(col, "sent_offsets.npy"), mmap_mode="r")
        self.ids = {
            False: np.load(os.path.join(col, "token_ids.npy"), mmap_mode="r"),
            True: np.load(os.path.join(col, "attr_lemma_ids.npy"), mmap_mode="r"),
        }
        self.strings = {lemmas: self._strings(col, "attr_lemma_" if lemmas else "") for lemmas in (False, True)}
        self.col = col

    @staticmethod
    def _strings(col, prefix):
        offsets = np.load(os.path.join(col, f"{prefix}vocab_offsets.npy"))
        with open(os.path.join(col, f"{prefix}vocab_strings.bin"), "rb") as f:
            blob = f.read()
        return [blob[offsets[i] : offsets[i + 1]].decode("utf-8") for i in range(len(offsets) - 1)]

    def values(self, name):
        ids = np.load(os.path.join(self.col, f"attr_{name}_ids.npy"), mmap_mode="r")
        vocab = [str(v) for v in np.load(os.path.join(self.col, f"attr_{name}_vocab.npy"), allow_pickle=True)]
        return ids, vocab

    def sentence(self, row):
        key = np.ascontiguousarray(row[:6]).astype(">u4").view("S24")[0]
        si = int(np.searchsorted(self.sent_keys, key))
        assert self.sent_keys[si] == key
        return int(self.sent_offsets[si]), int(self.sent_offsets[si + 1])

    def word(self, index, lemmas):
        return self.strings[lemmas][int(self.ids[lemmas][index])]


def expected_counts(db, arrays, result, filters, q, distance=None):
    """The collocates of q counted word by word from the arrays: in the hit's sentence (within distance words of the
    hit's words), but the hit's first word, the words of the report's filter list (as word forms or lemmas, whichever
    they are), and the words without a kept value of each property"""
    hits = db.query(q, "single_term", "0", raw_results=True)
    hits.finish()
    rows = hits.read_array()
    left_out = set(result["filter_list"])
    properties = [(arrays.values(name), set(values)) for name, values in filters.properties]
    counts = Counter()
    for row in rows:
        start, end = arrays.sentence(row)
        q_pos = int(row[7])
        for index in range(start, end):
            pos = index - start + 1
            if pos == q_pos or (distance is not None and abs(pos - q_pos) > distance):
                continue
            if arrays.word(index, False) in left_out or arrays.word(index, True) in left_out:
                continue
            if all(vocab[int(ids[index])] in kept for (ids, vocab), kept in properties):
                counts[arrays.word(index, filters.count_lemmas)] += 1
    return counts


@pytest.fixture(scope="module")
def arrays(eltec_db):
    return Arrays(eltec_db)


class TestCounts:
    """Each filter, and combinations, against counts made word by word from the arrays"""

    @pytest.mark.parametrize(
        "params",
        [
            {"q": "lemma:light", "colloc_lemmas": "no", "colloc_filter_choice": "nofilter"},
            {"q": "light", "colloc_lemmas": "yes", "colloc_filter_choice": "frequency"},
            {"q": "light", "colloc_lemmas": "no", "colloc_filter_choice": "frequency", "colloc_attr_pos": "NOUN | ADJ"},
            {
                "q": "lemma:light",
                "colloc_lemmas": "yes",
                "colloc_filter_choice": "nofilter",
                "colloc_attr_pos": "VERB",
                "colloc_attr_join": "left",
            },
            {
                "q": "lemma:light",
                "colloc_lemmas": "yes",
                "colloc_filter_choice": "frequency",
                "colloc_attr_pos": "ADJ",
                "method_arg": "3",
            },
        ],
    )
    def test_counts(self, eltec_db, arrays, params):
        request, config = request_for(eltec_db, **params)
        filters = collocate_filters(request, eltec_db)
        result = collocation_results(request, config)
        distance = int(params["method_arg"]) if params.get("method_arg") else None
        expected = expected_counts(eltec_db, arrays, result, filters, params["q"], distance)
        assert expected  # something is counted
        assert all_collocates(result) == expected

    def test_lemma_counted_as_forms(self, eltec_db):
        """A lemma's collocates counted as word forms: its own forms are left out (see test_counts)"""
        result = report(eltec_db, q="lemma:light", colloc_lemmas="no", colloc_filter_choice="nofilter")
        assert not any(c.startswith("lemma:") for c in all_collocates(result))
        assert "lemma:light" in result["filter_list"]

    def test_property_query_word_not_counted(self, eltec_db):
        """The word of a query with a property (light:pos:NOUN) is not its own collocate either"""
        result = report(eltec_db, q="light:pos:NOUN", colloc_lemmas="no", colloc_filter_choice="nofilter")
        assert "light" not in all_collocates(result)

    def test_old_attribute_links(self, eltec_db):
        """A link made before filters could be combined: one value of one property, no other filter"""
        old = report(eltec_db, q="light", colloc_filter_choice="attribute", q_attribute="pos", q_attribute_value="NOUN")
        new = report(eltec_db, q="light", colloc_filter_choice="nofilter", colloc_attr_pos="NOUN")
        assert all_collocates(old) == all_collocates(new)

    def test_old_lemma_links(self, eltec_db):
        """Without colloc_lemmas, a lemma query has its collocates counted as lemmas, as before"""
        old = report(eltec_db, q="lemma:light", colloc_filter_choice="nofilter")
        new = report(eltec_db, q="lemma:light", colloc_filter_choice="nofilter", colloc_lemmas="yes")
        assert all_collocates(old) == all_collocates(new)
        assert all(c.startswith("lemma:") for c in all_collocates(old))

    def test_common_words_combined(self, eltec_db):
        """Both lists of common words can be left out together"""
        both = report(eltec_db, q="light", colloc_filter_choice="frequency | stopwords")
        frequent = report(eltec_db, q="light", colloc_filter_choice="frequency")
        assert set(frequent["filter_list"]) <= set(both["filter_list"])


class TestLinks:
    """The hits of a collocate's link are those it was counted in"""

    @pytest.mark.parametrize(
        "params",
        [
            {"q": "light", "colloc_lemmas": "no", "colloc_attr_pos": "NOUN | ADJ"},
            {"q": "lemma:light", "colloc_lemmas": "yes", "colloc_attr_pos": "NOUN | VERB"},
            {
                "q": "lemma:light",
                "colloc_lemmas": "no",
                "colloc_filter_choice": "nofilter",
                "colloc_attr_pos": "VERB",
                "colloc_attr_join": "left",
            },
        ],
    )
    def test_links(self, eltec_db, params):
        request, config = request_for(eltec_db, **params)
        filters = collocate_filters(request, eltec_db)
        result = collocation_results(request, config)
        assert result["collocates"]
        for collocate, count in result["collocates"][:5]:
            link = "|".join(collocate_keys(collocate, filters))
            assert count == total_hits(eltec_db, f"{params['q']} {link}", "sentence_unordered"), collocate


class TestRequests:
    def test_keys(self):
        filters = CollocateFilters(True, (), (("join", ("left",)), ("pos", ("AUX", "VERB"))))
        assert collocate_keys("lemma:be", filters) == ["lemma:be:join:left:pos:AUX", "lemma:be:join:left:pos:VERB"]
        assert collocate_keys("be", CollocateFilters(False)) == ["be"]

    @pytest.mark.parametrize(
        "params, message",
        [
            ({"colloc_attr_nope": "X"}, "no word property"),
            ({"colloc_lemmas": "maybe"}, "counted as lemmas"),
            ({"colloc_filter_choice": "rare"}, "No such collocate filter"),
            ({"colloc_filter_choice": "attribute"}, "needs the attribute"),
        ],
    )
    def test_bad_requests(self, eltec_db, params, message):
        with pytest.raises(BadRequest, match=message):
            report(eltec_db, q="light", **params)

    def test_no_lemmas(self, shakespeare_db):
        with pytest.raises(BadRequest, match="no lemmas"):
            report(shakespeare_db, q="lord", colloc_lemmas="yes")


class TestSeveralProperties:
    """Searches for a word with several properties, against the words of the arrays that have them all"""

    def expected(self, arrays, word, lemmas, conditions):
        ids = arrays.ids[lemmas]
        strings = arrays.strings[lemmas]
        wanted = [i for i, s in enumerate(strings) if re.fullmatch(word, s)]
        keep = np.isin(ids, wanted)
        for name, value in conditions:
            values, vocab = arrays.values(name)
            keep &= values == (vocab.index(value) if value in vocab else -1)
        return int(keep.sum())

    @pytest.mark.parametrize(
        "q, word, lemmas, conditions",
        [
            ("s:pos:VERB:join:left", "s", False, [("pos", "VERB"), ("join", "left")]),
            ("s:join:left:pos:PART", "s", False, [("pos", "PART"), ("join", "left")]),
            ("lemma:be:pos:AUX:join:left", "lemma:be", True, [("pos", "AUX"), ("join", "left")]),
            ("s:pos:VERB:pos:VERB", "s", False, [("pos", "VERB")]),
            ("s:pos:VERB:pos:PART", "s", False, [("pos", "VERB"), ("pos", "PART")]),
            ("s:pos:VERB:join:nope", "s", False, [("pos", "VERB"), ("join", "nope")]),
            ("ll:pos:VERB:join:left | ve:pos:VERB:join:left", "ll|ve", False, [("pos", "VERB"), ("join", "left")]),
        ],
    )
    def test_single_term(self, eltec_db, arrays, q, word, lemmas, conditions):
        assert total_hits(eltec_db, q, "single_term") == self.expected(arrays, word, lemmas, conditions)

    def test_pattern(self, eltec_db, arrays):
        """A pattern is matched with the word and its first property"""
        assert total_hits(eltec_db, "l.*:pos:VERB:join:left", "single_term") == self.expected(
            arrays, "l.*", False, [("pos", "VERB"), ("join", "left")]
        )

    def test_with_other_words(self, eltec_db):
        """In a co-occurrence, the hits are those of the word with one of the properties whose word has the other"""
        with_both = total_hits(eltec_db, "he s:pos:VERB:join:left", "sentence_unordered")
        with_one = total_hits(eltec_db, "he s:pos:VERB", "sentence_unordered")
        assert 0 < with_both <= with_one
