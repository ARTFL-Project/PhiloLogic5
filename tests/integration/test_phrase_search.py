"""Integration tests for quoted phrases in searches of terms in a sentence or within n words: the words of a phrase are
next to each other, in order, and the phrase is one term, so within n words of it is within n words of its first or
last word. Checked against every combination of the hits of each word that satisfies that."""

import itertools
import sys
from collections import defaultdict
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime.Query import group_terms, parse_query, phrase_lengths, split_terms

TITLE = "Henry|Richard|Hamlet"


def found(db, q, method, method_arg="0", **metadata):
    """The hits of the search, each as its sentence and the (position, byte offset) of each of its words."""
    hits = db.query(q, method, method_arg, raw_results=True, **metadata)
    hits.finish()
    return [
        (tuple(int(x) for x in row[:6]), tuple(sorted(zip(map(int, row[7::2]), map(int, row[8::2])))))
        for row in hits.read_array()
    ]


def expected(db, q, ordered, distance=None, exact=False, **metadata):
    """The hits of q, from those of each of its words: in the same sentence, with the words of each phrase next to each
    other, in order, and the terms within distance words (or exactly distance words) of each other."""
    grouped = group_terms(parse_query(q, query_patterns=db.locals.query_patterns))
    groups = split_terms(grouped)
    phrase_next = []
    for length in phrase_lengths(grouped):
        phrase_next += [True] * (length - 1) + [False]
    phrase_words = sum(phrase_next)
    words_by_sentence = []
    for group in groups:
        by_sentence = defaultdict(list)
        for sentence, words in found(db, " | ".join(token for _, token in group), "single_term", **metadata):
            by_sentence[sentence].extend(words)
        words_by_sentence.append(by_sentence)
    hits = set()
    for sentence in set.intersection(*(set(by_sentence) for by_sentence in words_by_sentence)):
        for words in itertools.product(*(by_sentence[sentence] for by_sentence in words_by_sentence)):
            positions = [position for position, _ in words]
            if len(set(positions)) < len(positions):
                continue
            if ordered and positions != sorted(positions):
                continue
            if any(phrase_next[i] and positions[i + 1] != positions[i] + 1 for i in range(len(positions) - 1)):
                continue
            if distance is not None:
                span = max(positions) - min(positions) - phrase_words
                if span != distance if exact else span > distance:
                    continue
            hits.add((sentence, tuple(sorted(words))))
    return hits


CASES = [
    # q, method, method_arg, ordered, distance, exact
    ('"my lord" king', "sentence_unordered", "0", False, None, False),
    ('"my lord" king', "sentence_ordered", "0", True, None, False),
    ('"my lord" king', "proxy_unordered", "3", False, 3, False),
    ('"my lord" king', "proxy_ordered", "3", True, 3, False),
    ('"my lord" king', "exact_cooc_unordered", "2", False, 2, True),
    ('king "my lord"', "proxy_unordered", "1", False, 1, False),
    ('"my good lord" sir', "sentence_unordered", "0", False, None, False),
    ('"my lord" "my lady"', "sentence_unordered", "0", False, None, False),
    ('"my lord" good | noble', "proxy_unordered", "5", False, 5, False),
    ("my lord king", "sentence_unordered", "0", False, None, False),
    ("my lord king", "proxy_unordered", "3", False, 3, False),
]


@pytest.mark.integration
class TestPhraseInCooccurrence:
    @pytest.mark.parametrize("q, method, method_arg, ordered, distance, exact", CASES)
    def test_hits(self, shakespeare_db, q, method, method_arg, ordered, distance, exact):
        hits = found(shakespeare_db, q, method, method_arg)
        assert len(hits) == len(set(hits))
        assert set(hits) == expected(shakespeare_db, q, ordered, distance, exact)

    @pytest.mark.parametrize("q, method, method_arg, ordered, distance, exact", CASES[:3])
    def test_hits_in_metadata(self, shakespeare_db, q, method, method_arg, ordered, distance, exact):
        hits = found(shakespeare_db, q, method, method_arg, title=TITLE)
        assert set(hits) == expected(shakespeare_db, q, ordered, distance, exact, title=TITLE)
        assert 0 < len(hits) < len(found(shakespeare_db, q, method, method_arg))

    def test_phrase_is_not_the_words(self, shakespeare_db):
        """Quotes make a phrase: '"my lord" king' finds fewer hits than 'my lord king'."""
        assert 0 < len(found(shakespeare_db, '"my lord" king', "sentence_unordered")) < len(
            found(shakespeare_db, "my lord king", "sentence_unordered")
        )


@pytest.mark.integration
class TestQueryOfOnePhrase:
    """A query that is one phrase finds the phrase, whatever the search."""

    @pytest.mark.parametrize("method, method_arg", [
        ("sentence_unordered", "0"),
        ("sentence_ordered", "0"),
        ("proxy_unordered", "5"),
        ("exact_cooc_unordered", "3"),
        ("phrase_unordered", "0"),
    ])
    def test_phrase(self, shakespeare_db, method, method_arg):
        phrase = found(shakespeare_db, '"my lord"', "phrase_ordered")
        assert phrase
        assert found(shakespeare_db, '"my lord"', method, method_arg) == phrase
