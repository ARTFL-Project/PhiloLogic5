"""Unit tests for the search collocates are counted around."""

import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime.reports.collocation import collocation_search_method


@pytest.mark.unit
@pytest.mark.parametrize(
    "q, method",
    [
        ("liberté", "single_term"),
        ('"liberté"', "single_term"),
        ("liberté | égalité", "single_term"),
        ("amour NOT amours", "single_term"),
        ("lemma:aimer", "single_term"),
        ('"assemblée nationale"', "phrase_ordered"),
        ("peuple souverain", "sentence_unordered"),
        ('"liberté" "amis"', "sentence_unordered"),
        ('"assemblée nationale" peuple', "sentence_unordered"),
    ],
)
def test_collocation_search_method(q, method):
    """One term: its occurrences; a quoted phrase: its occurrences; several terms: their co-occurrences in a sentence."""
    assert collocation_search_method(q) == (method, "0")


@pytest.mark.unit
@pytest.mark.parametrize(
    "q, method",
    [
        ("liberté", ("single_term", "0")),
        ('"assemblée nationale"', ("phrase_ordered", "0")),
        ("peuple souverain", ("proxy_unordered", "5")),
    ],
)
def test_collocation_search_method_within_n_words(q, method):
    """Within n words: several terms co-occur within n words, unordered."""
    assert collocation_search_method(q, distance=5) == method
