"""Unit tests for the comparison of two subcorpora's collocates: the words over-represented in the primary one (top)
and in the one compared with (bottom)."""

import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "www"))

from scripts.comparative_collocations import get_relative_proportions  # noqa: E402


def test_over_and_under_represented():
    top, bottom = get_relative_proportions(Counter(roi=20, peuple=1), Counter(roi=1, peuple=20), whole_corpus=False)
    assert [word for word, _ in top] == ["roi"]
    assert [word for word, _ in bottom] == ["peuple"]


def test_words_as_frequent_in_both_are_in_neither():
    """Their score is 0: two subcorpora with the same collocates have none over- or under-represented."""
    collocates = Counter(roi=20, peuple=1, loi=5)
    assert get_relative_proportions(collocates, Counter(collocates), whole_corpus=False) == ([], [])


def test_whole_corpus_compared_with_its_empty_rest():
    """The primary subcorpus is the whole corpus: the rest of the corpus, compared with, has no collocates."""
    collocates = Counter(roi=20, peuple=1, loi=5)
    assert get_relative_proportions(collocates, Counter(collocates), whole_corpus=True) == ([], [])
