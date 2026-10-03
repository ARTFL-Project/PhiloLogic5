"""Unit tests for the order of metadata values in sorts (HitList.sort_key, sort_ranks), which the KWIC sorted by
metadata shares with the concordance."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime.HitList import sort_ranks

TITLES = ["Zoloé", "Élégies", "émile", "Emile", "Adolphe", "", None]


def in_rank_order(ranks):
    return sorted(ranks, key=lambda value: (ranks[value], value))


@pytest.mark.unit
class TestSortRanks:
    def test_regardless_of_accents(self):
        """With ascii_conversion, É sorts as E: "Élégies" was ranked after "Zoloé" in KWIC sorts."""
        ranks = sort_ranks(TITLES, ascii_conversion=True)
        assert in_rank_order(ranks) == ["Adolphe", "Élégies", "Emile", "émile", "Zoloé"]
        assert ranks["Emile"] == ranks["émile"]  # sort alike, so tied

    def test_accents_kept(self):
        ranks = sort_ranks(TITLES, ascii_conversion=False)
        assert in_rank_order(ranks) == ["Adolphe", "Emile", "Zoloé", "Élégies", "émile"]

    def test_missing_values_unranked(self):
        """Hits without a value get the KWIC's last rank."""
        assert set(sort_ranks(TITLES, ascii_conversion=True)) == {"Zoloé", "Élégies", "émile", "Emile", "Adolphe"}

    def test_numbers_in_numeric_order(self):
        assert in_rank_order(sort_ranks([1850, 900, 1700], ascii_conversion=True)) == [900, 1700, 1850]
