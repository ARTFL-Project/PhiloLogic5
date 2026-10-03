"""Unit tests for the aggregation CSV export: each breakdown row has its own object's metadata."""

import csv
import io
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime.reports.aggregation import aggregation_to_csv


def doc(doc_id, title, year):
    return {"philo_id": f"{doc_id} 0 0 0 0 0 0", "author": "Hugo, Victor", "title": title, "year": year,
            "philo_doc_id": str(doc_id), "field_name": "Hugo, Victor"}


@pytest.mark.unit
def test_breakdown_rows():
    """The group's metadata are those of its first object: only its own field is the group's in each row."""
    results = [{
        "metadata_fields": doc(1547, "Bug-Jargal", 1820),
        "count": 30,
        "break_up_field": [
            {"count": 20, "metadata_fields": doc(2257, "Les misérables", 1862)},
            {"count": 10, "metadata_fields": doc(1547, "Bug-Jargal", 1820)},
        ],
    }]
    rows = list(csv.DictReader(io.StringIO(aggregation_to_csv(results, "title", "author"))))
    assert [(r["title"], r["year"], r["philo_doc_id"], r["count"]) for r in rows] == [
        ("Les misérables", "1862", "2257", "20"),
        ("Bug-Jargal", "1820", "1547", "10"),
    ]
    assert all(r["author"] == "Hugo, Victor" and r["group_count"] == "30" for r in rows)
