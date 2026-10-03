"""Unit tests for the aggregation CSV export: each breakdown row has its own object's metadata."""

import csv
import io
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime.reports.aggregation import aggregation_to_csv, group_results


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


def histoire_de_france():
    """Bainville's book and Michelet's 3 volumes share a title; Michelet's last volume's has a period."""
    docs = {
        (1,): ("Histoire de France", "Bainville, Jacques", 1924, 52),
        (2,): ("Histoire de France", "Michelet, Jules", 1893, 8),
        (3,): ("Histoire de France", "Michelet, Jules", 1893, 15),
        (4,): ("Histoire de France", "Michelet, Jules", 1893, 18),
        (5,): ("Histoire de France.", "Michelet, Jules", 1893, 42),
    }
    metadata = {
        doc: {"title": title, "author": author, "year": year, "pub_place": "Paris", "field_name": title}
        for doc, (title, author, year, _) in docs.items()
    }
    return {doc: hits for doc, (*_, hits) in docs.items()}, metadata


@pytest.mark.unit
class TestGroupsByTitle:
    def test_same_title_one_group(self):
        """A title's documents make one row, whose link (the title) finds all its hits: by document, Michelet's
        volumes made 3 rows of 8, 15 and 18 hits whose links each found 41."""
        id_counts, metadata = histoire_de_france()
        groups = group_results(id_counts, metadata, "doc")
        assert [(g["metadata_fields"]["title"], g["count"], g["object_count"]) for g in groups] == [
            ("Histoire de France", 93, 4),
            ("Histoire de France.", 42, 1),
        ]

    def test_group_shows_what_its_documents_share(self):
        """Not one document's author and year, as if they were all the group's."""
        id_counts, metadata = histoire_de_france()
        merged, single = group_results(id_counts, metadata, "doc")
        assert merged["metadata_fields"] == {"title": "Histoire de France", "pub_place": "Paris", "field_name": "Histoire de France"}
        assert single["metadata_fields"]["author"] == "Michelet, Jules"

    def test_csv_columns_of_every_group(self):
        id_counts, metadata = histoire_de_france()
        rows = list(csv.DictReader(io.StringIO(aggregation_to_csv(group_results(id_counts, metadata, "doc")))))
        assert rows == [
            {"author": "", "pub_place": "Paris", "title": "Histoire de France", "year": "", "count": "93"},
            {"author": "Michelet, Jules", "pub_place": "Paris", "title": "Histoire de France.", "year": "1893", "count": "42"},
        ]
