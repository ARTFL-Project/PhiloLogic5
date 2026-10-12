"""Integration tests for the senses over time (get_sense_periods)."""

import os
import sys
import urllib.parse
from pathlib import Path

import orjson
import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "www"))

from philologic.runtime import WebConfig, WSGIHandler  # noqa: E402
from scripts.get_sense_passages import get_sense_passages  # noqa: E402
from scripts.get_sense_periods import get_sense_periods  # noqa: E402


def call(db, script, **params):
    root = os.path.dirname(os.path.normpath(db.path))
    config = WebConfig(root)
    query = {"report": "collocation", **params}
    request = WSGIHandler({"QUERY_STRING": urllib.parse.urlencode(query), "PHILOLOGIC_DBPATH": root}, config)
    return script(request, config)


@pytest.mark.integration
class TestSensePeriods:
    def test_the_report_holds_together(self, eltec_db):
        report = call(eltec_db, get_sense_periods, q="lemma:light")
        orjson.dumps(report)  # as the web app sends it, the first time as the next
        assert report["senses"]
        nodes = report["nodes"]
        for s in report["senses"]:
            assert nodes[s["id"]]["sense"] == s["sense"]
            periods = s["periods"]
            # In year order, each after the last
            assert all(p["start"] <= p["end"] for p in periods)
            assert all(a["end"] < b["start"] for a, b in zip(periods, periods[1:]))
            # The periods of the last number tried that held, every one tried before it holding; one when none did
            held = [ok for _, ok in s["tried"]]
            assert held == sorted(held, reverse=True) and held.count(False) <= 1
            assert len(periods) == (s["tried"][held.count(True) - 1][0] if any(held) else 1)
            if len(periods) == 1:
                assert "new" not in periods[0]
                continue
            assert "characteristic" in periods[0] and "new" not in periods[0]
            for p in periods[1:]:
                assert p["new"], "every period after the first has a new word"
                for w in p["new"]:
                    assert w["ratio"] is None or w["ratio"] > 1
                    assert w["hits"] and len(set(w["hits"])) == len(w["hits"])
                assert all(w["ratio"] is not None and w["ratio"] < 1 for w in p["fading"])
            for p in periods:
                authors = [w["author"] for w in p.get("representative", [])]
                assert len(set(authors)) == len(authors)  # one work for an author
                assert all(w["hits"] for w in p.get("representative", []))

    def test_a_words_sentences_render_as_passages(self, eltec_db):
        report = call(eltec_db, get_sense_periods, q="lemma:light")
        s = next(s for s in report["senses"] if len(s["periods"]) > 1)
        w = s["periods"][1]["new"][0]
        passages = call(eltec_db, get_sense_passages, q="lemma:light", hits=",".join(map(str, w["hits"][:3])))[
            "passages"
        ]
        assert [p["hit"] for p in passages] == w["hits"][:3]


@pytest.mark.integration
def test_no_dates_no_periods(shakespeare_db):
    # Every play has the same year: the senses, but no periods
    report = call(shakespeare_db, get_sense_periods, q="king")
    assert report["n_sentences"] > 0 and report["nodes"] and report["senses"] == []
