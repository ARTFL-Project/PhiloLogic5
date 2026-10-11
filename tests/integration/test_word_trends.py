"""Integration tests for the word trends reports: rising words (get_rising_words) and the works ahead of their time
(get_ahead_of_time)."""

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
from scripts.get_ahead_of_time import get_ahead_of_time  # noqa: E402
from scripts.get_rising_words import get_rising_words  # noqa: E402
from scripts.get_sense_passages import get_sense_passages  # noqa: E402


def call(db, script, **params):
    root = os.path.dirname(os.path.normpath(db.path))
    config = WebConfig(root)
    query = {"report": "collocation", **params}
    request = WSGIHandler({"QUERY_STRING": urllib.parse.urlencode(query), "PHILOLOGIC_DBPATH": root}, config)
    return script(request, config)


@pytest.mark.integration
class TestRisingWords:
    def test_the_report_holds_together(self, eltec_db):
        report = call(eltec_db, get_rising_words, q="lemma:light")
        orjson.dumps(report)  # as the web app sends it, the first time as the next
        assert report["n_sentences"] > 0 and report["time"] is not None
        assert 0 < report["rising"] <= report["candidates"]
        assert report["uses"] and len(report["time"]["share"]["value"]) == len(report["uses"])
        start, end = report["time"]["start"], report["time"]["end"]
        for use in report["uses"]:
            assert use["words"] and use["n_words"] >= len(use["words"])
            assert start <= use["onset"] <= end
            if use["rises"]:
                assert use["after"] > use["before"]
            years = [s["year"] for s in use["sources"]]
            assert years == sorted(years)
            for w in use["words"]:
                assert w["after"] > w["before"] and start <= w["onset"] <= end
                assert not w["new_word"] or w["corpus_before"] == 0

    def test_a_uses_earliest_sentences_render_as_passages(self, eltec_db):
        report = call(eltec_db, get_rising_words, q="lemma:light")
        sources = report["uses"][0]["sources"][:3]
        hits = ",".join(str(s["hit"]) for s in sources)
        passages = call(eltec_db, get_sense_passages, q="lemma:light", hits=hits)["passages"]
        assert [p["hit"] for p in passages] == [s["hit"] for s in sources]


@pytest.mark.integration
class TestAheadOfTime:
    def test_the_report_holds_together(self, eltec_db):
        from philologic.runtime.reports.word_trends import _FOLLOWERS, _SOURCES

        report = call(eltec_db, get_ahead_of_time, q="lemma:light")
        orjson.dumps(report)
        assert report["tested"] > 0 and report["threshold"] > report["evidence"]
        assert sum(p["tested"] for p in report["periods"]) == report["tested"]
        assert sum(p["clear"] for p in report["periods"]) == report["n_clear"] >= len(report["clear"])
        assert sum(p["perhaps"] for p in report["periods"]) == report["n_perhaps"] >= len(report["perhaps"])
        assert all(w["z"] > report["threshold"] for w in report["clear"])
        assert all(report["evidence"] < w["z"] <= report["threshold"] for w in report["perhaps"])
        # The clear ones the most consistently ahead first, the others the most surely ahead first
        assert [w["share"] for w in report["clear"]] == sorted((w["share"] for w in report["clear"]), reverse=True)
        assert [w["z"] for w in report["perhaps"]] == sorted((w["z"] for w in report["perhaps"]), reverse=True)
        for w in report["clear"] + report["perhaps"]:
            assert 0.5 < w["share"] <= 1 and w["before"] >= 2 and w["after"] >= 2
            assert 0 < len(w["sources"]) <= min(_SOURCES, w["sentences"])
            assert len({s["hit"] for s in w["sources"]}) == len(w["sources"])
            assert 0 < len(w["followers"]) <= _FOLLOWERS
            for f in w["followers"]:  # later works of the span, by other authors
                assert w["year"] < f["year"] <= w["year"] + report["span"]
                assert not w["author"] or f["author"] != w["author"]

    def test_the_electric_light_is_ahead_of_its_time(self, eltec_db):
        # Novels of the 1890s light their rooms with electric lights, as those after them do and those before do not
        report = call(eltec_db, get_ahead_of_time, q="lemma:light")
        ahead = [w for w in report["clear"] + report["perhaps"] if any(x["word"] == "electric" for x in w["words"])]
        assert ahead and all(w["year"] >= 1890 for w in ahead)

    def test_a_works_sentences_render_as_passages(self, eltec_db):
        report = call(eltec_db, get_ahead_of_time, q="lemma:light")
        work = (report["clear"] + report["perhaps"])[0]
        hits = ",".join(str(s["hit"]) for s in work["sources"])
        passages = call(eltec_db, get_sense_passages, q="lemma:light", hits=hits)["passages"]
        assert {p["hit"] for p in passages} == {s["hit"] for s in work["sources"]}


@pytest.mark.integration
def test_no_dates_no_trends(shakespeare_db):
    # Every play has the same year: nothing rises, no work is set against others before and after it
    rising = call(shakespeare_db, get_rising_words, q="king")
    assert rising["n_sentences"] > 0 and rising["time"] is None and rising["uses"] == []
    ahead = call(shakespeare_db, get_ahead_of_time, q="king")
    assert ahead["n_sentences"] > 0 and ahead["tested"] == 0 and ahead["clear"] == ahead["perhaps"] == []
