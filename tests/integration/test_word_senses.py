"""Integration tests for the word senses report (get_word_senses) and the passages of its examples (get_sense_passages)."""

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
from scripts.get_word_senses import get_word_senses  # noqa: E402


def call(db, script, **params):
    root = os.path.dirname(os.path.normpath(db.path))
    config = WebConfig(root)
    query = {"report": "collocation", **params}
    request = WSGIHandler({"QUERY_STRING": urllib.parse.urlencode(query), "PHILOLOGIC_DBPATH": root}, config)
    return script(request, config)


def shown(report, detail=None):
    """The senses shown at a detail (by default the report's): the splits at least that share as important as the main
    one, with those they are within, and the senses they end in."""
    nodes = report["nodes"]
    detail = report["detail"] if detail is None else detail
    revealed = set()
    for node in nodes:
        if node["children"] and node["split"]["relative"] >= detail:
            n = node
            while n is not None:
                revealed.add(n["id"])
                n = None if n["parent"] is None else nodes[n["parent"]]
    out = []

    def walk(node):
        if node["id"] in revealed:
            for child in node["children"]:
                walk(nodes[child])
        else:
            out.append(node)

    walk(nodes[0])
    return out


@pytest.mark.integration
class TestWordSenses:
    def test_the_senses_of_light(self, eltec_db):
        # The light of the sky, and lighting a candle or a pipe
        report = call(eltec_db, get_word_senses, q="lemma:light")
        senses = shown(report)
        assert len(senses) >= 2
        assert report["nodes"][0]["split"]["z"] > report["evidence"]
        words = [{w["word"] for w in node["words"]} for node in report["nodes"]]
        assert any({"sky", "moon", "sun"} & w for w in words)
        assert any({"pipe", "candle", "match"} & w for w in words)

    def test_senses_2_finds_the_senses_of_light_too(self, eltec_db):
        report = call(eltec_db, get_word_senses, q="lemma:light", senses_version="2")
        assert len(shown(report)) >= 2
        assert report["nodes"][0]["split"]["z"] > report["evidence"]
        words = [{w["word"] for w in node["words"]} for node in report["nodes"]]
        assert any({"sky", "moon", "sun"} & w for w in words)
        assert any({"pipe", "candle", "match"} & w for w in words)
        assert report != call(eltec_db, get_word_senses, q="lemma:light")  # its own, not the first's from the cache

    def test_the_report_holds_together(self, eltec_db):
        report = call(eltec_db, get_word_senses, q="lemma:light")
        orjson.dumps(report)  # what the web server sends
        nodes = report["nodes"]
        assert [n["id"] for n in nodes] == list(range(len(nodes)))
        assert nodes[0]["parent"] is None and nodes[0]["share"] == pytest.approx(1.0, abs=1e-3)
        for node in nodes:
            if node["children"]:
                assert sum(nodes[c]["share"] for c in node["children"]) == pytest.approx(node["share"], abs=1e-3)
                assert all(nodes[c]["parent"] == node["id"] and c > node["id"] for c in node["children"])
                assert node["split"]["seed"]["group"] in (0, 1, 2)
            assert 0 <= node["works"] <= report["n_works"]
        relative = [n["split"]["relative"] for n in nodes if n["children"]]
        assert max(relative) == pytest.approx(1.0)
        assert min(relative) >= report["finest"] - 1e-9
        assert sum(n["share"] for n in shown(report, 0.0)) == pytest.approx(1.0, abs=1e-3)
        assert report["time"]["start"] >= 1841 and report["time"]["end"] <= 1920
        n_years = report["time"]["end"] - report["time"]["start"] + 1
        assert len(report["time"]["share"]["value"]) == len(nodes)
        assert all(len(series) == n_years for series in report["time"]["share"]["value"])

    def test_word_groups_point_into_the_word_lists(self, eltec_db):
        report = call(eltec_db, get_word_senses, q="lemma:light")
        assert any(node["groups"] for node in report["nodes"])
        for node in report["nodes"]:
            seen = set()
            for group in node["groups"]:
                positions = [group["hub"], *group["members"]]
                assert all(0 <= p < len(node["words"]) for p in positions)
                assert not seen & set(positions)  # a word is in one group of a sense
                seen |= set(positions)

    def test_the_query_words_are_never_their_own_context(self, shakespeare_db):
        report = call(shakespeare_db, get_word_senses, q='"my lord"')
        for node in report["nodes"]:
            for group in ("words", "before", "after"):
                assert not {"my", "lord"} & {w["word"] for w in node[group]}

    def test_examples_render_as_passages(self, eltec_db):
        report = call(eltec_db, get_word_senses, q="lemma:light")
        examples = shown(report)[0]["examples"][:3]
        hits = ",".join(str(e["hit"]) for e in examples)
        passages = call(eltec_db, get_sense_passages, q="lemma:light", hits=hits)["passages"]
        assert [p["hit"] for p in passages] == [e["hit"] for e in examples]
        assert all("highlight" in p["context"] for p in passages)

    def test_no_dates_no_course_over_time(self, shakespeare_db):
        # Every play has the same year: senses, but no course over time
        report = call(shakespeare_db, get_word_senses, q="king")
        assert report["n_sentences"] > 0
        assert report["time"] is None
