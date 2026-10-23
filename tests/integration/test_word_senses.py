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

    def test_senses_keep_their_numbers_at_any_detail(self, shakespeare_db):
        from philologic.runtime.reports.word_senses import MOST, elbow, leaves_at

        report = call(shakespeare_db, get_word_senses, q='"my lord"')
        common = {w.removeprefix("lemma:") for w in report["filter_list"]}
        nodes = report["nodes"]
        levels = sorted({n["split"]["relative"] for n in nodes if n["children"]}, reverse=True)
        before = set()
        for detail in levels:
            leaves = leaves_at(nodes, detail)
            numbers = [n["sense"] for n in leaves if n["sense"] is not None]
            assert len(numbers) == len(set(numbers)) and before <= set(numbers)
            top = sorted(int(x.split(".")[0]) for x in numbers if "." not in x)
            assert top == list(range(1, len(top) + 1))
            for n in leaves:  # the rest of the sentences has nothing of its own to show but the filter's words
                if n["sense"] is None:
                    assert not n["words"] and {w["word"] for w in n["before"] + n["after"]} <= common
            before = set(numbers)
        covered = sum(n["share"] for n in leaves_at(nodes, report["detail"]) if n["sense"] is not None)
        relative = sorted((n["split"]["relative"] for n in nodes if n["children"]), reverse=True)
        assert covered > MOST or report["detail"] == relative[elbow(relative) - 1]

    def test_words_next_to_the_query_read_as_the_text_has_them(self, eltec_db, shakespeare_db):
        light = call(eltec_db, get_word_senses, q="lemma:light")["nodes"][0]
        assert {w["word"]: w.get("phrase") for w in light["before"]}["the"] == "the light"
        assert {w["word"]: w.get("phrase") for w in light["after"]}["up"] == "lit up"  # the forms, not the lemmas
        lord = call(shakespeare_db, get_word_senses, q='"my lord"')
        assert {w["word"]: w.get("phrase") for w in lord["nodes"][0]["before"]}["good"] == "good my lord"
        for node in lord["nodes"]:
            for w in node["before"] + node["after"]:
                if "phrase" in w:  # one space between words, lower case, the query in it
                    assert w["phrase"] == " ".join(w["phrase"].split()) == w["phrase"].lower()
                    assert "my lord" in w["phrase"]

    def test_the_query_words_are_never_their_own_context(self, shakespeare_db):
        report = call(shakespeare_db, get_word_senses, q='"my lord"')
        for node in report["nodes"]:
            for group in ("words", "before", "after"):
                assert not {"my", "lord"} & {w["word"] for w in node[group]}

    def test_sentences_with_no_word_on_one_side_of_the_query_read_no_phrase(self, eltec_db):
        from philologic.runtime.reports.word_senses import _Phrases, query_sentences

        sentences = query_sentences(eltec_db, "lemma:light", True)
        before = set(sentences["neighbors"][1]["unit"].tolist())
        first = next(u for u in range(sentences["n_units"]) if u not in before)  # the query opens the sentence
        near = _Phrases(eltec_db, sentences).near([first])
        assert near[1] == {} and near[2]

    def test_the_filters_words_are_left_out_of_the_sentences_words_but_not_next_to_the_word(self, eltec_db):
        from philologic.runtime.reports.word_senses import query_sentences

        report = call(eltec_db, get_word_senses, q="lemma:light")
        common = {w.removeprefix("lemma:") for w in report["filter_list"]}
        nodes = report["nodes"]
        in_sentence = [node["words"] + [w for w in node["rarer"] if w["group"] == 0] for node in nodes]
        assert not any(common & {w["word"] for w in words} for words in in_sentence)
        assert any(common & {w["word"] for w in node["before"] + node["after"]} for node in nodes)
        sentences = query_sentences(eltec_db, "lemma:light", True, filter_words=report["filter_list"])
        (X, ids), before, after = sentences["channels"]
        assert not sentences["filtered"][ids].any()
        assert sentences["filtered"][before[1]].any() and sentences["filtered"][after[1]].any()

    def test_the_filters_words_seed_no_split(self, eltec_db):
        report = call(eltec_db, get_word_senses, q="lemma:light")
        common = {w.removeprefix("lemma:") for w in report["filter_list"]}
        seeds = [node["split"]["seed"] for node in report["nodes"] if node["children"]]
        assert seeds and not any(s["word"] in common for s in seeds)

    def test_examples_render_as_passages(self, eltec_db):
        report = call(eltec_db, get_word_senses, q="lemma:light")
        examples = shown(report)[0]["examples"][:3]
        hits = ",".join(str(e["hit"]) for e in examples)
        passages = call(eltec_db, get_sense_passages, q="lemma:light", hits=hits)["passages"]
        assert [p["hit"] for p in passages] == [e["hit"] for e in examples]
        assert all("highlight" in p["context"] for p in passages)

    def test_a_senses_words_bring_its_sentences_with_them(self, eltec_db):
        from philologic.runtime.reports.word_senses import _WORD_EXAMPLES

        report = call(eltec_db, get_word_senses, q="lemma:light")
        for node in report["nodes"]:
            for w in node["words"] + node["before"] + node["after"]:
                assert 0 < len(w["hits"]) <= _WORD_EXAMPLES and len(set(w["hits"])) == len(w["hits"])
        # Each is a sentence of the query with the word (the passages shown around a hit may be too short to show it)
        from philologic.runtime.reports.word_senses import query_sentences

        sentences = query_sentences(eltec_db, "lemma:light", True)
        X, ids = sentences["channels"][0]
        blob, offsets = sentences["v_blob"], sentences["v_offsets"]
        pipe = next(k for k, i in enumerate(ids) if blob[offsets[i] : offsets[i + 1]] == b"lemma:pipe")
        unit = {int(hit): u for u, hit in enumerate(sentences["first_hit"])}
        w = next(w for n in report["nodes"] if n["sense"] for w in n["words"] if w["word"] == "pipe")
        assert all(X[unit[hit], pipe] > 0 for hit in w["hits"])

    def test_no_dates_no_course_over_time(self, shakespeare_db):
        # Every play has the same year: senses, but no course over time
        report = call(shakespeare_db, get_word_senses, q="king")
        assert report["n_sentences"] > 0
        assert report["time"] is None
