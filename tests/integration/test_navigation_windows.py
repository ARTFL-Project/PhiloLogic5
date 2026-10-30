"""Integration tests for the navigation report's windows: a client that can show a long text object a part at a time
(windowed=1) gets one window of it, from a fixed partition of the object ending at paragraphs; any other client gets
the whole object as before. The test objects are cut into windows of a few kilobytes."""

import json
import os
import urllib.parse
from pathlib import Path

import pytest
import regex as re

from philologic.runtime import WebConfig, WSGIHandler
from philologic.runtime.exceptions import BadRequest
from philologic.runtime.get_text import _get_mmap, get_text_obj, text_file_path, window_bounds
from philologic.runtime.reports.navigation import generate_text_object

LENGTH = 5000  # bytes: the test objects are tens of kilobytes


def config_of(db, **values):
    config = WebConfig(os.path.dirname(os.path.normpath(db.path)))
    for key, value in values.items():
        config[key] = value
    return config


def navigate(db, query, length=LENGTH, note=False):
    config = config_of(db, navigation_window_length=length)
    request = WSGIHandler({"QUERY_STRING": urllib.parse.urlencode(query, doseq=True), "PHILOLOGIC_DBURL": ""}, config)
    return generate_text_object(request, config, note=note)


def long_division(db, low=40000, high=80000):
    """A division of low to high bytes with paragraphs all along"""
    for row in db.dbh.execute(
        "SELECT philo_id FROM toms WHERE philo_type IN ('div1', 'div2', 'div3') AND philo_name != '__philo_virtual' "
        "AND CAST(end_byte AS INTEGER) - CAST(start_byte AS INTEGER) BETWEEN ? AND ? ORDER BY rowid",
        (low, high),
    ):
        obj = db[row["philo_id"]]
        if len(paragraphs(db, obj)) >= 30:
            return obj
    pytest.skip("no long division")


def paragraphs(db, obj):
    start, end = int(obj.start_byte), int(obj.end_byte)
    doc = obj.philo_id[0]
    return sorted(
        int(row[0])
        for row in db.dbh.execute(
            "SELECT start_byte FROM toms WHERE philo_type = 'para' AND philo_id LIKE ?", (f"{doc} %",)
        )
        if start < int(row[0]) < end
    )


def short_id(obj):
    return " ".join(str(i) for i in obj.philo_id).rstrip(" 0")


def raw(db, obj):
    config = config_of(db)
    return _get_mmap(text_file_path(obj, config))


WORD_AFTER_TAG = re.compile(rb">\s*([A-Za-z]{3,})")


def word_starts(db, obj, bounds, window):
    """Byte offsets of words in a window, found right after the end of a tag: none is inside a tag"""
    mm = raw(db, obj)
    start, end = bounds[window], bounds[window + 1]
    return [start + m.start(1) for m in WORD_AFTER_TAG.finditer(mm[start:end])]


@pytest.fixture(params=["eltec", "shakespeare"])
def corpus(request, eltec_db, shakespeare_db):
    return {"eltec": eltec_db, "shakespeare": shakespeare_db}[request.param]


@pytest.fixture
def division(corpus):
    return corpus, long_division(corpus)


@pytest.mark.integration
class TestPartition:
    def test_windows_cover_the_object(self, division):
        db, obj = division
        bounds = window_bounds(obj, config_of(db), LENGTH)
        assert bounds[0] == int(obj.start_byte) and bounds[-1] == int(obj.end_byte)
        assert len(bounds) > 5
        assert all(0 < end - start <= LENGTH for start, end in zip(bounds, bounds[1:]))

    def test_windows_end_where_the_last_paragraph_that_fits_starts(self, division):
        db, obj = division
        bounds = window_bounds(obj, config_of(db), LENGTH)
        starts = paragraphs(db, obj)
        for start, end in zip(bounds[:-2], bounds[1:-1]):
            fitting = [p for p in starts if start < p <= start + LENGTH]
            if fitting:
                assert end == fitting[-1]

    def test_no_window_ends_inside_a_tag(self, division):
        db, obj = division
        mm = raw(db, obj)
        for cut in window_bounds(obj, config_of(db), LENGTH)[1:-1]:
            before = mm[int(obj.start_byte) : cut]
            assert before.rfind(b"<") < before.rfind(b">") or b"<" not in before

    def test_a_paragraph_starting_at_the_length_ends_the_window(self, division):
        db, obj = division
        start = int(obj.start_byte)
        paragraph = paragraphs(db, obj)[10]
        assert window_bounds(obj, config_of(db), paragraph - start)[1] == paragraph

    def test_a_paragraph_a_byte_past_the_length_does_not(self, division):
        db, obj = division
        start = int(obj.start_byte)
        paragraph = paragraphs(db, obj)[10]
        assert window_bounds(obj, config_of(db), paragraph - start - 1)[1] < paragraph

    def test_every_window_names_the_same_partition(self, division):
        db, obj = division
        bounds = window_bounds(obj, config_of(db), LENGTH)
        for part, start in enumerate(bounds[:-1], 1):
            window = navigate(db, {"philo_id": short_id(obj), "windowed": "1", "window_start": start})["window"]
            assert window == {
                "start_byte": start,
                "end_byte": bounds[part],
                "prev_start_byte": bounds[part - 2] if part > 1 else None,
                "next_start_byte": bounds[part] if part < len(bounds) - 1 else None,
                "part": part,
                "part_count": len(bounds) - 1,
            }

    def test_the_text_of_a_window_is_its_bytes(self, division):
        """The window's text is that of its bytes alone, as the whole object's is of its own"""
        db, obj = division
        bounds = window_bounds(obj, config_of(db), LENGTH)
        query = {"philo_id": short_id(obj), "windowed": "1", "window_start": bounds[2]}
        response = navigate(db, query)
        config = config_of(db)
        request = WSGIHandler({"QUERY_STRING": urllib.parse.urlencode(query), "PHILOLOGIC_DBURL": ""}, config)
        expected, _ = get_text_obj(obj, config, request, db.locals["token_regex"], window=(bounds[2], bounds[3]))
        assert response["text"] == expected
        assert len(response["text"]) < 3 * LENGTH


@pytest.mark.integration
class TestWindowChoice:
    def test_first_window_by_default(self, division):
        db, obj = division
        window = navigate(db, {"philo_id": short_id(obj), "windowed": "1"})["window"]
        assert window["part"] == 1 and window["start_byte"] == int(obj.start_byte) and window["prev_start_byte"] is None

    def test_window_start_inside_a_window_opens_it(self, division):
        db, obj = division
        bounds = window_bounds(obj, config_of(db), LENGTH)
        window = navigate(db, {"philo_id": short_id(obj), "windowed": "1", "window_start": bounds[3] + 7})["window"]
        assert window["part"] == 4 and window["start_byte"] == bounds[3]

    @pytest.mark.parametrize("where, part", [("before", 1), ("after", -1)])
    def test_window_start_outside_the_object(self, division, where, part):
        db, obj = division
        bounds = window_bounds(obj, config_of(db), LENGTH)
        window_start = 0 if where == "before" else bounds[-1] + 100
        window = navigate(db, {"philo_id": short_id(obj), "windowed": "1", "window_start": window_start})["window"]
        assert window["part"] == (part if part > 0 else len(bounds) - 1)

    def test_bad_window_start(self, division):
        db, obj = division
        with pytest.raises(BadRequest, match="window_start must be a whole number"):
            navigate(db, {"philo_id": short_id(obj), "windowed": "1", "window_start": "abc"})

    def test_the_first_hit_opens_its_window(self, division):
        db, obj = division
        bounds = window_bounds(obj, config_of(db), LENGTH)
        hit = word_starts(db, obj, bounds, 3)[1]
        response = navigate(db, {"philo_id": short_id(obj), "windowed": "1", "byte": hit})
        assert response["window"]["part"] == 4
        assert response["text"].count('class="highlight"') == 1

    def test_hits_outside_the_window_are_left_out(self, division):
        db, obj = division
        bounds = window_bounds(obj, config_of(db), LENGTH)
        first, later = word_starts(db, obj, bounds, 2)[0], word_starts(db, obj, bounds, 4)[0]
        response = navigate(db, {"philo_id": short_id(obj), "windowed": "1", "byte": [later, first]})
        assert response["window"]["part"] == 3
        assert response["text"].count('class="highlight"') == 1
        whole = navigate(db, {"philo_id": short_id(obj), "byte": [later, first]})
        assert whole["text"].count('class="highlight"') == 2

    def test_window_start_wins_over_hits(self, division):
        db, obj = division
        bounds = window_bounds(obj, config_of(db), LENGTH)
        hit = word_starts(db, obj, bounds, 3)[0]
        response = navigate(db, {"philo_id": short_id(obj), "windowed": "1", "byte": hit, "window_start": bounds[1]})
        assert response["window"]["part"] == 2
        assert 'class="highlight"' not in response["text"]

    def test_passage_opens_its_window_and_is_cut_to_each(self, division):
        db, obj = division
        bounds = window_bounds(obj, config_of(db), LENGTH)
        start, end = word_starts(db, obj, bounds, 2)[-1], word_starts(db, obj, bounds, 3)[1]
        query = {"philo_id": short_id(obj), "windowed": "1", "start_byte": start, "end_byte": end}
        opened = navigate(db, query)
        assert opened["window"]["part"] == 3
        assert f'data-offsets="{start}-{bounds[3]}"' in opened["text"]
        following = navigate(db, {**query, "window_start": bounds[3]})
        assert f'data-offsets="{bounds[3]}-{end}"' in following["text"]
        assert "passage-marker" not in navigate(db, {**query, "window_start": bounds[0]})["text"]


@pytest.mark.integration
class TestUnchanged:
    """Old clients don't ask for windows, and objects that fit are sent whole: as before windows existed"""

    def whole(self, db, obj, query):
        config = config_of(db)
        request = WSGIHandler({"QUERY_STRING": urllib.parse.urlencode(query, doseq=True), "PHILOLOGIC_DBURL": ""}, config)
        text, imgs = get_text_obj(obj, config, request, db.locals["token_regex"])
        return text

    def test_without_windowed_the_whole_object(self, division):
        db, obj = division
        bounds = window_bounds(obj, config_of(db), LENGTH)
        hit = word_starts(db, obj, bounds, 3)[0]
        query = {"philo_id": short_id(obj), "byte": hit, "window_start": bounds[2]}
        response = navigate(db, query)
        assert "window" not in response
        assert response["text"] == self.whole(db, obj, query)
        assert response == navigate(db, query, length=0)

    @pytest.mark.parametrize("windowed", ["", "0", "false"])
    def test_windowed_off(self, division, windowed):
        db, obj = division
        assert "window" not in navigate(db, {"philo_id": short_id(obj), "windowed": windowed})

    def test_an_object_that_fits_is_sent_whole(self, division):
        db, obj = division
        length = int(obj.end_byte) - int(obj.start_byte)
        query = {"philo_id": short_id(obj), "windowed": "1"}
        response = navigate(db, query, length=length)
        assert "window" not in response
        assert json.dumps(response) == json.dumps(navigate(db, query, length=0))
        unflagged = navigate(db, {"philo_id": short_id(obj)}, length=length)
        assert {**response, "query": None} == {**unflagged, "query": None}
        assert "window" in navigate(db, query, length=length - 1)

    def test_default_length(self, eltec_db):
        config = config_of(eltec_db)
        assert config["navigation_window_length"] == 500000


@pytest.fixture(scope="module")
def web(eltec_db_path):
    pytest.importorskip("falcon")
    from tests.fixtures.web_app import web_app_client

    corpus = Path(eltec_db_path).parent
    with web_app_client(corpus.parent) as client:
        yield lambda path, **params: client.simulate_get(f"/philologic5/{corpus.name}/{path}", params=params)


@pytest.mark.integration
class TestThroughTheWebApp:
    """With the default length of 500 kB: some ELTeC divisions are megabytes long"""

    def longest_division(self, db):
        """The shortest division over a megabyte"""
        row = db.dbh.execute(
            "SELECT philo_id FROM toms WHERE philo_type = 'div1' AND philo_name != '__philo_virtual' "
            "AND CAST(end_byte AS INTEGER) - CAST(start_byte AS INTEGER) > 1000000 "
            "ORDER BY CAST(end_byte AS INTEGER) - CAST(start_byte AS INTEGER) LIMIT 1"
        ).fetchone()
        return db[row["philo_id"]]

    @pytest.mark.parametrize("path", ["reports/navigation.py", "scripts/get_text_object.py"])
    def test_a_window_of_a_long_division(self, web, eltec_db, path):
        obj = self.longest_division(eltec_db)
        assert int(obj.end_byte) - int(obj.start_byte) > 1000000
        philo_id = short_id(obj)
        response = web(path, philo_id=philo_id, windowed="1")
        assert response.status_code == 200
        window = response.json["window"]
        assert window["part"] == 1 and window["part_count"] > 2
        assert 0 < window["end_byte"] - window["start_byte"] <= 500000
        following = web(path, philo_id=philo_id, windowed="1", window_start=str(window["next_start_byte"]))
        assert following.json["window"]["part"] == 2

    def test_old_clients_get_the_whole_division(self, web, eltec_db):
        obj = self.longest_division(eltec_db)
        response = web("reports/navigation.py", philo_id=short_id(obj))
        assert response.status_code == 200
        assert "window" not in response.json
        assert len(response.json["text"]) > 1000000
