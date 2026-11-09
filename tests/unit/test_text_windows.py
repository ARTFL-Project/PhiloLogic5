"""Unit tests for the windows of the navigation report on a plain-text book, as plain-text databases have them: one
document of one division holding all its paragraphs, so that its single division is the whole book. Each test makes
a small database of it, and cuts it into windows of a few hundred bytes."""

import os
import sqlite3
import sys
import urllib.parse
from pathlib import Path

import pytest
import regex as re

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime import WebConfig, WSGIHandler
from philologic.runtime import get_text as get_text_module
from philologic.runtime.DB import DB
from philologic.runtime.get_text import window_bounds
from philologic.runtime.reports.navigation import generate_text_object

COLUMNS = ["philo_type", "philo_name", "philo_id", "start_byte", "end_byte", "filename", "title", "author", "id"]


def paragraphs_of(text):
    """The start bytes of the paragraphs of a plain text: its blocks of lines between blank lines"""
    starts = [0]
    position = text.find(b"\n\n")
    while position != -1:
        while position < len(text) and text[position : position + 1] == b"\n":
            position += 1
        if position < len(text):
            starts.append(position)
        position = text.find(b"\n\n", position)
    return starts


def book_rows(doc, filename, text, paragraphs, notes=()):
    """toms rows of a plain-text book: a document, its virtual div1, div2 and div3, and the paragraphs; notes are
    (start, end, id) of paragraphs of notes, in the order of the bytes"""
    end = len(text)
    rows = [
        ("doc", "text", f"{doc} 0 0 0 0 0 0", 0, end, filename, f"Book {doc}", "Anonymous", None),
        ("div1", "__philo_virtual", f"{doc} 1 0 0 0 0 0", 0, end, None, None, None, None),
        ("div2", "__philo_virtual", f"{doc} 1 1 0 0 0 0", 0, end, None, None, None, None),
        ("div3", "__philo_virtual", f"{doc} 1 1 1 0 0 0", 0, end, None, None, None, None),
    ]
    ends = paragraphs[1:] + [end]
    note_ids = {start: (note_end, note_id) for start, note_end, note_id in notes}
    for number, (start, para_end) in enumerate(zip(paragraphs, ends), 1):
        if start in note_ids:
            rows.append(("para", "note", f"{doc} 1 1 1 {number} 0 0", start, note_ids[start][0], None, None, None,
                         note_ids[start][1]))
        else:
            rows.append(("para", "p", f"{doc} 1 1 1 {number} 0 0", start, para_end, None, None, None, None))
    return rows


def make_database(root, text, paragraphs=None, other_book=None, notes=(), pages=(), web_config=""):
    """A database of the book of text, and of another book after it (the bytes of its paragraphs, over a text as
    long), whose paragraphs would end the windows elsewhere if they were taken for the first book's. pages are the
    (start, end) of the book's pages, with the images p1, p2..."""
    data = root / "data"
    (data / "TEXT").mkdir(parents=True)
    (data / "TEXT" / "book.txt").write_bytes(text)
    (data / "db.locals.py").write_text(
        'metadata_fields = ["title", "author", "filename"]\n'
        'metadata_types = {"title": "doc", "author": "doc", "filename": "doc"}\n'
        f"metadata_sql_types = {dict.fromkeys(COLUMNS, 'text')!r}\n"
        'token_regex = "[\\\\p{L}\\\\p{M}\\\\p{N}]+"\n'
    )
    (data / "web_config.cfg").write_text('dbname = "plain"\n' + web_config)
    rows = book_rows(1, "book.txt", text, paragraphs_of(text) if paragraphs is None else paragraphs, notes)
    if other_book is not None:
        (data / "TEXT" / "other.txt").write_bytes(b"x" * len(text))
        rows += book_rows(2, "other.txt", b"x" * len(text), other_book)
    dbh = sqlite3.connect(data / "toms.db")
    dbh.execute(f"CREATE TABLE toms ({', '.join(c + ' text' for c in COLUMNS)})")
    dbh.execute("CREATE INDEX toms_philo_id_index ON toms (philo_id)")
    dbh.execute("CREATE INDEX toms_philo_type_index ON toms (philo_type)")
    dbh.executemany(f"INSERT INTO toms VALUES ({', '.join('?' * len(COLUMNS))})", [tuple(map(str_or_none, r)) for r in rows])
    dbh.execute("CREATE TABLE pages (philo_type text, philo_id text, n text, facs text, id text, start_byte text, end_byte text)")
    dbh.executemany(
        "INSERT INTO pages VALUES ('page', ?, ?, ?, NULL, ?, ?)",
        [(f"1 0 0 0 0 0 0 0 {n}", str(n), f"p{n}", str(start), str(end)) for n, (start, end) in enumerate(pages, 1)],
    )
    dbh.commit()
    dbh.close()
    return root


def str_or_none(value):
    return None if value is None else str(value)


@pytest.fixture(autouse=True)
def empty_caches():
    get_text_module._window_cache.clear()
    get_text_module._mmap_cache.clear()
    yield
    get_text_module._window_cache.clear()
    get_text_module._mmap_cache.clear()


def bounds_of(root, length, philo_id="1 1"):
    config = WebConfig(str(root))
    db = DB(str(root / "data"))
    return window_bounds(db[philo_id], config, length)


def navigate(root, length, note=False, **query):
    config = WebConfig(str(root))
    config["navigation_window_length"] = length
    request = WSGIHandler({"QUERY_STRING": urllib.parse.urlencode(query, doseq=True), "PHILOLOGIC_DBURL": ""}, config)
    return generate_text_object(request, config, note=note)


def check_partition(bounds, text, length):
    assert bounds[0] == 0 and bounds[-1] == len(text)
    assert all(0 < end - start <= length for start, end in zip(bounds, bounds[1:]))


def lines(count, width=60, word=b"word"):
    line = (word + b" ") * (width // (len(word) + 1))
    return b"".join(line.strip() + b"\n" for _ in range(count))


BOOK = b"".join(b"Chapter %d\n\n" % n + lines(4) + b"\n" for n in range(1, 41))


@pytest.mark.unit
class TestPlainTextPartition:
    def test_windows_end_at_the_last_paragraph_that_fits(self, tmp_path):
        root = make_database(tmp_path, BOOK)
        paragraphs = paragraphs_of(BOOK)
        bounds = bounds_of(root, 1000)
        check_partition(bounds, BOOK, 1000)
        assert len(bounds) > 10
        for start, end in zip(bounds[:-2], bounds[1:-1]):
            assert end == max(p for p in paragraphs if start < p <= start + 1000)

    def test_the_paragraphs_of_the_next_book_are_not_its_own(self, tmp_path):
        """Its rows end where the next document's start: the other book's paragraphs, over bytes of the same numbers,
        start every 7 bytes"""
        root = make_database(tmp_path, BOOK, other_book=list(range(0, len(BOOK), 7)))
        paragraphs = set(paragraphs_of(BOOK))
        assert set(bounds_of(root, 1000)[1:-1]) <= paragraphs

    def test_a_paragraph_longer_than_a_window_is_cut_at_a_line_break(self, tmp_path):
        text = b"Title\n\n" + lines(60) + b"\nEnd\n"
        root = make_database(tmp_path, text)
        bounds = bounds_of(root, 1000)
        check_partition(bounds, text, 1000)
        assert len(bounds) > 4
        for cut in bounds[2:-2]:  # past the title, and before the paragraph at the end
            assert text[cut - 1 : cut] == b"\n"

    def test_the_last_line_break_that_fits(self, tmp_path):
        text = lines(60)
        root = make_database(tmp_path, text, paragraphs=[0])
        bounds = bounds_of(root, 1000)
        for start, end in zip(bounds[:-2], bounds[1:-1]):
            assert end == text.rfind(b"\n", start, start + 1000) + 1

    def test_a_line_longer_than_a_window_is_cut_at_a_space(self, tmp_path):
        text = b"Title\n\n" + b"word " * 600 + b"\n"
        root = make_database(tmp_path, text)
        bounds = bounds_of(root, 1000)
        check_partition(bounds, text, 1000)
        for start, end in zip(bounds[1:-2], bounds[2:-1]):
            assert text[end - 1 : end] == b" " and end == text.rfind(b" ", start, start + 1000) + 1

    def test_without_spaces_no_character_cut_in_two(self, tmp_path):
        text = "é".encode("utf8") * 1500
        root = make_database(tmp_path, text, paragraphs=[0])
        bounds = bounds_of(root, 1001)
        check_partition(bounds, text, 1001)
        for start, end in zip(bounds, bounds[1:]):
            assert text[start:end].decode("utf8") == "é" * ((end - start) // 2)

    def test_no_cut_inside_a_tag(self, tmp_path):
        """The line break inside the page tag is the last before the end of the window: the window ends at the one
        before"""
        head = lines(10)
        tag = b'<pb n="12"\nfacs="p12.png"/>'
        text = head + b"words before the page " + tag + b" and after it\n" + lines(20)
        root = make_database(tmp_path, text, paragraphs=[0])
        length = len(head) + len(b"words before the page ") + 15
        assert text.rfind(b"\n", 0, length) > len(head)  # the one inside the tag
        bounds = bounds_of(root, length)
        assert bounds[1] == len(head)

    def test_a_less_than_sign_in_text_opens_no_tag(self, tmp_path):
        text = lines(5) + b"1 < 2 and so on\n" + lines(20)
        root = make_database(tmp_path, text, paragraphs=[0])
        length = len(lines(5)) + len(b"1 < 2 and so on\n") + 10
        assert bounds_of(root, length)[1] == length - 10

    def test_no_entity_cut_in_two(self, tmp_path):
        text = b"&amp;" * 400
        root = make_database(tmp_path, text, paragraphs=[0])
        bounds = bounds_of(root, 1003)
        assert all(cut % 5 == 0 for cut in bounds)

    def test_a_tag_longer_than_a_window_ends_it(self, tmp_path):
        """Where no window of the length can end, it ends after the tag"""
        tag = b'<graphic url="' + b"A" * 1500 + b'"/>'
        text = tag + b" " + lines(40)
        root = make_database(tmp_path, text, paragraphs=[0])
        bounds = bounds_of(root, 1000)
        assert bounds[1] == len(tag)
        assert all(0 < end - start <= 1000 for start, end in zip(bounds[1:], bounds[2:]))

    def test_an_object_that_fits_is_one_window(self, tmp_path):
        root = make_database(tmp_path, BOOK)
        assert bounds_of(root, len(BOOK)) == [0, len(BOOK)]

    def test_partition_kept_until_the_files_change(self, tmp_path):
        root = make_database(tmp_path, BOOK)
        first = bounds_of(root, 1000)
        assert bounds_of(root, 1000) is first
        os.utime(root / "data" / "toms.db", ns=(1, 1))
        assert bounds_of(root, 1000) is not first
        assert bounds_of(root, 1000) == first


@pytest.mark.unit
class TestPlainTextNavigation:
    def test_the_part_of_the_hit(self, tmp_path):
        root = make_database(tmp_path, BOOK)
        bounds = bounds_of(root, 1000)
        hit = BOOK.index(b"Chapter 30\n")
        response = navigate(root, 1000, philo_id="1 1", windowed="1", byte=str(hit))
        window = response["window"]
        assert window["start_byte"] <= hit < window["end_byte"]
        assert window["part"] == bounds.index(window["start_byte"]) + 1
        assert window["part_count"] == len(bounds) - 1
        assert response["text"].count('class="highlight"') == 1
        assert ">Chapter</span> 30" in response["text"]
        assert "Chapter 1\n" not in response["text"]

    @pytest.mark.parametrize("cut", [False, True], ids=["in_the_part", "cut_to_the_part"])
    def test_a_passage_after_highlights_in_a_part(self, tmp_path, cut):
        """The highlights of the hits before the bounds of a passage don't move them in a part either, where the
        passage may be cut to the part: the passage spans have the text of its bytes in the part"""
        root = make_database(tmp_path, BOOK)
        bounds = bounds_of(root, 1000)
        part_start, part_end = bounds[3], bounds[4]
        start = BOOK.index(b"word", part_start + 300)
        end = BOOK.index(b"word", part_end + 100) if cut else BOOK.index(b"word", part_start + 700)
        hits = [BOOK.index(b"word", part_start + offset) for offset in (20, 100, 200, start - part_start + 50)]
        query = {"philo_id": "1 1", "windowed": "1", "window_start": str(part_start)}
        query["byte"] = [str(hit) for hit in hits]
        response = navigate(root, 1000, start_byte=str(start), end_byte=str(end), **query)
        assert response["window"]["start_byte"] == part_start
        text = response["text"]
        in_passage = "".join(re.findall(r'<span class="passage-0" n="0">([^<]*)</span>', text))
        assert in_passage == BOOK[start : min(end, part_end)].decode()
        assert f'data-offsets="{start}-{min(end, part_end)}"' in text
        assert text.count('<span class="highlight">') == 4
        assert re.sub(r"<[^>]+>", "", text) == re.sub(r"<[^>]+>", "", navigate(root, 1000, **query)["text"])

    def test_old_clients_get_the_whole_book(self, tmp_path):
        root = make_database(tmp_path, BOOK)
        response = navigate(root, 1000, philo_id="1 1", byte=str(BOOK.index(b"Chapter 30\n")))
        assert "window" not in response
        assert "Chapter 1\n" in response["text"] and "Chapter 40\n" in response["text"]

    def test_a_part_opens_with_the_image_of_its_page(self, tmp_path):
        """Not with that of the page the book starts on"""
        pages = [(start, min(start + 700, len(BOOK))) for start in range(0, len(BOOK), 700)]
        config = 'page_images_url_root = "https://images.example.org"\npage_image_extension = ".jpg"\n'
        root = make_database(tmp_path, BOOK, pages=pages, web_config=config)
        bounds = bounds_of(root, 1000)
        response = navigate(root, 1000, philo_id="1 1", windowed="1", window_start=str(bounds[3]))
        page = next(n for n, (start, end) in enumerate(pages, 1) if end >= bounds[3])
        assert page > 1
        assert response["text"].startswith(f'<span class="xml-pb-image"><a href="https://images.example.org/p{page}.jpg"')
        assert response["imgs"]["current_obj_img"] == [f"p{page}"]
        whole = navigate(root, 1000, philo_id="1 1")
        assert whole["text"].startswith('<span class="xml-pb-image"><a href="https://images.example.org/p1.jpg"')

    def test_notes_are_not_windowed(self, tmp_path):
        note = b"A note. " * 300
        text = BOOK + note + b"\n"
        paragraphs = paragraphs_of(BOOK) + [len(BOOK)]
        root = make_database(tmp_path, text, paragraphs=paragraphs, notes=[(len(BOOK), len(text), "n1")])
        response = navigate(root, 1000, note=True, philo_id="1 1", target="#n1", windowed="1")
        assert "window" not in response
        assert response["text"].count("A note.") == 300

    def test_tags_lxml_makes_no_element_of(self, tmp_path):
        """The etymologies of an OCR'd dictionary (The Century Dictionary, in scold_bridle) have "<abducere: ", which
        both the fragment parser and its HTML fallback took for a tag of no name: the book gave a 500"""
        text = BOOK + b"abduction (ab-duk'shon), n. [< L. abductio{)U\n<abducere: see abduce.]\t1. The act of\n"
        root = make_database(tmp_path, text)
        response = navigate(root, 0, philo_id="1 1")
        assert "\n&lt;abducere: see abduce.]\t1. The act of\n" in response["text"]
        assert "Chapter 40\n" in response["text"]
