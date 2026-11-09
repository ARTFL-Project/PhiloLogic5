"""Unit tests for the page markers of plain texts (page_marker_regex) at display time: the text page shows the lines
the plain-text parser made page breaks of as it shows the <pb> tags of XML texts, and concordances and KWIC leave them
out, as they leave those out. Each test makes a small database of a HathiTrust-like book."""

import io
import json
import sqlite3
import sys
import urllib.parse
from pathlib import Path
from types import SimpleNamespace

import pytest
import regex as re

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.loadtime.PlainTextParser import PlainTextParser
from philologic.runtime import WebConfig, WSGIHandler
from philologic.runtime import get_text as get_text_module
from philologic.runtime.DB import DB
from philologic.runtime.get_text import get_concordance_text, window_bounds
from philologic.runtime.ObjectFormatter import format_concordance, format_concordance_distinctive, format_strip
from philologic.runtime.reports import kwic as kwic_module
from philologic.runtime.reports.kwic import hit_bounds, kwic_hit_object
from philologic.runtime.reports.navigation import generate_text_object

HATHI_MARKERS = r"^\{(?:[^{}]*_)?(\d+)\}$"
TOKEN_REGEX = r"[\p{L}\p{M}\p{N}]+|[&\p{L};]+"
TAGS = re.compile(r"<[^>]+>")
COLUMNS = ["philo_type", "philo_name", "philo_id", "start_byte", "end_byte", "filename", "title", "author", "head"]

# A book whose @247, @248 and @249 are page markers: those of HathiTrust in PLAIN, <pb> tags in XML
BOOK = (
    "\n"
    "@247\n"
    "\n"
    "N° 1431] THE ATHENÆUM 375\n"
    "Journal of literature, with {braces} in it.\n"
    "\n"
    "@248\n"
    "\n"
    "Café ſont here, and there.\n"
    "A line ending a page\n"
    "   @249  \n"
    "the paragraph goes on: {00000250} is no marker,\n"
    "{nor_this}\n"
    "\n"
    "Last words.\n"
)
MARKERS = {"@247": "{UCAL_C3470619_00000247}", "@248": "{00000248}", "@249": "{39002007368898_000249}"}
PLAIN = BOOK
XML = BOOK
for placeholder, marker in MARKERS.items():
    PLAIN = PLAIN.replace(placeholder, marker)
    XML = XML.replace(placeholder, f'<pb n="{placeholder[1:]}"/>')


def pb_span(n):
    """A page break as the text page shows those of XML texts, with no page images"""
    return f'<span n="{n}" class="xml-pb-image">--{n}--</span>'


def parsed_rows(text, page_marker_regex, section_length):
    """The toms and pages rows of the objects and pages the plain-text parser makes of text"""
    output = io.StringIO()
    parser = PlainTextParser(
        output, 1, len(text.encode()), known_metadata={"filename": "book.txt", "title": "Book", "author": "Anonymous"},
        page_marker_regex=page_marker_regex, plain_text_section_length=section_length,
    )  # fmt: skip
    parser.parse(io.StringIO(text, newline=""))
    toms, pages = [], []
    for line in output.getvalue().splitlines():
        philo_type, name, philo_id, attrib = line.split("\t")
        attrib = json.loads(attrib)
        if philo_type == "page":
            pages.append((philo_id, attrib["n"], attrib["start_byte"], attrib["end_byte"]))
        elif philo_type in ("doc", "div1", "div2", "div3", "para"):
            philo_id = " ".join(philo_id.split()[:7])
            toms.append(
                (philo_type, name, philo_id, attrib["start_byte"], attrib["end_byte"], attrib.get("filename"),
                 attrib.get("title"), attrib.get("author"), attrib.get("head"))
            )  # fmt: skip
    toms.sort(key=lambda row: [int(i) for i in row[2].split()])
    return toms, pages


def make_database(root, text, page_marker_regex=HATHI_MARKERS, section_length=0, parse=True, web_config=""):
    """A database of the book text, with page_marker_regex in its db.locals.py (none, as databases loaded before it
    was written there, when it is None). Its objects and pages are those the plain-text parser makes of text with
    page_marker_regex and section_length, or else (parse=False) a document and a div1 of all of it."""
    data = root / "data"
    (data / "TEXT").mkdir(parents=True)
    (data / "TEXT" / "book.txt").write_bytes(text.encode())
    (data / "db.locals.py").write_text(
        'metadata_fields = ["title", "author", "filename", "head"]\n'
        'metadata_types = {"title": "doc", "author": "doc", "filename": "doc", "head": "div"}\n'
        f"metadata_sql_types = {dict.fromkeys(COLUMNS, 'text')!r}\n"
        f"token_regex = {TOKEN_REGEX!r}\n" + (f"page_marker_regex = {page_marker_regex!r}\n" if page_marker_regex else "")
    )
    (data / "web_config.cfg").write_text('dbname = "plain"\n' + web_config)
    if parse:
        toms, pages = parsed_rows(text, page_marker_regex, section_length)
    else:
        end = len(text.encode())
        toms = [
            ("doc", "text", "1 0 0 0 0 0 0", 0, end, "book.txt", "Book", "Anonymous", None),
            ("div1", "div", "1 1 0 0 0 0 0", 0, end, None, None, None, "Section 1"),
        ]
        pages = []
    dbh = sqlite3.connect(data / "toms.db")
    dbh.execute(f"CREATE TABLE toms ({', '.join(c + ' text' for c in COLUMNS)})")
    dbh.execute("CREATE INDEX toms_philo_id_index ON toms (philo_id)")
    dbh.executemany(f"INSERT INTO toms VALUES ({', '.join('?' * len(COLUMNS))})", [tuple(map(str_or_none, r)) for r in toms])
    dbh.execute("CREATE TABLE pages (philo_type text, philo_id text, n text, facs text, id text, start_byte text, end_byte text)")
    dbh.executemany("INSERT INTO pages VALUES ('page', ?, ?, NULL, NULL, ?, ?)", [tuple(map(str, p)) for p in pages])
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


def navigate(root, window_length=0, **query):
    config = WebConfig(str(root))
    config["navigation_window_length"] = window_length
    request = WSGIHandler({"QUERY_STRING": urllib.parse.urlencode(query, doseq=True), "PHILOLOGIC_DBURL": ""}, config)
    return generate_text_object(request, config)


def byte_of(text, words):
    """The byte offset of words in text"""
    return len(text[: text.index(words)].encode())


def page_breaks(html):
    """The numbers of the page breaks the text page shows, in order"""
    return re.findall(r'<span n="([^"]*)" class="xml-pb-image">--\1--</span>', html)


@pytest.mark.unit
class TestTextPage:
    def test_markers_shown_as_page_breaks_of_xml_texts(self, tmp_path):
        """The very same text page as for the XML text with <pb> tags where the markers are"""
        plain = navigate(make_database(tmp_path / "plain", PLAIN, parse=False), philo_id="1 1")["text"]
        xml = navigate(make_database(tmp_path / "xml", XML, page_marker_regex=None, parse=False), philo_id="1 1")
        assert plain == xml["text"]
        assert page_breaks(plain) == ["247", "248", "249"]
        assert f"\n   {pb_span(249)}  \nthe paragraph goes on" in plain
        for marker in MARKERS.values():
            assert marker not in plain
        for not_a_marker in ("{braces} in it", "{00000250} is no marker", "\n{nor_this}\n"):
            assert not_a_marker in plain

    @pytest.mark.parametrize("section_length", [0, 40])
    def test_the_numbers_of_the_pages(self, tmp_path, section_length):
        """Each text object, a section starting at its marker or the book's single virtual div1, shows the pages of
        the pages table which start in it"""
        root = make_database(tmp_path, PLAIN, section_length=section_length)
        dbh = sqlite3.connect(root / "data" / "toms.db")
        pages = dbh.execute("SELECT CAST(start_byte AS INT), n FROM pages ORDER BY CAST(start_byte AS INT)").fetchall()
        divs = dbh.execute("SELECT philo_id, start_byte, end_byte FROM toms WHERE philo_type = 'div1'").fetchall()
        assert [n for _, n in pages] == ["247", "248", "249"]
        shown = []
        for philo_id, start, end in divs:
            text = navigate(root, philo_id=" ".join(philo_id.split()[:2]))["text"]
            assert page_breaks(text) == [n for byte, n in pages if int(start) <= byte < int(end)]
            assert not any(marker in text for marker in MARKERS.values())
            shown += page_breaks(text)
        # The sections start with the book, then at markers; the virtual div1 at its first paragraph, past a marker
        assert shown == (["247", "248", "249"] if section_length else ["248", "249"])

    def test_a_page_with_no_number(self, tmp_path):
        text = "Some text.\n\n[page]\n\nMore text.\n"
        root = make_database(tmp_path, text, page_marker_regex=r"\[page\]", section_length=10)
        dbh = sqlite3.connect(root / "data" / "toms.db")
        assert dbh.execute("SELECT n FROM pages").fetchall() == [("na",)]
        texts = "".join(navigate(root, philo_id=f"1 {div}")["text"] for div in (1, 2))
        assert page_breaks(texts) == ["na"]
        assert "[page]" not in texts

    def test_highlights_next_to_a_marker(self, tmp_path):
        """The words on the lines before and after a marker"""
        root = make_database(tmp_path, PLAIN, parse=False)
        hits = [byte_of(PLAIN, "page\n   {"), byte_of(PLAIN, "the paragraph")]
        text = navigate(root, philo_id="1 1", byte=[str(hit) for hit in hits])["text"]
        assert (
            f'<span class="highlight">page</span>\n   {pb_span(249)}  \n<span class="highlight">the</span> paragraph'
            in text
        )

    def test_a_passage_over_a_marker(self, tmp_path):
        root = make_database(tmp_path, PLAIN, parse=False)
        start, end = byte_of(PLAIN, "A line ending"), byte_of(PLAIN, " on: {")
        text = navigate(root, philo_id="1 1", start_byte=str(start), end_byte=str(end))["text"]
        assert page_breaks(text) == ["247", "248", "249"]
        assert MARKERS["@249"] not in text
        passage = text[text.index('class="passage-marker"') : text.index('id="end-passage-0"')]
        assert pb_span(249) in passage and "A line ending a page" in passage and "goes" in passage

    def test_the_parts_of_a_windowed_text(self, tmp_path):
        """The parts of the book's virtual div1 start at paragraphs, after marker lines: each marker is a page
        break in the part it ends"""
        text = "".join(f"{{UCAL_C3470619_{page:08d}}}\n\nWords of page {page}.\n\n" for page in range(1, 21))
        root = make_database(tmp_path, text, section_length=0)
        config = WebConfig(str(root))
        bounds = window_bounds(DB(str(root / "data"))["1 1"], config, 100)
        assert len(bounds) > 5
        shown = []
        for start in bounds[:-1]:
            response = navigate(root, 100, philo_id="1 1", windowed="1", window_start=str(start))
            assert response["window"]["start_byte"] == start
            assert "{UCAL" not in response["text"]
            shown += page_breaks(response["text"])
        assert shown == [str(page) for page in range(2, 21)]  # the book's div1 starts after the first marker

    def test_markers_stay_text_without_a_regex(self, tmp_path):
        """In databases loaded without one, and older ones, whose db.locals.py has none"""
        root = make_database(tmp_path, PLAIN, page_marker_regex=None, parse=False)
        text = navigate(root, philo_id="1 1")["text"]
        assert "xml-pb-image" not in text
        for marker in MARKERS.values():
            assert marker in text


def hit_text(text, words):
    """The bytes of text, and the offset of the hit on words in them"""
    return text.encode(), [byte_of(text, words)]


def format_distinctive(text, word_regex, byte_offsets, page_marker_regex=None):
    """The concordance of the distinctive uses, its hits those of the query"""
    return format_concordance_distinctive(text, word_regex, byte_offsets, [], page_marker_regex=page_marker_regex)


@pytest.mark.unit
@pytest.mark.parametrize("formatter", [format_concordance, format_strip, format_distinctive])
class TestConcordanceAndKwic:
    def test_markers_left_out_as_pb_tags(self, formatter):
        """As the <pb> tags of an XML text are"""
        for words in ("ATHENÆUM", "Café", "paragraph"):
            text, hits = hit_text(PLAIN, words)
            plain = formatter(text, TOKEN_REGEX, hits, page_marker_regex=HATHI_MARKERS)
            text, hits = hit_text(XML, words)
            xml = formatter(text, TOKEN_REGEX, hits)
            assert plain == xml
            assert not any(marker in plain for marker in MARKERS.values())
            assert f'<span class="highlight">{words}</span>' in plain
            assert "{braces}" in plain

    def test_a_text_ending_in_the_middle_of_a_marker(self, formatter):
        text, hits = hit_text(PLAIN, "Café")
        cut = text[: text.index(b"{39002007368898_") + len(b"{3900")]
        output = formatter(cut, TOKEN_REGEX, hits, HATHI_MARKERS)
        assert "{3900" not in output
        assert TAGS.sub("", output).rstrip().endswith("A line ending a page")
        assert '<span class="highlight">Café</span>' in output

    def test_a_hit_on_a_last_line_like_a_marker_kept(self, formatter):
        text = "A line.\n{in the hit and more".encode()
        output = formatter(text, TOKEN_REGEX, [text.index(b"hit")], HATHI_MARKERS)
        assert '<span class="highlight">hit</span> and more' in output

    def test_markers_stay_text_without_a_regex(self, formatter):
        text, hits = hit_text(PLAIN, "Café")
        output = formatter(text, TOKEN_REGEX, hits)
        assert output == formatter(text, TOKEN_REGEX, hits, None)
        for marker in MARKERS.values():
            assert marker in output


class Hit(SimpleNamespace):
    """A hit of the book, with its metadata"""

    def __getitem__(self, key):
        return getattr(self, key)


@pytest.mark.unit
class TestReports:
    """The concordance and KWIC of a hit get the page_marker_regex of the database"""

    @pytest.fixture
    def hit(self, tmp_path):
        root = make_database(tmp_path, PLAIN, parse=False)
        db = DB(str(root / "data"))
        hit = Hit(
            bytes=[byte_of(PLAIN, "Café")], doc=SimpleNamespace(filename="book.txt"), philo_id=[1, 1, 1, 1, 1, 1, 1],
            title="Book", author="Anonymous", filename="book.txt", head="",
        )  # fmt: skip
        return root, db, hit

    def test_concordance(self, hit):
        root, db, hit = hit
        context = get_concordance_text(db, hit, str(root), 300)
        assert '<span class="highlight">Café</span>' in context
        assert "{UCAL" not in context and "{00000248}" not in context and "{3900" not in context

    def test_kwic(self, hit, monkeypatch):
        root, db, hit = hit
        monkeypatch.setattr(kwic_module, "citation_links", lambda *args, **kwargs: {})
        monkeypatch.setattr(kwic_module, "citations", lambda *args, **kwargs: [])
        context = kwic_hit_object(hit, WebConfig(str(root)), db)["context"]
        assert '<span class="kwic-highlight"><span class="highlight">Café</span></span>' in context
        assert "{UCAL" not in context and "{00000248}" not in context and "{3900" not in context
