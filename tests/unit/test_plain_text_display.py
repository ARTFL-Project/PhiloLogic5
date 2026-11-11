"""Unit tests for the "&", "<" and ">" of plain texts at display time. In a database of file_type "plain_text" they are
text, which the text page, concordances and KWIC show as they are: the formatters took a stray "<" of OCR for the start
of a tag, and the text after it, up to the next ">", for its attributes. Each test of a report makes a small database
of a plain-text book."""

import io
import json
import sqlite3
import sys
import urllib.parse
from html.parser import HTMLParser
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
from philologic.runtime.ObjectFormatter import format_concordance, format_strip
from philologic.runtime.reports import kwic as kwic_module
from philologic.runtime.reports.kwic import kwic_hit_object
from philologic.runtime.reports.navigation import generate_text_object
from philologic.utils import page_markers

HATHI_MARKERS = r"^\{(?:[^{}]*_)?(\d+)\}$"
MARKUP_MARKERS = r"^(?:<<|\[p&)(\d+)(?:>>|\])$"  # page markers with "<", ">" and "&" in them
TOKEN_REGEX = r"[\p{L}\p{M}\p{N}]+|[&\p{L};]+"
COLUMNS = ["philo_type", "philo_name", "philo_id", "start_byte", "end_byte", "filename", "title", "author", "head"]

# OCR'd text with the stray markup of scold_bridle's books: "[<Mtj." in The Athenaeum, "<?cal'" and the "<abducere: "
# of The Century Dictionary's etymologies, ">" alone, "&" alone, "&c.", an "&lt;" and an "&eacute;" which are text
BOOK = (
    "{00000001}\n"
    "\n"
    "THE ATHENAEUM, [<Mtj. 14, of the year, and the first hit of the page.\n"
    "A line with <?cal' in it, and words after it.\n"
    "\n"
    "abduction (ab-duk'shon), n. [< L. abductio, <abducere: see abduce.] 1. The act\n"
    "of leading away, & so on, &c.; where a > b and the hit is &lt;not&gt; a tag, nor &eacute; an e.\n"
    "<\n"
    "{00000002}\n"
    ">\n"
    "Last <words of the book, <philoHighlight/> and <pb n=\"3\"/> as they are.\n"
)
SPECIALS = ("[<Mtj. 14", "<?cal'", "[< L.", "<abducere: ", "& so on, &c.;", "a > b", "&lt;not&gt;", "&eacute;")


def escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Html(HTMLParser):
    """What a browser reads in some HTML: its tags, comments, declarations and processing instructions, and its text"""

    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.tags, self.text = [], []
        self.feed(html)
        self.close()
        self.text = "".join(self.text)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_startendtag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)

    def handle_comment(self, data):
        self.tags.append(("!--", {}))

    def handle_decl(self, decl):
        self.tags.append(("!", {}))

    def handle_pi(self, data):
        self.tags.append(("?", {}))

    def unknown_decl(self, data):
        self.tags.append(("![", {}))


def philologic_tags_only(html):
    """Whether the HTML has none but the tags of the formatters: spans of their classes, those of page breaks, and the
    divs around the text page"""
    for tag, attributes in Html(html).tags:
        if tag == "div" and attributes in ({}, {"class": "philologic-fragment"}):  # around the text page
            continue
        if tag != "span":
            return False
        if set(attributes) == {"n"} or set(attributes) == {"n", "class"} and attributes["class"] == "xml-pb-image":
            continue  # a page break, in a concordance or KWIC, or the text page
        if set(attributes) == {"class", "n"} and attributes["class"].startswith("passage-"):
            continue
        if set(attributes) == {"class"} and attributes["class"] in (
            "highlight", "philologic-fragment", "kwic-text", "kwic-before", "inner-before", "kwic-highlight",
            "kwic-after",
        ):  # fmt: skip
            continue
        if set(attributes) == {"class", "data-offsets", "n"} and attributes["class"] == "passage-marker":
            continue
        if set(attributes) == {"id", "n"} and attributes["id"].startswith("end-passage-"):
            continue
        return False
    return True


def byte_of(text, words):
    """The byte offset of words in text"""
    return len(text[: text.index(words)].encode())


def parsed_rows(text, section_length=0, page_marker_regex=HATHI_MARKERS):
    """The toms and pages rows of the objects and pages the plain-text parser makes of text"""
    output = io.StringIO()
    parser = PlainTextParser(
        output, 1, len(text.encode()), known_metadata={"filename": "book.txt", "title": "Book", "author": "Anonymous"},
        page_marker_regex=page_marker_regex, plain_text_section_length=section_length,
    )  # fmt: skip
    parser.parse(io.StringIO(text, newline=""))
    toms, pages, words = [], [], []
    for line in output.getvalue().splitlines():
        philo_type, name, philo_id, attrib = line.split("\t")
        attrib = json.loads(attrib)
        if philo_type == "page":
            pages.append((philo_id, attrib["n"], attrib["start_byte"], attrib["end_byte"]))
        elif philo_type == "word":
            words.append((name, attrib["start_byte"]))
        elif philo_type in ("doc", "div1", "div2", "div3", "para"):
            philo_id = " ".join(philo_id.split()[:7])
            toms.append(
                (philo_type, name, philo_id, attrib["start_byte"], attrib["end_byte"], attrib.get("filename"),
                 attrib.get("title"), attrib.get("author"), attrib.get("head"))
            )  # fmt: skip
    toms.sort(key=lambda row: [int(i) for i in row[2].split()])
    return toms, pages, words


def make_database(root, text, file_type="plain_text", section_length=0, page_marker_regex=HATHI_MARKERS):
    """A database of the book text, of file_type in its db.locals.py (none, as databases loaded before it was written
    there, when it is None), with the objects and pages the plain-text parser makes of it with page_marker_regex"""
    data = root / "data"
    (data / "TEXT").mkdir(parents=True)
    (data / "TEXT" / "book.txt").write_bytes(text.encode())
    (data / "db.locals.py").write_text(
        'metadata_fields = ["title", "author", "filename", "head"]\n'
        'metadata_types = {"title": "doc", "author": "doc", "filename": "doc", "head": "div"}\n'
        f"metadata_sql_types = {dict.fromkeys(COLUMNS, 'text')!r}\n"
        f"token_regex = {TOKEN_REGEX!r}\n"
        f"page_marker_regex = {page_marker_regex!r}\n" + (f"file_type = {file_type!r}\n" if file_type else "")
    )
    (data / "web_config.cfg").write_text('dbname = "plain"\n')
    toms, pages, _ = parsed_rows(text, section_length, page_marker_regex)
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


def shown(text):
    """The text of the book as the text page shows it: its page markers as the numbers of page breaks"""
    return text.replace("{00000001}", "1").replace("{00000002}", "2")


@pytest.mark.unit
class TestEscapes:
    """philologic.utils.page_markers.escape_plain_text, and the page markers of escaped texts"""

    @pytest.mark.parametrize("text", [BOOK, "<", "&&<>>", "no markup here", "", "é<ß>&", "&lt;&amp;lt;"])
    def test_escaped_with_the_position_of_each_byte(self, text):
        raw = text.encode()
        escaped, position = page_markers.escape_plain_text(raw)
        assert escaped == escape(text).encode()
        for offset in range(len(raw) + 1):  # the bytes before it, escaped, come before its position
            assert escaped[: position(offset)] == raw[:offset].replace(b"&", b"&amp;").replace(b"<", b"&lt;").replace(
                b">", b"&gt;"
            )
        assert position(len(raw)) == len(escaped)
        assert page_markers.unescape_plain_text(escaped.decode()) == text

    def test_a_text_without_markup_characters_as_it_is(self):
        raw = "Café ſont here".encode()
        escaped, position = page_markers.escape_plain_text(raw)
        assert escaped is raw
        assert [position(offset) for offset in range(len(raw) + 1)] == list(range(len(raw) + 1))

    def test_markers_of_markup_characters_matched_as_they_are_in_the_file(self):
        """The regex sees the line of the file: "<<12>>" and "[p&13]" are markers, though escaped"""
        regex = MARKUP_MARKERS
        text = "Words <here>.\n<<12>>\nMore & more.\n [p&13] \nEnd <<14>> not a marker\n"
        escaped = escape(text)
        converted = page_markers.page_markers_to_pb(escaped, regex, escaped=True)
        assert converted == escaped.replace("&lt;&lt;12&gt;&gt;", '<pb n="12"/>').replace("[p&amp;13]", '<pb n="13"/>')
        raw = page_markers.page_markers_to_pb(text, regex)
        assert raw == text.replace("<<12>>", '<pb n="12"/>').replace("[p&13]", '<pb n="13"/>')
        assert page_markers.page_markers_to_pb(escaped, regex) == escaped  # not seen as the file has them

    def test_a_text_ending_in_the_middle_of_an_escaped_marker(self):
        regex = r"^<<(\d+)>>$"
        text = escape("Words.\n<<1").encode()
        assert page_markers.cut_page_marker_start(text, regex, escaped=True) == len(b"Words.\n")
        whole = escape("Words.\n<<12>>").encode()
        assert page_markers.cut_page_marker_start(whole, regex, escaped=True) is None


def hit_text(text, words):
    """The bytes of text, and the offset of the hit on words in them"""
    return text.encode(), [byte_of(text, words)]


@pytest.mark.unit
@pytest.mark.parametrize("formatter", [format_concordance, format_strip])
class TestConcordanceAndKwic:
    @pytest.mark.parametrize("words", ["first hit", "Mtj", "cal'", "abducere", "so on", "hit is", "not&gt;", "words"])
    def test_shown_as_the_xml_text_of_the_same_characters(self, formatter, words):
        """The very same as the XML text which has the same characters, the hits where their words are"""
        text, hits = hit_text(BOOK, words)
        plain = formatter(text, TOKEN_REGEX, hits, HATHI_MARKERS, file_type="plain_text")
        xml_text, xml_hits = hit_text(escape(BOOK), escape(words))
        assert plain == formatter(xml_text, TOKEN_REGEX, xml_hits, HATHI_MARKERS)
        assert philologic_tags_only(plain)
        highlight = plain.split('<span class="highlight">', 1)[1].split("</span>", 1)[0]
        assert highlight == escape(re.match(TOKEN_REGEX, words).group())

    def test_the_text_after_a_stray_less_than_sign_and_its_hit(self, formatter):
        """The hit after "[<Mtj." is highlighted, and all the text after it shown"""
        text, hits = hit_text(BOOK, "first hit")
        output = formatter(text, TOKEN_REGEX, hits, HATHI_MARKERS, file_type="plain_text")
        assert '[&lt;Mtj. 14, of the year, and the <span class="highlight">first</span> hit' in output
        assert "A line with &lt;?cal'" in output
        assert philologic_tags_only(output)
        visible = Html(output).text
        assert "[<Mtj. 14, of the year, and the first hit of the page.\nA line with <?cal'" in visible

    def test_the_whole_context(self, formatter):
        """Every character of the context is shown, as it is but for the page markers, which are left out, the partial
        words at its ends and the spaces KWIC drops around apostrophes"""
        start, end = byte_of(BOOK, "HENAEUM"), byte_of(BOOK, "e book,")
        text = BOOK.encode()[start:end]
        hit = byte_of(BOOK, "so on") - start
        output = formatter(text, TOKEN_REGEX, [hit], HATHI_MARKERS, file_type="plain_text")
        expected = BOOK[BOOK.index(" [<Mtj.") : BOOK.index("e book,")].replace("{00000002}", "")
        visible = Html(output).text
        if formatter is format_strip:
            expected = expected.replace("' ", "'").replace(" '", "'")
        assert visible == expected
        assert philologic_tags_only(output)

    def test_an_escaped_entity_as_it_is(self, formatter):
        """"&lt;" and "&eacute;" in a plain text are text: shown as such, not as "<" and "é", by convert_entities"""
        text, hits = hit_text(BOOK, "hit is")
        output = formatter(text, TOKEN_REGEX, hits, HATHI_MARKERS, file_type="plain_text")
        assert "is &amp;lt;not&amp;gt; a tag, nor &amp;eacute; an e." in output
        assert "is &lt;not&gt; a tag, nor &eacute; an e." in Html(output).text

    def test_an_ampersand_alone(self, formatter):
        text, hits = hit_text(BOOK, "so on")
        output = formatter(text, TOKEN_REGEX, hits, HATHI_MARKERS, file_type="plain_text")
        assert 'away, &amp; <span class="highlight">so</span> on, &amp;c.;' in output
        assert "& so on, &c.;" in Html(output).text

    def test_a_hit_on_an_ampersand_word(self, formatter):
        """"&c" is a word of the token regex: highlighted whole"""
        text, hits = hit_text(BOOK, "&c.")
        output = formatter(text, TOKEN_REGEX, hits, HATHI_MARKERS, file_type="plain_text")
        assert '<span class="highlight">&amp;c</span>.;' in output

    def test_less_than_signs_next_to_a_highlight(self, formatter):
        text = "Before the <hit> and <after, <<x>>.".encode()
        output = formatter(text, TOKEN_REGEX, [text.index(b"hit"), text.index(b"after")], file_type="plain_text")
        assert (
            '&lt;<span class="highlight">hit</span>&gt; and &lt;<span class="highlight">after</span>, &lt;&lt;x&gt;&gt;.'
            in output
        )

    def test_less_than_signs_next_to_a_page_marker(self, formatter):
        """The marker between the lines "<" and ">" is a page break, left out, and they are shown"""
        text, hits = hit_text(BOOK, "nor &eacute;")
        output = formatter(text, TOKEN_REGEX, hits, HATHI_MARKERS, file_type="plain_text")
        assert "{00000002}" not in output
        assert "an e.\n&lt;\n" in output and "\n&gt;\nLast &lt;words" in output
        assert "an e.\n<\n" in Html(output).text

    def test_markers_of_markup_characters(self, formatter):
        """Matched as the file has them, and left out, the start of the last one too"""
        text = "Words <here>.\n<<12>>\nMore & more, the hit.\n [p&13] \nEnd <<14>> no marker.\n<<1".encode()
        output = formatter(text, TOKEN_REGEX, [text.index(b"hit")], MARKUP_MARKERS, file_type="plain_text")
        # after the partial word the context starts with
        assert Html(output).text == " <here>.\n\nMore & more, the hit.\n  \nEnd <<14>> no marker.\n"

    def test_tags_in_a_plain_text_are_text(self, formatter):
        """A plain text that has the tags of the formatters has them as text: no highlight, no page break"""
        text, hits = hit_text(BOOK, "Last")
        output = formatter(text, TOKEN_REGEX, hits, HATHI_MARKERS, file_type="plain_text")
        assert '&lt;philoHighlight/&gt; and &lt;pb n="3"/&gt; as they are.' in output
        assert output.count('class="highlight"') == 1

    def test_a_context_starting_at_a_greater_than_sign(self, formatter):
        """No partial tag is cut at its start: the ">" before the hit is text. Its partial word is"""
        text = "ord > and the text before the hit".encode()
        output = formatter(text, TOKEN_REGEX, [text.index(b"hit")], file_type="plain_text")
        assert Html(output).text == " > and the text before the hit"
        xml = formatter(text, TOKEN_REGEX, [text.index(b"hit")])
        assert Html(xml).text == " and the text before the hit"  # an XML text's partial tag

    def test_a_partial_word_with_a_less_than_sign(self, formatter):
        """Cut as the partial word of the XML text, whose "&lt;" is no tag either"""
        text = "rd<Mtj. and the hit".encode()
        output = formatter(text, TOKEN_REGEX, [text.index(b"hit")], file_type="plain_text")
        assert Html(output).text == " and the hit"

    def test_a_context_ending_at_a_less_than_sign(self, formatter):
        """No partial tag is cut at its end: the "<" after the hit and the text after it are shown"""
        text = "the hit and the text after, <abducere: see abd".encode()
        output = formatter(text, TOKEN_REGEX, [text.index(b"hit")], file_type="plain_text")
        assert Html(output).text.endswith("text after, <abducere: see abd")
        xml = formatter(text, TOKEN_REGEX, [text.index(b"hit")])
        assert Html(xml).text.endswith("text after, ")  # an XML text's partial tag

    def test_xml_without_a_file_type(self, formatter):
        """As before for XML texts, and databases loaded before file_type was written to db.locals.py"""
        text, hits = hit_text(BOOK, "first hit")
        assert formatter(text, TOKEN_REGEX, hits, HATHI_MARKERS) == formatter(
            text, TOKEN_REGEX, hits, HATHI_MARKERS, file_type="xml"
        )
        assert "[&lt;Mtj." not in formatter(text, TOKEN_REGEX, hits, HATHI_MARKERS)


@pytest.mark.unit
class TestTextPage:
    def test_the_whole_text(self, tmp_path):
        """All of it, the stray markup of OCR included, as text: the page breaks of its markers, the highlights of the
        hits in what was taken for tags, no other tag"""
        root = make_database(tmp_path, BOOK)
        hits = [byte_of(BOOK, "first hit"), byte_of(BOOK, "words after"), byte_of(BOOK, "abducere"),
                byte_of(BOOK, "words of")]  # fmt: skip
        html = navigate(root, philo_id="1 1", byte=[str(hit) for hit in hits])["text"]
        assert philologic_tags_only(html)
        assert Html(html).text == shown(BOOK)[shown(BOOK).index("THE") :]  # the div1 starts with the first paragraph
        for special in SPECIALS:  # escaped, in the HTML the browser reads
            assert escape(special) in re.sub(r"<[^>]+>", "", html)
        assert html.count('class="highlight"') == 4
        for word in ("first", "words", "abducere"):
            assert f'<span class="highlight">{word}</span>' in html
        assert '<span n="2" class="xml-pb-image">2</span>' in html

    def test_a_passage_over_stray_markup(self, tmp_path):
        """The spans of a passage keep its text escaped: parsed again, its "&lt;" were "<", which the convert_entities
        of the output made markup of"""
        root = make_database(tmp_path, BOOK)
        start, end = byte_of(BOOK, "abduction"), byte_of(BOOK, "Last")
        text = navigate(root, philo_id="1 1", start_byte=str(start), end_byte=str(end))["text"]
        assert philologic_tags_only(text)
        assert Html(text).text == shown(BOOK)[shown(BOOK).index("THE") :]  # the div1 starts with the first paragraph
        passage = text[text.index('class="passage-marker"') : text.index('id="end-passage-0"')]
        assert f'data-offsets="{start}-{end}"' in text
        for special in ("&lt;abducere: ", "&amp; so on, &amp;c.;", "a &gt; b", "&amp;lt;not&amp;gt;"):
            assert special in passage

    @pytest.mark.parametrize("last", ["Last", "an e."])
    def test_a_passage_after_highlights(self, tmp_path, last):
        """The highlights of the hits before the bounds of a passage don't move them, in the escaped text either: the
        passage spans have the text of its bytes, its stray markup included. The last hit, "nor", is fewer bytes before
        the end of the passage at "an e." than the escapes before it add."""
        root = make_database(tmp_path, BOOK)
        start, end = byte_of(BOOK, "abduction"), byte_of(BOOK, last)
        inside = ("abducere", "so", "nor")
        hits = [byte_of(BOOK, words) for words in ("first hit", "words after") + inside]
        query = {"philo_id": "1 1", "byte": [str(hit) for hit in hits]}
        text = navigate(root, start_byte=str(start), end_byte=str(end), **query)["text"]
        assert philologic_tags_only(text)
        assert f'data-offsets="{start}-{end}"' in text
        in_passage = "".join(re.findall(r'<span class="passage-0" n="0">([^<]*)</span>', text))
        assert Html(in_passage).text == BOOK.encode()[start:end].decode().replace("{00000002}", "")  # but its page
        for word in ("first", "words"):
            assert f'<span class="highlight">{word}</span>' in text
        for word in inside:
            assert f'<span class="highlight"><span class="passage-0" n="0">{word}</span></span>' in text
        assert Html(text).text == Html(navigate(root, **query)["text"]).text

    def test_the_parts_of_a_windowed_text(self, tmp_path):
        """A "<" opening no tag doesn't move the ends of the parts of a plain text: they end at the same line breaks as
        without it, and show all of the text"""
        lines = "".join(f"Line {n} of a long paragraph with no blank line, and words.\n" for n in range(200))
        text = "{00000001}\n\n" + lines.replace("Line 3 ", "Line [<Mtj. 3 ")
        root = make_database(tmp_path / "plain", text)
        same = make_database(tmp_path / "same", text.replace("[<Mtj.", "[(Mtj."))
        xml = make_database(tmp_path / "xml", text, file_type="xml")
        bounds = {name: window_bounds(DB(str(db / "data"))["1 1"], WebConfig(str(db)), 1000) for name, db in
                  (("plain", root), ("same", same), ("xml", xml))}  # fmt: skip
        assert bounds["plain"] == bounds["same"]
        assert len(bounds["plain"]) > 10
        assert bounds["xml"] != bounds["plain"]  # which take it for a tag up to the end of the text
        parts = []
        for start in bounds["plain"][:-1]:
            response = navigate(root, 1000, philo_id="1 1", windowed="1", window_start=str(start))
            assert response["window"]["start_byte"] == start
            assert philologic_tags_only(response["text"])
            parts.append(Html(response["text"]).text)
        assert "".join(parts) == shown(text)[len("1\n\n") :]

    def test_markers_of_markup_characters(self, tmp_path):
        """Matched as the file has them: the parser made pages of them"""
        text = "Words <here>.\n\n<<12>>\n\nMore & more.\n\n[p&13]\n\nEnd <<14>> no marker.\n"
        root = make_database(tmp_path, text, page_marker_regex=MARKUP_MARKERS)
        html = navigate(root, philo_id="1 1")["text"]
        assert re.findall(r'<span n="(\d+)" class="xml-pb-image">\1</span>', html) == ["12", "13"]
        assert Html(html).text == text.replace("<<12>>", "12").replace("[p&13]", "13")

    def test_xml_without_a_file_type(self, tmp_path):
        """The text page of databases loaded before file_type was written to db.locals.py is that of XML texts"""
        old = navigate(make_database(tmp_path / "old", BOOK, file_type=None), philo_id="1 1")["text"]
        assert old == navigate(make_database(tmp_path / "xml", BOOK, file_type="xml"), philo_id="1 1")["text"]
        assert "[&lt;Mtj." not in old


class Hit(SimpleNamespace):
    """A hit of the book, with its metadata"""

    def __getitem__(self, key):
        return getattr(self, key)


@pytest.mark.unit
class TestReports:
    """The concordance, more context and KWIC of a hit get the file_type of the database"""

    @pytest.fixture
    def hit(self, tmp_path):
        root = make_database(tmp_path, BOOK)
        db = DB(str(root / "data"))
        hit = Hit(
            bytes=[byte_of(BOOK, "words after")], doc=SimpleNamespace(filename="book.txt"),
            philo_id=[1, 1, 1, 1, 1, 1, 1], title="Book", author="Anonymous", filename="book.txt", head="",
        )  # fmt: skip
        return root, db, hit

    @pytest.mark.parametrize("context_size", [300, 900])  # a concordance, and more context
    def test_concordance(self, hit, context_size):
        root, db, hit = hit
        context = get_concordance_text(db, hit, str(root), context_size)
        assert 'A line with &lt;?cal\' in it, and <span class="highlight">words</span> after it.' in context
        assert "[&lt;Mtj. 14" in context
        assert philologic_tags_only(context)

    def test_kwic(self, hit, monkeypatch):
        root, db, hit = hit
        monkeypatch.setattr(kwic_module, "citation_links", lambda *args, **kwargs: {})
        monkeypatch.setattr(kwic_module, "citations", lambda *args, **kwargs: [])
        result = kwic_hit_object(hit, WebConfig(str(root)), db)
        context = result["context"]
        assert "A line with &lt;?cal'in it, and </span></span>&nbsp;" in context
        assert '<span class="kwic-highlight"><span class="highlight">words</span></span>&nbsp;' in context
        assert '<span class="kwic-after"> after it.' in context
        assert result["highlighted_text"] == "words"
        assert philologic_tags_only(context.replace("&nbsp;", " "))
