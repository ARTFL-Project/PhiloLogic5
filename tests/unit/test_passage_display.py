"""Unit tests for the passages of the text page (start_byte and end_byte, as TextPAIR links have them), which
format_text_object shows in spans: the text in them is the text of the file, as around them. Its "&lt;", "&gt;" and
"&amp;" were decoded once more when the spans were parsed again, and the browser took the "<b>" and the
"<img src=x onerror=...>" they made for markup. Each test makes a small database of an XML text."""

import sqlite3
import sys
import urllib.parse
from html.parser import HTMLParser
from pathlib import Path

import pytest
import regex as re
from lxml import etree

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime import WebConfig, WSGIHandler
from philologic.runtime import get_text as get_text_module
from philologic.runtime.ObjectFormatter import passage_span
from philologic.runtime.reports.navigation import generate_text_object
from philologic.utils import convert_entities

COLUMNS = ["philo_type", "philo_name", "philo_id", "start_byte", "end_byte", "filename", "title", "author"]
TOKEN_REGEX = r"[\p{L}\p{M}\p{N}]+|[&\p{L};]+"
# Markup escaped in the text of an element and in a tail, an "&amp;" which is a "&" and one before "lt;", which is
# text: "&lt;" shown as such
ESCAPED = "&lt;b&gt;bold&lt;/b&gt; &amp; &lt;img src=x onerror=alert(1)&gt; and &amp;lt;i&amp;gt;"
SHOWN = "<b>bold</b> & <img src=x onerror=alert(1)> and &lt;i&gt;"


def book(entity=""):
    """A TEI text whose second paragraph has ESCAPED in the text of an element and in a tail, and entity (an
    "&eacute;", which the HTML parser decoded where the XML parser could not)"""
    return (
        "<TEI><text><body><div1 type=\"chapter\">\n"
        "<p>Before the passage.</p>\n"
        f"<p>Words <hi>{ESCAPED.replace('bold', 'first')} in a text{entity}</hi> and {ESCAPED.replace('bold', 'second')}"
        f" in a tail{entity}, with words.</p>\n"
        "<p>After the passage, a few more words to end it.</p>\n"
        "</div1></body></text></TEI>\n"
    )


def byte_of(text, words):
    return len(text[: text.index(words)].encode())


def make_database(root, text):
    data = root / "data"
    (data / "TEXT").mkdir(parents=True)
    (data / "TEXT" / "book.xml").write_bytes(text.encode())
    (data / "db.locals.py").write_text(
        'metadata_fields = ["title", "author", "filename"]\n'
        'metadata_types = {"title": "doc", "author": "doc", "filename": "doc"}\n'
        f"metadata_sql_types = {dict.fromkeys(COLUMNS, 'text')!r}\n"
        f"token_regex = {TOKEN_REGEX!r}\n"
        'file_type = "xml"\n'
    )
    (data / "web_config.cfg").write_text('dbname = "xml"\n')
    div_start, div_end = byte_of(text, "<div1"), byte_of(text, "</body>")
    paragraphs = [m.start() for m in re.finditer("<p>", text)]
    rows = [
        ("doc", "TEI", "1 0 0 0 0 0 0", 0, len(text.encode()), "book.xml", "Book", "Anonymous"),
        ("div1", "div1", "1 1 0 0 0 0 0", div_start, div_end, None, None, None),
        ("div2", "__philo_virtual", "1 1 1 0 0 0 0", div_start, div_end, None, None, None),
        ("div3", "__philo_virtual", "1 1 1 1 0 0 0", div_start, div_end, None, None, None),
    ]
    for number, start in enumerate(paragraphs, 1):
        end = text.index("</p>", start) + len("</p>")
        rows.append(("para", "p", f"1 1 1 1 {number} 0 0", len(text[:start].encode()), len(text[:end].encode()), None,
                     None, None))  # fmt: skip
    dbh = sqlite3.connect(data / "toms.db")
    dbh.execute(f"CREATE TABLE toms ({', '.join(c + ' text' for c in COLUMNS)})")
    dbh.execute("CREATE INDEX toms_philo_id_index ON toms (philo_id)")
    dbh.executemany(f"INSERT INTO toms VALUES ({', '.join('?' * len(COLUMNS))})", [tuple(None if v is None else str(v) for v in r) for r in rows])
    dbh.execute("CREATE TABLE pages (philo_type text, philo_id text, n text, facs text, id text, start_byte text, end_byte text)")
    dbh.commit()
    dbh.close()
    return root


@pytest.fixture(autouse=True)
def empty_caches():
    get_text_module._window_cache.clear()
    get_text_module._mmap_cache.clear()
    yield
    get_text_module._window_cache.clear()
    get_text_module._mmap_cache.clear()


def navigate(root, **query):
    config = WebConfig(str(root))
    request = WSGIHandler({"QUERY_STRING": urllib.parse.urlencode(query, doseq=True), "PHILOLOGIC_DBURL": ""}, config)
    return generate_text_object(request, config)["text"]


class Html(HTMLParser):
    """What a browser reads in some HTML: its tags, with their attributes, and its text"""

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


def passage_spans(html):
    """The text of the spans of passage 0"""
    return re.findall(r'<span class="passage-0" n="0">([^<]*)</span>', html)


def passage_text(html):
    """The text of the spans of passage 0, as the browser shows it"""
    return Html("".join(passage_spans(html))).text


@pytest.mark.unit
class TestPassageWithHighlights:
    """The highlights of hits (byte) and a passage (start_byte and end_byte) on the same page. Each highlight was
    inserted in the text before the passage was cut out of it at its byte offsets, and moved the bounds of the passage
    after it 17 bytes (a <philoHighlight/>) early: into a word, or a tag, whose text was then shown in the passage."""

    @pytest.mark.parametrize("before", [0, 1, 3])
    def test_the_passage_where_its_bytes_are(self, tmp_path, before):
        """With none, one or three hits before the passage, and one in it"""
        text = book()
        root = make_database(tmp_path, text)
        start, end = byte_of(text, "in a text"), byte_of(text, "with words")
        hits = [byte_of(text, word) for word in ("Before", "Words", "first")[3 - before :]] + [byte_of(text, "second")]
        query = {"philo_id": "1 1", "byte": [str(hit) for hit in hits]}
        page = navigate(root, start_byte=str(start), end_byte=str(end), **query)
        assert passage_text(page) == Html(text[start:end]).text
        assert f'data-offsets="{start}-{end}"' in page  # the byte offsets of the file
        for word in ("Before", "Words", "first")[3 - before :]:
            assert f'<span class="highlight">{word}</span>' in page
        assert HIGHLIGHT.format("second") in page
        assert Html(page).text == Html(navigate(root, **query)).text

    def test_highlights_at_the_bounds(self, tmp_path):
        """A hit at the start of the passage is in it, one at its end after it"""
        text = book()
        root = make_database(tmp_path, text)
        start, end = byte_of(text, "Words"), byte_of(text, "with words")
        hits = [byte_of(text, "Before"), start, byte_of(text, "second"), end]
        page = navigate(root, philo_id="1 1", start_byte=str(start), end_byte=str(end), byte=[str(hit) for hit in hits])
        assert passage_text(page) == Html(text[start:end]).text
        assert HIGHLIGHT.format("Words") in page and HIGHLIGHT.format("second") in page
        assert '<span n="0" id="end-passage-0"></span><span class="highlight">with</span>' in page


HIGHLIGHT = '<span class="highlight"><span class="passage-0" n="0">{}</span></span>'
# The escapes after the highlight, in the text of the <hi> and in its tail
AFTER = '<span class="passage-0" n="0">&lt;/b&gt; &amp; &lt;img src=x onerror=alert(1)&gt; and &amp;lt;i&amp;gt; in a '


@pytest.mark.unit
class TestPassageText:
    @pytest.fixture(
        params=[("", "Words"), ("&eacute;", "Words"), ("", "&lt;b&gt;second")],
        ids=["xml", "with_an_html_entity", "from_the_escapes"],
    )
    def page(self, request, tmp_path):
        """The text page of the book without passage, and with a passage from "Words" (or from the escapes of the tail)
        to the last paragraph, and highlights next to the escapes: on "first" and "second" (or "second", the
        highlights before a passage moving it)"""
        entity, start_at = request.param
        text = book(entity)
        root = make_database(tmp_path, text)
        start, end = byte_of(text, start_at), byte_of(text, "end it")
        hits = [byte_of(text, word) for word in ("first", "second") if byte_of(text, word) > start]
        plain = navigate(root, philo_id="1 1")
        passage = navigate(root, philo_id="1 1", start_byte=str(start), end_byte=str(end), byte=[str(b) for b in hits])
        return start_at, plain, passage

    def test_no_live_tag(self, page):
        """The browser finds none of the escaped tags in the HTML: only those of the text page"""
        _, plain, passage = page
        tags = Html(passage).tags
        assert {tag for tag, _ in tags} == {"div", "p", "span"}
        assert not any("onerror" in attributes or "src" in attributes for _, attributes in tags)
        assert not re.search(r"<(b|img|i)\b", passage)

    def test_the_text_shown_as_without_the_passage(self, page):
        """Every character of the text, "<", ">" and "&" included, as the page without the passage shows it"""
        _, plain, passage = page
        assert Html(passage).text == Html(plain).text
        for word in ("first", "second"):
            assert SHOWN.replace("bold", word) in Html(passage).text

    def test_escaped_in_the_passage_spans(self, page):
        """In the text of an element and in a tail, next to the highlights, escaped as in the file"""
        start_at, _, passage = page
        span = '<span class="passage-0" n="0">'  # the tails' are in spans of their own

        def around(word, after):
            return re.escape("&lt;b&gt;") + "(?:</span>)+" + re.escape(HIGHLIGHT.format(word) + AFTER + after)

        assert re.search(around("second", "tail"), passage)
        if start_at == "Words":
            assert re.search(around("first", "text"), passage)
            assert re.search(f"(?:{re.escape(span)})+ and &lt;b&gt;</span>", passage)
        else:  # the text right after the start of the passage
            assert re.search(r'<span class="passage-marker"[^>]*></span>' + re.escape(span + "&lt;b&gt;</span>"), passage)

    def test_the_rest_of_the_text_as_without_the_passage(self, page):
        """Outside the passage spans, the HTML is that of the page without the passage"""
        _, plain, passage = page
        unwrapped = None
        while unwrapped != passage:  # the spans of the tails are in spans of their own; those after the end, no class
            unwrapped, passage = passage, re.sub(r'<span(?: class="passage-0" n="0")?>([^<]*)</span>', r"\1", passage)
        unwrapped = re.sub(r'<span class="passage-marker"[^>]*></span>|<span n="0" id="end-passage-0"></span>', "", unwrapped)
        unwrapped = re.sub(r'<span class="highlight">(first|second)</span>', r"\1", unwrapped)
        assert unwrapped == plain


@pytest.mark.unit
class TestPassageSpan:
    @pytest.mark.parametrize(
        "text", ["&lt;b&gt;bold&lt;/b&gt;", "a &amp; b", "caf&eacute; &lt;i&gt;", "&#233; &amp;lt;", ""]
    )
    def test_its_text_as_in_the_tree(self, text):
        """Serialized as the text page is, the text is that of the file, escaped as it is; an empty one has an end tag
        too (HTML reads <span/> as an open span)"""
        span = passage_span("passage-0", "0", text)
        assert (span.tag, dict(span.attrib), span.text) == ("span", {"class": "passage-0", "n": "0"}, text)
        assert convert_entities(etree.tostring(span, encoding="unicode")) == f'<span class="passage-0" n="0">{text}</span>'


# The elements HTML has no end tag for: any other tag written <tag/> is an open one to the browser
VOID_ELEMENTS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


class Browser(HTMLParser):
    """The text of some HTML as a browser reads it, and the part of it in the spans of the passages: a <span .../> is
    an open span to it, which ends with its parent"""

    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.open, self.text, self.in_passage = [], [], []
        self.feed(html)
        self.close()
        self.text, self.in_passage = "".join(self.text), "".join(self.in_passage)

    def handle_starttag(self, tag, attrs):
        if tag not in VOID_ELEMENTS:
            self.open.append((tag, dict(attrs).get("class") or ""))

    handle_startendtag = handle_starttag  # the "/" ignored

    def handle_endtag(self, tag):
        for index in range(len(self.open) - 1, -1, -1):
            if self.open[index][0] == tag:
                del self.open[index:]
                break

    def handle_data(self, data):
        self.text.append(data)
        if any(re.fullmatch(r"passage-\d+", class_name) for _, class_name in self.open):
            self.in_passage.append(data)


@pytest.mark.unit
class TestPassageStartingAtATag:
    """A passage starting right before a tag, or the highlight of a hit there, starts with an empty passage span (the
    text of the start of the passage, which is the tag's): it has an end tag. HTML has no self-closing span, and the
    browser read the <span class="passage-0" n="0"/> lxml writes of an element with no text as an open span, which
    coloured all the rest of the paragraph, past the end of the passage."""

    @staticmethod
    def between_its_markers(page):
        """The text of the page from the passage's marker to its end, as the browser shows it"""
        start = page.index("</span>", page.index('class="passage-marker"')) + len("</span>")
        return Browser(page[start : page.index('<span n="0" id="end-passage-0"></span>')]).text

    @pytest.mark.parametrize("hit", [False, True], ids=["tag", "highlight"])
    def test_the_passage_coloured_up_to_its_end(self, tmp_path, hit):
        text = book()
        root = make_database(tmp_path, text)
        start, end = byte_of(text, "first") if hit else byte_of(text, "<hi>"), byte_of(text, "second")
        query = {"byte": [str(start)]} if hit else {}
        page = navigate(root, philo_id="1 1", start_byte=str(start), end_byte=str(end), **query)
        assert not re.search(r"<span[^>]*/>", page)
        assert '<span class="passage-0" n="0"></span>' in page
        coloured = Browser(page).in_passage
        assert coloured == self.between_its_markers(page)
        assert coloured.startswith(SHOWN.replace("bold", "first")[3:] if hit else SHOWN.replace("bold", "first"))
        if not hit:  # the text of its bytes, and that of the page without it (with hits: TestPassageWithHighlights)
            assert coloured == Browser(text[start:end]).text
            assert Browser(page).text == Browser(navigate(root, philo_id="1 1")).text
