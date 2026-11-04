"""Unit tests for the page markers of plain texts (philologic.utils.page_markers): the lines of page_marker_regex,
which the plain-text parser makes pages of, and the formatters make the <pb> tags of XML texts of."""

import io
import json
import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.loadtime.PlainTextParser import PlainTextParser
from philologic.utils.page_markers import (
    compile_page_marker_regex,
    cut_page_marker_start,
    page_marker_number,
    page_markers_to_pb,
)

HATHI_MARKERS = r"^\{(?:[^{}]*_)?(\d+)\}$"


def to_pb(text, page_marker_regex=HATHI_MARKERS):
    return page_markers_to_pb(text, page_marker_regex)


@pytest.mark.unit
class TestPageMarkersToPb:
    @pytest.mark.parametrize(
        "marker, n",
        [("{UCAL_C3470619_00000247}", "247"), ("{00000001}", "1"), ("{39002007368898_000000}", "0")],
    )
    def test_the_three_hathitrust_markers(self, marker, n):
        assert to_pb(f"end of a page.\n{marker}\nNext page.\n") == f'end of a page.\n<pb n="{n}"/>\nNext page.\n'

    @pytest.mark.parametrize(
        "group, n", [("00000247", "247"), ("247", "247"), ("0000", "0"), ("000000000000000000001200", "1200")]
    )
    def test_numbers_without_their_leading_zeros(self, group, n):
        assert to_pb(f"{{{group}}}\n") == f'<pb n="{n}"/>\n'

    @pytest.mark.parametrize(
        "regex, marker",
        [
            (r"\[page\]", "[page]"),  # no group
            (r"\[page(?: (\d+))?\]", "[page]"),  # a group that matched nothing
            (r"\[page (\w+)\]", "[page xiv]"),  # a group that is no number
        ],
    )
    def test_na_for_a_page_with_no_number(self, regex, marker):
        assert to_pb(f"text\n{marker}\nmore text\n", regex) == 'text\n<pb n="na"/>\nmore text\n'

    def test_whitespace_around_the_marker_kept(self):
        assert to_pb("a\n   {00000004}  \nb\n\t{00000005}\t\n") == 'a\n   <pb n="4"/>  \nb\n\t<pb n="5"/>\t\n'

    def test_the_regex_matches_the_whole_line(self):
        """Anchored or not"""
        regex = r"\[page (\d+)\]"
        assert to_pb("a\n  [page 12] \nb\n", regex) == 'a\n  <pb n="12"/> \nb\n'
        assert to_pb("see [page 12] here\n[page 12] and on\n", regex) == "see [page 12] here\n[page 12] and on\n"

    @pytest.mark.parametrize("line_break", ["\n", "\r\n", "\r"])
    def test_line_breaks_of_every_kind(self, line_break):
        """As the parser reads its lines, from a file opened with newline="" """
        text = line_break.join(["a", "{00000001}", "b", "{00000002}", ""])
        assert to_pb(text) == line_break.join(["a", '<pb n="1"/>', "b", '<pb n="2"/>', ""])

    def test_markers_at_the_start_and_end_of_the_text(self):
        assert to_pb("{00000001}\nword\n{00000002}") == '<pb n="1"/>\nword\n<pb n="2"/>'
        assert to_pb("{00000003}") == '<pb n="3"/>'

    @pytest.mark.parametrize(
        "line",
        [
            "foo {00000003}",
            "{00000003} bar",
            "{braces} in a line",
            "{not_a_marker}",
            "{{00000001}}",
            "{UCAL_C3470619_00000247",
            "UCAL_C3470619_00000247}",
            "{}",
            "{ 00000001 }",
            "<pb n=\"3\"/>",
        ],
    )
    def test_other_lines_with_braces_unchanged(self, line):
        text = f"a\n{line}\nb\n"
        assert to_pb(text) == text

    @pytest.mark.parametrize("regex", [None, ""])
    def test_no_regex_no_change(self, regex):
        text = "a\n{UCAL_C3470619_00000247}\n{00000001}\nb"
        assert to_pb(text, regex) is text

    def test_text_without_markers_unchanged(self):
        text = "Some text\nwith lines\n\nand {braces}.\n"
        assert to_pb(text) is text
        assert to_pb("") == ""

    def test_regex_compiled_once(self):
        assert compile_page_marker_regex(HATHI_MARKERS) is compile_page_marker_regex(HATHI_MARKERS)


@pytest.mark.unit
class TestPageMarkerNumber:
    def test_the_line_whole_but_for_its_whitespace(self):
        regex = compile_page_marker_regex(HATHI_MARKERS)
        assert page_marker_number("  {UCAL_C3470619_00000247} \r\n", regex) == "247"
        assert page_marker_number("{UCAL_C3470619_00000247} x\n", regex) is None
        assert page_marker_number("\n", regex) is None


@pytest.mark.unit
class TestCutPageMarkerStart:
    """A concordance's text, cut out of the file by bytes, may end in the middle of a marker line"""

    @pytest.mark.parametrize("cut", ["{", "{U", "{UCAL_C3470619_000", "   {0000", "{39002007368898_"])
    def test_the_start_of_a_marker(self, cut):
        text = f"a line\n{cut}".encode()
        assert cut_page_marker_start(text, HATHI_MARKERS) == len(b"a line\n")

    @pytest.mark.parametrize("line_break", [b"\n", b"\r\n", b"\r"])
    def test_after_every_line_break(self, line_break):
        assert cut_page_marker_start(b"a" + line_break + b"{UCAL_", HATHI_MARKERS) == 1 + len(line_break)
        assert cut_page_marker_start(b"{UCAL_", HATHI_MARKERS) == 0

    @pytest.mark.parametrize(
        "text",
        [
            "a line\n{00000001}",  # a whole marker, which page_markers_to_pb makes a <pb>
            "a line\n{00000001}  ",
            "a line\n{00000001}\n",
            "a line\n{UCAL_C3470619_00000247}\n",
            "a line\n{braces} in",
            "a line\nwords {UCAL_",  # no marker at the start of the line
            "a line\n",
            "a line\n   ",
            "",
        ],
    )
    def test_no_marker_cut(self, text):
        assert cut_page_marker_start(text.encode(), HATHI_MARKERS) is None

    @pytest.mark.parametrize("regex", [None, ""])
    def test_no_regex(self, regex):
        assert cut_page_marker_start(b"a line\n{UCAL_", regex) is None


def parser_pages(text, page_marker_regex):
    """The (start_byte, n) of the pages the plain-text parser makes of text, read as the loader reads its files."""
    output = io.StringIO()
    parser = PlainTextParser(
        output, 1, len(text.encode()), known_metadata={"filename": "t.txt"}, page_marker_regex=page_marker_regex,
        plain_text_section_length=0,
    )  # fmt: skip
    parser.parse(io.StringIO(text, newline=""))
    pages = []
    for line in output.getvalue().splitlines():
        philo_type, _, _, attrib = line.split("\t")
        if philo_type == "page":
            attrib = json.loads(attrib)
            pages.append((attrib["start_byte"], attrib["n"]))
    return sorted(pages)


@pytest.mark.unit
@pytest.mark.parametrize(
    "text, regex",
    [
        (
            "\n{UCAL_C3470619_00000001}\n\nTHE ATHENÆUM\n{00000002}\nCafé ſont here,\n   {39002007368898_000003}  \n"
            "{braces} and {00000004} in a line\n{UCAL_C3470619_00000005}",
            HATHI_MARKERS,
        ),
        ("a\r\n{39002007368898_000000}\r\nb\r{00000002}\rc\n", HATHI_MARKERS),
        ("see [page 3] here\n[page 4]\n  [page 05]\n", r"\[page (\d+)\]"),
        ("Some text.\n[page]\nMore text.\n[page 12]\n", r"\[page(?: (\d+))?\]"),
        ("a\n\nb\n  \n", r"(\d*)"),  # a regex that blank lines match: they are pages to the parser too
    ],
)
def test_the_runtime_finds_the_pages_of_the_parser(text, regex):
    """The lines made <pb> tags are the lines the parser started its pages at, with the same numbers."""
    expected = parser_pages(text, regex)
    assert expected
    lines = list(io.StringIO(text, newline=""))
    converted = list(io.StringIO(page_markers_to_pb(text, regex), newline=""))
    assert len(converted) == len(lines)
    found = []
    position = 0
    for line, converted_line in zip(lines, converted):
        if converted_line != line:
            found.append((position, converted_line.strip()[len('<pb n="') : -len('"/>')]))
        position += len(line.encode())
    assert found == expected
