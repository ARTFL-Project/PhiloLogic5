"""Unit tests for PlainTextParser: page markers and div1 sections."""

import io
import json
import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.loadtime.PlainTextParser import PlainTextParser

HATHI_MARKERS = r"^\{(?:[^{}]*_)?(\d+)\}$"

# The three forms of HathiTrust page markers, one of them after a line of text, and one with spaces around it
HATHI_TEXT = (
    "\n"
    "{UCAL_C3470619_00000001}\n"
    "\n"
    "THE ATHENÆUM\n"
    "Journal of literature.\n"
    "\n"
    "{UCAL_C3470619_00000002}\n"
    "\n"
    "Café ſont here, and there.\n"
    "A line ending a page\n"
    "{00000003}\n"
    "the paragraph goes on.\n"
    "\n"
    "   {39002007368898_000004}  \n"
    "\n"
    "Last words.\n"
)
HATHI_WORDS = [
    "the", "athenæum", "journal", "of", "literature",
    "café", "sont", "here", "and", "there", "a", "line", "ending", "a", "page",
    "the", "paragraph", "goes", "on",
    "last", "words",
]  # fmt: skip


def parse(text, **options):
    """Parse text with the plain text parser, and return its records as (type, name, philo_id, attributes)."""
    output = io.StringIO()
    parser = PlainTextParser(output, 1, len(text.encode()), known_metadata={"filename": "t.txt"}, **options)
    parser.parse(io.StringIO(text))
    records = []
    for line in output.getvalue().splitlines():
        philo_type, name, philo_id, attrib = line.split("\t")
        records.append((philo_type, name, [int(i) for i in philo_id.split()], json.loads(attrib)))
    return records


def of_type(records, philo_type):
    """The records of a type, in the order of their start bytes."""
    return sorted((r for r in records if r[0] == philo_type), key=lambda r: r[3]["start_byte"])


def line_start(text, line):
    """The byte offset of the start of the line of text (the first one) which begins with line."""
    position = text.index(line)
    assert position == 0 or text[position - 1] == "\n"
    return len(text[:position].encode())


def assert_tile(objects, start, end):
    """Assert that objects follow each other from start to end, without gaps or overlaps."""
    assert objects
    assert objects[0][3]["start_byte"] == start
    for current, following in zip(objects, objects[1:]):
        assert current[3]["end_byte"] == following[3]["start_byte"]
    assert objects[-1][3]["end_byte"] == end


@pytest.mark.unit
class TestPageMarkers:
    """Tests for the lines which page_marker_regex makes page breaks."""

    def test_words_are_at_their_bytes(self):
        """Test that every word's byte range slices out that word, after page markers and multibyte characters."""
        words = of_type(parse(HATHI_TEXT, page_marker_regex=HATHI_MARKERS, plain_text_section_length=60), "word")
        text = HATHI_TEXT.encode()
        for _, name, _, attrib in words:
            assert text[attrib["start_byte"] : attrib["end_byte"]].decode().lower().replace("ſ", "s") == name

    @pytest.mark.parametrize("section_length", [0, 60])
    def test_markers_are_not_indexed(self, section_length):
        """Test that the words of page markers are left out of the index, and no other words."""
        records = parse(HATHI_TEXT, page_marker_regex=HATHI_MARKERS, plain_text_section_length=section_length)
        assert [r[1] for r in of_type(records, "word")] == HATHI_WORDS

    def test_markers_are_indexed_without_a_regex(self):
        """Test that, without a page_marker_regex, the words of page markers are words of the text, as they were."""
        words = [r[1] for r in of_type(parse(HATHI_TEXT, plain_text_section_length=0), "word")]
        assert words.count("ucal") == 2
        assert not of_type(parse(HATHI_TEXT, plain_text_section_length=0), "page")

    def test_pages(self):
        """Test that pages start at their marker lines, follow each other to the end, and are numbered by them."""
        records = parse(HATHI_TEXT, page_marker_regex=HATHI_MARKERS, plain_text_section_length=0)
        pages = of_type(records, "page")
        assert [page[3]["start_byte"] for page in pages] == [
            line_start(HATHI_TEXT, "{UCAL_C3470619_00000001}"),
            line_start(HATHI_TEXT, "{UCAL_C3470619_00000002}"),
            line_start(HATHI_TEXT, "{00000003}"),
            line_start(HATHI_TEXT, "   {39002007368898_000004}"),
        ]
        assert_tile(pages, pages[0][3]["start_byte"], len(HATHI_TEXT.encode()))
        assert [page[3]["n"] for page in pages] == ["1", "2", "3", "4"]
        assert [page[2] for page in pages] == [[1, 0, 0, 0, 0, 0, 0, 0, i] for i in (1, 2, 3, 4)]

    def test_words_have_their_page(self):
        """Test that each word gets the id of the page it is on, which the index stores with it."""
        records = parse(HATHI_TEXT, page_marker_regex=HATHI_MARKERS, plain_text_section_length=60)
        pages = of_type(records, "page")
        for _, name, philo_id, attrib in of_type(records, "word"):
            [page] = [p for p in pages if p[3]["start_byte"] <= attrib["start_byte"] < p[3]["end_byte"]]
            assert attrib["page"] == philo_id[8] == page[2][8], name

    def test_marker_ends_the_paragraph(self):
        """Test that a marker after a line of text ends its paragraph, and the next line starts another one."""
        paras = of_type(parse(HATHI_TEXT, page_marker_regex=HATHI_MARKERS, plain_text_section_length=0), "para")
        spans = [(p[3]["start_byte"], p[3]["end_byte"]) for p in paras]
        assert (line_start(HATHI_TEXT, "Café"), line_start(HATHI_TEXT, "{00000003}")) in spans
        assert (line_start(HATHI_TEXT, "the paragraph"), line_start(HATHI_TEXT, "   {39002007368898")) in spans
        assert len(paras) == 4

    def test_marker_without_a_number(self):
        """Test that a page whose marker the regex finds no number in is numbered "na", as XML pages with no n."""
        records = parse("Some text.\n[page]\nMore text.\n", page_marker_regex=r"\[page\]", plain_text_section_length=0)
        assert [page[3]["n"] for page in of_type(records, "page")] == ["na"]
        assert [r[1] for r in of_type(records, "word")] == ["some", "text", "more", "text"]


def pages_text(page_count):
    """A text of pages of 32 characters (and bytes) each, a marker and a paragraph: page n starts at 32 * (n - 1)."""
    text = "".join(f"{{{n:08d}}}\n\nWord of page {n:04d}.\n\n" for n in range(1, page_count + 1))
    assert len(text) == len(text.encode()) == 32 * page_count
    return text


PARAGRAPH = "A paragraph of no page.\n\n"
GREEK_PARAGRAPH = "αβγδεζηθικλμνξοπρ.\n\n"  # 20 characters, 37 bytes


def section_spans(records):
    """The (start, end) bytes of the div1 sections."""
    return [(s[3]["start_byte"], s[3]["end_byte"]) for s in of_type(records, "div1")]


def section_heads(records):
    """The heads of the div1 sections."""
    return [s[3]["head"] for s in of_type(records, "div1")]


@pytest.mark.unit
class TestSections:
    """Tests for the div1 sections which plain_text_section_length splits texts into."""

    def test_sections_end_at_the_first_marker_past_the_size(self):
        """Test that a section ends at the first marker once it holds plain_text_section_length characters."""
        text = pages_text(5)
        records = parse(text, page_marker_regex=HATHI_MARKERS, plain_text_section_length=64)
        assert section_spans(records) == [(0, 64), (64, 128), (128, 160)]
        assert section_heads(records) == ["Pages 1–2", "Pages 3–4", "Page 5"]
        # A section of 64 characters goes on to the next marker
        records = parse(text, page_marker_regex=HATHI_MARKERS, plain_text_section_length=65)
        assert section_spans(records) == [(0, 96), (96, 160)]
        assert section_heads(records) == ["Pages 1–3", "Pages 4–5"]

    def test_sections_and_pages_tile_the_doc(self):
        """Test that sections follow each other from the start of the doc to its end, pages from the first marker."""
        for text in (pages_text(7), HATHI_TEXT, PARAGRAPH * 3 + pages_text(4)):
            records = parse(text, page_marker_regex=HATHI_MARKERS, plain_text_section_length=50)
            sections = of_type(records, "div1")
            assert len(sections) > 1
            assert all(s[1] != "__philo_virtual" for s in sections)
            assert_tile(sections, 0, len(text.encode()))
            pages = of_type(records, "page")
            assert_tile(pages, pages[0][3]["start_byte"], len(text.encode()))
            [doc] = of_type(records, "doc")
            assert (doc[3]["start_byte"], doc[3]["end_byte"]) == (0, len(text.encode()))
            for _, name, philo_id, attrib in of_type(records, "word") + of_type(records, "para"):
                [section] = [s for s in sections if s[2][1] == philo_id[1]]
                assert section[3]["start_byte"] <= attrib["start_byte"] < attrib["end_byte"] <= section[3]["end_byte"]

    def test_section_with_no_pages(self):
        """Test that a section with no pages is labeled with its number, and the pages of the next one with theirs."""
        text = PARAGRAPH * 5 + pages_text(4)
        start = len(PARAGRAPH.encode()) * 5
        records = parse(text, page_marker_regex=HATHI_MARKERS, plain_text_section_length=start)
        assert section_spans(records) == [(0, start), (start, len(text.encode()))]
        assert section_heads(records) == ["Section 1", "Pages 1–4"]

    def test_long_sections_end_at_a_paragraph(self):
        """Test that a section with no markers ends at the first paragraph once it holds twice the section size."""
        size = len(PARAGRAPH.encode())
        records = parse(PARAGRAPH * 7, plain_text_section_length=size)
        breaks = [2 * size, 4 * size, 6 * size]
        assert section_spans(records) == list(zip([0] + breaks, breaks + [7 * size]))
        assert section_heads(records) == ["Section 1", "Section 2", "Section 3", "Section 4"]

    def test_long_page(self):
        """Test that a page too long for a section is split at paragraphs, and its sections are labeled with it."""
        marker = "{00000001}\n\n"
        text = marker + PARAGRAPH * 5 + "{00000002}\n\n" + PARAGRAPH
        size = len(PARAGRAPH.encode())
        records = parse(text, page_marker_regex=HATHI_MARKERS, plain_text_section_length=size)
        first = len(marker.encode())
        breaks = [first + 2 * size, first + 4 * size, first + 5 * size]  # two paragraphs, then the second marker
        assert section_spans(records) == list(zip([0] + breaks, breaks + [len(text.encode())]))
        assert section_heads(records) == ["Page 1", "Page 1", "Page 1", "Page 2"]

    def test_sections_get_their_ids(self):
        """Test that sections are numbered in the doc, with philo_div1_id as the XML parser gives it."""
        records = parse(pages_text(5), page_marker_regex=HATHI_MARKERS, plain_text_section_length=64)
        sections = of_type(records, "div1")
        assert [s[2][:7] for s in sections] == [[1, i, 0, 0, 0, 0, 0] for i in (1, 2, 3)]
        assert [s[3]["philo_div1_id"] for s in sections] == ["1 1", "1 2", "1 3"]

    def test_section_lengths_are_in_characters(self):
        """Test that sections are as long in characters, not bytes, in a text whose letters take two bytes."""
        greek_page = "{00000001}\n\nσελίδα αβγδεζηθικλμνξ 001.\n\n"
        assert (len(greek_page), len(greek_page.encode())) == (40, 60)
        text = "".join(greek_page.replace("1", str(n)) for n in range(1, 6))
        records = parse(text, page_marker_regex=HATHI_MARKERS, plain_text_section_length=60)
        # The third marker is the first 60 characters in (the second one is 60 bytes in), at byte 120
        assert section_spans(records) == [(0, 120), (120, 240), (240, 300)]
        assert section_heads(records) == ["Pages 1–2", "Pages 3–4", "Page 5"]
        # Without markers, the first paragraph 60 characters in (the second one is 74 bytes in) ends a section
        size = len(GREEK_PARAGRAPH.encode())
        assert (len(GREEK_PARAGRAPH), size) == (20, 37)
        records = parse(GREEK_PARAGRAPH * 7, plain_text_section_length=30)
        assert section_spans(records) == [(0, 3 * size), (3 * size, 6 * size), (6 * size, 7 * size)]
        words = of_type(records, "word")
        assert [w[3]["start_byte"] for w in words] == [i * size for i in range(7)]
        assert all(w[3]["end_byte"] - w[3]["start_byte"] == 34 for w in words)

    @pytest.mark.parametrize("section_length", [0, None])
    def test_no_sections(self, section_length):
        """Test that a section size of 0 or None leaves the text in one virtual div1, as it was."""
        text = pages_text(5) + PARAGRAPH * 10
        records = parse(text, page_marker_regex=HATHI_MARKERS, plain_text_section_length=section_length)
        [div1] = of_type(records, "div1")
        assert div1[1] == "__philo_virtual" and "head" not in div1[3]
        assert len(of_type(records, "page")) == 5

    def test_sections_by_default(self):
        """Test that the parser splits texts into sections of 100,000 characters unless told otherwise."""
        text = pages_text(9_000)
        sections = of_type(parse(text, page_marker_regex=HATHI_MARKERS), "div1")
        assert [s[3]["start_byte"] for s in sections][:3] == [0, 100_000, 200_000]  # 100,000 is 32 * 3,125
        assert sections[0][3]["head"] == "Pages 1–3125"


@pytest.mark.unit
def test_load_config_options_reach_the_parser(tmp_path, monkeypatch):
    """Test that page_marker_regex and plain_text_section_length of a load_config get to the plain text parser."""
    from philologic.loadtime import LoadFilters
    from philologic.loadtime.Loader import Loader
    from philologic.loadtime.LoadOptions import LoadOptions

    # The Loader keeps its state in class attributes, which other tests load corpora with
    for attribute in ("parser_config", "metadata_fields", "metadata_types", "metadata_hierarchy", "data_dicts",
                      "filequeue"):  # fmt: skip
        monkeypatch.setattr(Loader, attribute, type(getattr(Loader, attribute))())
    for attribute in ("post_filters", "debug", "words_to_index", "destination", "workdir", "textdir", "web_app_dir",
                      "default_object_level", "token_regex", "cores", "ascii_conversion", "metadata_sql_types",
                      "suppress_word_attributes"):  # fmt: skip
        monkeypatch.setattr(Loader, attribute, getattr(Loader, attribute, None), raising=False)
    (tmp_path / "t.txt").write_text(pages_text(5), encoding="utf8")
    load_config = tmp_path / "load_config.py"

    def parse_with(config):
        load_config.write_text(config, encoding="utf8")
        load_options = LoadOptions()
        load_options.parse(["philoload5", "-t", "plain_text", "-l", str(load_config), "db", str(tmp_path / "t.txt")])
        destination = tmp_path / f"db{len(list(tmp_path.glob('db*')))}"
        load_options["db_destination"] = str(destination)
        load_options["data_destination"] = str(destination / "data")
        load_options["load_filters"] = [LoadFilters.generate_words_sorted, LoadFilters.generate_pages]
        load_options["debug"] = True  # keeps the parser's output
        loader = Loader.set_class_attributes(load_options.values)
        loader.add_files(load_options.files)
        loader.set_file_data(None, loader.textdir, loader.workdir)
        Loader.parse_file(0)
        records = []
        for line in Path(Loader.filequeue[0]["raw"]).read_text(encoding="utf8").splitlines():
            philo_type, name, philo_id, attrib = line.split("\t")
            records.append((philo_type, name, philo_id.split(), json.loads(attrib)))
        return records

    records = parse_with(f'page_marker_regex = r"{HATHI_MARKERS}"\nplain_text_section_length = 64\n')
    assert section_heads(records) == ["Pages 1–2", "Pages 3–4", "Page 5"]
    assert len(of_type(records, "page")) == 5
    # A size of 0 in a load_config, which LoadConfig would drop as an empty value, turns sections off
    records = parse_with(f'page_marker_regex = r"{HATHI_MARKERS}"\nplain_text_section_length = 0\n')
    assert [r[1] for r in of_type(records, "div1")] == ["__philo_virtual"]
    assert len(of_type(records, "page")) == 5
    # Without them, there are sections of 100,000 characters, and no pages
    records = parse_with("words_to_index = ''\n")
    assert section_heads(records) == ["Section 1"]
    assert not of_type(records, "page")
