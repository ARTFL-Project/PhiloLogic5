"""Unit tests for FragmentParser, which rebuilds the cut-off XML of concordances, KWICs and text objects, and for the
events of the ShlaxIngestor it builds them from, which TagCensus reads too."""

import sys
from pathlib import Path

import pytest
from lxml import etree

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime.FragmentParser import parse
from philologic.shlaxtree import ShlaxIngestor


def parsed(text):
    """The fragment as parse rebuilds it, without the wrapper div."""
    output = etree.tostring(parse(text), encoding="unicode")
    assert output.startswith('<div class="philologic-fragment">') and output.endswith("</div>")
    return output[len('<div class="philologic-fragment">') : -len("</div>")]


class Recorder:
    """An ingestor target keeping the events"""

    def __init__(self):
        self.events = []

    def feed(self, *event):
        self.events.append(event)

    def close(self):
        return self.events


def events(*pieces):
    ingestor = ShlaxIngestor(target=Recorder())
    for piece in pieces:
        ingestor.feed(piece)
    return ingestor.close()


@pytest.mark.unit
class TestShlaxIngestor:
    """Tests for the events of ShlaxIngestor: kind, text, offset, name, attributes"""

    def test_start_text_end(self):
        assert events("<w xml:id=\"w1\" n='1.1'>say</w>") == [
            ("start", "<w xml:id=\"w1\" n='1.1'>", 0, "w", {"xml:id": "w1", "n": "1.1"}),
            ("text", "say", 23, None, None),
            ("end", "</w>", 26, "w", None),
            ("text", "", 30, None, None),
        ]

    def test_empty_tags_get_an_end_event(self):
        """A "/" after the attributes makes a tag empty, and the tag ends there, even with a ">" further on"""
        assert events('a<lb/>b<pb n="2" / >c') == [
            ("text", "a", 0, None, None),
            ("start", "<lb/>", 1, "lb", {}),
            ("end", "", 1, "lb", None),
            ("text", "b", 6, None, None),
            ("start", '<pb n="2" /', 7, "pb", {"n": "2"}),
            ("end", "", 7, "pb", None),
            ("text", " >c", 18, None, None),
        ]

    def test_end_tag_name_without_the_space_after_it(self):
        assert events("<p>x</p\n>")[2] == ("end", "</p\n>", 4, "p", None)

    def test_comments_and_processing_instructions_give_no_events(self):
        assert events("<p>x<!-- c --><?pi y?>z</p>") == [
            ("start", "<p>", 0, "p", {}),
            ("text", "x", 3, None, None),
            ("text", "z", 22, None, None),
            ("end", "</p>", 23, "p", None),
            ("text", "", 27, None, None),
        ]

    def test_stray_less_than_dropped(self):
        assert events("a < b") == [("text", "a ", 0, None, None), ("text", " b", 3, None, None)]

    def test_tag_cut_off_at_the_end(self):
        assert events('a<hi rend="i"') == [
            ("text", "a", 0, None, None),
            ("start", '<hi rend="i"', 1, "hi", {"rend": "i"}),
            ("text", "", 13, None, None),
        ]

    def test_fed_in_pieces(self):
        assert events("<w>a", "</w>") == [
            ("start", "<w>", 0, "w", {}),
            ("text", "a", 3, None, None),
            ("end", "</w>", 4, "w", None),
            ("text", "", 8, None, None),
        ]


@pytest.mark.unit
class TestFragmentParser:
    """Tests for FragmentParser.parse"""

    def test_word_tags_keep_their_attributes(self):
        assert parsed('<w xml:id="w1" n="1.1">say</w> <w xml:id="w2">you</w>') == (
            '<w n="1.1" id="w1">say</w> <w id="w2">you</w>'
        )

    def test_namespace_prefix_dropped_from_tag_names(self):
        assert parsed('<jx:cl a="1">t</jx:cl>') == '<cl a="1">t</cl>'

    def test_single_and_double_quoted_attributes(self):
        assert parsed("<seg type='x' subtype=\"y\">t</seg>") == '<seg type="x" subtype="y">t</seg>'

    def test_empty_attribute_value(self):
        assert parsed('<w lemma="">e</w>') == '<w lemma="">e</w>'

    def test_empty_elements(self):
        assert parsed('a<lb/>b<pb n="2" facs="x.png" />c') == 'a<lb></lb>b<pb n="2" facs="x.png"></pb>c'

    def test_cut_off_fragment(self):
        """The partial tag at the start stays as text, the end tag with no start is dropped, open tags are closed"""
        assert parsed('ord">tail</w> <l n="3">line <hi rend="i">open') == (
            'ord"&gt;tail <l n="3">line <hi rend="i">open</hi></l>'
        )

    def test_end_tag_that_does_not_match_is_dropped(self):
        assert parsed("<p>a</q>b</p>") == "<p>ab</p>"

    def test_comments_and_processing_instructions_dropped(self):
        assert parsed("<p>x<!-- c --><?pi y?>z</p>") == "<p>xz</p>"
