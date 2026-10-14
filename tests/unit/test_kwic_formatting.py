"""Unit tests for the text of KWIC lines: format_strip, which keeps the <w> of TEI as spans with their attributes, and
hit_bounds, which finds the hit in it."""

import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime.ObjectFormatter import format_strip
from philologic.runtime.reports.kwic import hit_bounds

TOKEN_REGEX = r"[\p{L}\p{M}\p{N}]+|[&\p{L};]+"


def kwic_text(text, *hits):
    """format_strip's text for the hits, each given as the text that follows its first byte"""
    data = text.encode("utf8")
    return format_strip(data, TOKEN_REGEX, [data.index(hit.encode("utf8")) for hit in hits])


def columns(text, *hits):
    """The text before the hit, the hit and the text after it"""
    output = kwic_text(text, *hits)
    start, end = hit_bounds(output)
    return output[:start], output[start:end], output[end:]


@pytest.mark.unit
class TestFormatStrip:
    def test_word_tags_kept_with_their_attributes(self):
        assert kwic_text('<l><w xml:id="w1" n="1.1">my</w> <w xml:id="w2" n="1.1">love</w></l>', "love<") == (
            '<span class="xml-w" n="1.1" id="w1">my</span> '
            '<span class="xml-w" n="1.1" id="w2"><span class="highlight">love</span></span>'
        )

    def test_other_tags_removed(self):
        assert kwic_text("<p>a <hi>plain</hi> text with love in it</p>", "love") == (
            'a plain text with <span class="highlight">love</span> in it'
        )

    def test_attribute_values_escaped_once(self):
        """As in the source, but for quotes and ">"; no lang, which gave accessibility issues"""
        text = '<l><w lemma="a&amp;b" pos="&quot;x&quot;" n=\'say "hi"\' rend="a>b" xml:lang="grc">love</w></l>'
        assert kwic_text(text, "love<") == (
            '<span class="xml-w" lemma="a&amp;b" pos="&quot;x&quot;" n="say &quot;hi&quot;" rend="a&gt;b">'
            '<span class="highlight">love</span></span>'
        )

    def test_spaces_around_hyphens_removed_outside_tags_only(self):
        assert kwic_text('<p><w n="a - b">x</w> well - known</p>', "x<") == (
            '<span class="xml-w" n="a - b"><span class="highlight">x</span></span> well-known'
        )


@pytest.mark.unit
class TestHitBounds:
    def test_hit_in_a_word_tag(self):
        assert columns("<l><w>my</w> <w>love</w> <w>is</w></l>", "love<") == (
            '<span class="xml-w">my</span> ',
            '<span class="xml-w"><span class="highlight">love</span></span>',
            ' <span class="xml-w">is</span>',
        )

    def test_hit_in_plain_text(self):
        assert columns("<p>my love is</p>", "love") == ("my ", '<span class="highlight">love</span>', " is")

    def test_word_tag_with_more_than_the_hit(self):
        """The hit has all of its <w>: the columns stay well-formed"""
        assert columns("<l><w>my</w> <w>Love’s</w> <w>mind</w></l>", "Love’s") == (
            '<span class="xml-w">my</span> ',
            '<span class="xml-w"><span class="highlight">Love</span>’s</span>',
            ' <span class="xml-w">mind</span>',
        )

    def test_hit_of_two_words(self):
        assert columns("<l><w>my</w> <w>dear</w> <w>lord</w> <w>and</w></l>", "dear<", "lord<") == (
            '<span class="xml-w">my</span> ',
            '<span class="xml-w"><span class="highlight">dear</span></span> '
            '<span class="xml-w"><span class="highlight">lord</span></span>',
            ' <span class="xml-w">and</span>',
        )

    def test_end_tag_with_no_start_ignored(self):
        text = 'a</span> <span class="highlight">b</span> c'
        assert hit_bounds(text) == (text.index('<span class="highlight">'), text.rindex(" c"))

    def test_no_highlight(self):
        with pytest.raises(ValueError):
            hit_bounds('<span class="xml-w">my</span> love')
