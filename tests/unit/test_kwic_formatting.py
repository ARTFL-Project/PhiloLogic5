"""Unit tests for the text of KWIC lines: format_strip, which keeps the <w> of TEI as spans with their attributes, and
hit_bounds, which finds the hit in it."""

import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime.ObjectFormatter import format_concordance, format_strip
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


HIT = '<span class="highlight">{}</span>'


@pytest.mark.unit
@pytest.mark.parametrize("formatter", [format_concordance, format_strip], ids=["concordance", "strip"])
class TestPartialTagAtTheEnd:
    """The text ends in a partial tag when it was cut in a tag: it is cut off, but never at a "<" before a hit, which
    is text in OCR (the Century Dictionary's "[< G. blich")."""

    @staticmethod
    def run(formatter, text, hits, offsets="hits"):
        data = text.encode("utf8")
        if offsets == "hits":
            offsets = [data.index(hit.encode("utf8")) for hit in hits]
        return formatter(data, TOKEN_REGEX, offsets)

    def test_stray_less_than_before_the_hit_keeps_the_hit(self, formatter):
        output = self.run(formatter, "the word [< G. blich and hit here", ["hit"])
        assert HIT.format("hit") + " here" in output
        assert "blich and " + HIT.format("hit") in output

    def test_stray_less_than_with_the_hit_at_the_end(self, formatter):
        assert HIT.format("hit") in self.run(formatter, "a [< b hit", ["hit"])

    def test_partial_tag_after_the_last_hit_is_cut(self, formatter):
        output = self.run(formatter, "see the hit then <zz marker", ["hit"])
        assert HIT.format("hit") + " then " in output
        assert "zz" not in output and "marker" not in output

    def test_partial_tag_right_after_the_hit_is_cut(self, formatter):
        output = self.run(formatter, "see the hit<zz marker", ["hit"])
        assert HIT.format("hit") in output
        assert "zz" not in output and "marker" not in output

    def test_several_hits_with_a_less_than_between_them(self, formatter):
        output = self.run(formatter, "a first b < c second d", ["first", "second"])
        assert HIT.format("first") in output
        assert HIT.format("second") + " d" in output

    def test_several_hits_with_a_partial_tag_after_the_last(self, formatter):
        output = self.run(formatter, "a first b c second d <i marker", ["first", "second"])
        assert HIT.format("first") in output
        assert HIT.format("second") + " d " in output
        assert "marker" not in output

    @pytest.mark.parametrize("offsets", [None, []], ids=["none", "empty"])
    def test_without_offsets_the_text_is_cut_at_the_first_unclosed_less_than(self, formatter, offsets):
        output = self.run(formatter, "the word [< G. blich and hit here", None, offsets)
        assert "blich" not in output and "hit" not in output
        assert "word [" in output

    @pytest.mark.parametrize("offsets", [None, []], ids=["none", "empty"])
    def test_without_offsets_a_partial_tag_is_cut(self, formatter, offsets):
        output = self.run(formatter, "see some text <zz marker", None, offsets)
        assert "zz" not in output and "marker" not in output and "text" in output

    def test_start_guard_keeps_the_hit_with_a_greater_than_after_it(self, formatter):
        output = self.run(formatter, "a hit b > c", ["hit"])
        assert HIT.format("hit") + " b > c" in output

    def test_start_cutoff_before_the_hit_and_stray_less_than_before_it(self, formatter):
        """Both guards at once: the tail of a tag is removed, the "<" is kept"""
        output = self.run(formatter, "rest of tag> a [< b hit c", ["hit"])
        assert "rest of tag" not in output
        assert "a [ b " + HIT.format("hit") + " c" in output

    def test_start_cutoff_and_partial_tag_after_the_hit(self, formatter):
        output = self.run(formatter, "rest of tag> y hit z <b marker", ["hit"])
        assert "rest of tag" not in output and "marker" not in output
        assert " y " + HIT.format("hit") + " z " in output

    def test_hits_around_a_greater_than_and_a_partial_tag(self, formatter):
        output = self.run(formatter, "a hit b > c hit2 d <i marker", ["hit", "hit2"])
        assert HIT.format("hit") + " b > c " + HIT.format("hit2") + " d " in output
        assert "marker" not in output

    @pytest.mark.parametrize("offsets", [None, []], ids=["none", "empty"])
    def test_without_offsets_a_partial_tag_at_the_very_start_is_cut(self, formatter, offsets):
        output = self.run(formatter, "<zz marker", None, offsets)
        assert "marker" not in output

    def test_a_hit_at_the_start_of_the_partial_tag_is_not_cut(self, formatter):
        """The "<" is at the last hit: it is not after it, so the text is kept"""
        output = self.run(formatter, "see <zz marker", None, [4])
        assert "marker" in output
