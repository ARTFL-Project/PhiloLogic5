"""Unit tests for the text of KWIC lines: format_strip, which keeps the <w> of TEI as spans with their attributes, and
hit_bounds, which finds the hit in it."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime.ObjectFormatter import format_concordance, format_concordance_distinctive, format_strip
from philologic.runtime.reports import kwic as kwic_module
from philologic.runtime.reports.kwic import hit_bounds, kwic_hit_object

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

    def test_hit_at_the_start(self):
        """Right after the partial tag cut from the start: nothing before it"""
        assert columns('n="1" who="#a #b">hit and the rest</sp>', "hit") == (
            "", '<span class="highlight">hit</span>', " and the rest"
        )

    def test_hit_at_the_start_before_word_tags(self):
        assert columns('xml:id="w1" n="1.1">hit</w> <w>two</w>', "hit") == (
            "", '<span class="highlight">hit</span>', ' <span class="xml-w">two</span>'
        )


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


@pytest.mark.unit
class TestDistinctivePartialTagAtTheEnd:
    """The concordance of the distinctive uses cuts the partial tag at the end of its text as the others do: never at
    a "<" before its last hit, of either kind."""

    @staticmethod
    def run(text, main, distinctive):
        data = text.encode("utf8")
        offsets = [[data.index(word.encode("utf8")) for word in words] for words in (main, distinctive)]
        return format_concordance_distinctive(data, TOKEN_REGEX, *offsets)

    def test_stray_less_than_before_the_hits_keeps_them(self):
        output = self.run("the word [< G. blich and hit then sig here", ["hit"], ["sig"])
        assert HIT.format("hit") in output
        assert '<span class="colloc-explainer">sig</span> here' in output

    def test_stray_less_than_before_the_last_hit_only(self):
        output = self.run("a sig b [< c hit d", ["hit"], ["sig"])
        assert '<span class="colloc-explainer">sig</span>' in output and HIT.format("hit") + " d" in output

    def test_partial_tag_after_the_last_hit_is_cut(self):
        output = self.run("see the hit then sig <zz marker", ["hit"], ["sig"])
        assert HIT.format("hit") in output and '<span class="colloc-explainer">sig</span>' in output
        assert "zz" not in output and "marker" not in output


@pytest.mark.unit
def format_distinctive(text, word_regex, byte_offsets=None, page_marker_regex=None, file_type="xml"):
    """The concordance of the distinctive uses, its hits those of the query"""
    return format_concordance_distinctive(text, word_regex, byte_offsets or [], [], page_marker_regex, file_type)


@pytest.mark.parametrize(
    "formatter", [format_concordance, format_strip, format_distinctive], ids=["concordance", "strip", "distinctive"]
)
class TestHitAtTheStart:
    """The partial tag and word the text starts with are cut up to the first hit at most: a cut ending at it makes it
    the first byte of the text, which is highlighted as any other."""

    run = staticmethod(TestPartialTagAtTheEnd.run)

    def test_hit_right_after_a_cut_partial_tag(self, formatter):
        """The context starts in a tag (a long who= of a <sp>), which ends right before the hit"""
        output = self.run(formatter, 'n="1" who="#a #b #c">hit and the rest</sp>', ["hit"])
        assert HIT.format("hit") + " and the rest" in output
        assert "who" not in output

    def test_hit_right_after_a_greater_than_in_the_text(self, formatter):
        """The ">" of OCR right before the hit, with no "<" before it, taken for the end of a tag"""
        output = self.run(formatter, "the word ce>hit est pas", ["hit"])
        assert HIT.format("hit") + " est pas" in output
        assert "the word" not in output

    def test_hit_right_after_a_cut_partial_word(self, formatter):
        """The cut of the partial word stops at a space, "|", line break or "<", which no word starts with: a hit at
        the "<" of a tag there is at the start, and marked as it is at any other tag"""
        output = self.run(formatter, "rd<hi>hit</hi> more", None, [2])
        assert "rd" not in output and "more" in output
        assert output.count('<span class="highlight">') == 1
        assert output == self.run(formatter, "<hi>hit</hi> more", None, [0])

    def test_two_hits_the_first_at_the_start(self, formatter):
        output = self.run(formatter, 'n="1">hit and then hit2 here', ["hit", "hit2"])
        assert HIT.format("hit") + " and then " + HIT.format("hit2") + " here" in output

    @pytest.mark.parametrize("file_type", ["xml", "plain_text"])
    def test_hit_at_the_first_byte(self, formatter, file_type):
        """A hit at the first byte of the file, as a plain text's first word is: nothing was cut before it"""
        output = formatter(b"hit at the start & more", TOKEN_REGEX, [0], file_type=file_type)
        assert HIT.format("hit") + " at the start" in output

    def test_no_hit_before_the_text(self, formatter):
        """A hit before the first byte, which the text does not have, is no highlight at its start"""
        assert "highlight" not in formatter(b"word and more", TOKEN_REGEX, [-3])


class Hit(SimpleNamespace):
    def __getitem__(self, key):
        return getattr(self, key)


@pytest.mark.unit
class TestKwicLineWithTheHitAtTheStart:
    """kwic_hit_object, whose context (concordance_length bytes on each side of the hit) starts in a tag that ends right
    before the hit"""

    TEXT = 'Words before. <sp who="#a #b #c">hit and the end.</sp> More after it.'

    def line(self, monkeypatch, concordance_length):
        data = self.TEXT.encode("utf8")
        monkeypatch.setattr(kwic_module, "get_text", lambda hit, start, length, path: data[start : start + length])
        monkeypatch.setattr(kwic_module, "citation_links", lambda *args, **kwargs: {})
        monkeypatch.setattr(kwic_module, "citations", lambda *args, **kwargs: [])
        config = SimpleNamespace(concordance_length=concordance_length, db_path="", kwic_formatting_regex=[])
        db = SimpleNamespace(
            locals={"metadata_fields": [], "token_regex": TOKEN_REGEX, "page_marker_regex": None, "file_type": "xml"}
        )
        return kwic_hit_object(Hit(bytes=[data.index(b"hit")], philo_id=[1, 1, 1, 1, 1, 1, 1]), config, db)

    def test_aligned_with_nothing_before_the_hit(self, monkeypatch):
        result = self.line(monkeypatch, 15)  # from the 'who="' of the <sp>
        assert result["context"] == (
            '<span class="kwic-text"><span class="kwic-before"><span class="inner-before"></span></span>&nbsp;'
            '<span class="kwic-highlight"><span class="highlight">hit</span></span>&nbsp;'
            '<span class="kwic-after"> and the end</span></span>'
        )
        assert result["highlighted_text"] == "hit"

    def test_aligned_as_with_the_tag_before_it(self, monkeypatch):
        """The same line as when the context starts before the tag, but for the text before the hit"""
        whole = self.line(monkeypatch, 30)["context"].split('<span class="kwic-after">')[0]
        assert '<span class="inner-before"> before. </span>' in whole
        cut = self.line(monkeypatch, 15)["context"].split('<span class="kwic-after">')[0]
        assert cut == whole.replace(" before. ", "")
