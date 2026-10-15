"""Sentences of early prints, whose virgules (/) and pilcrows (¶) do what later full stops do, and of lists: the same in
the XML parser with and without a tokenizer, and in the plain text parser"""

import io
import json

import pytest

from philologic.loadtime.Parser import XMLParser, pilcrow_between, virgule_ends_sentence
from philologic.loadtime.PlainTextParser import PlainTextParser

MODES = [pytest.param({}, id="token_regex"), pytest.param({"language": "en"}, id="spacy")]


def sentences(parser_class, text, **options):
    """The words of each sentence (as in the text), with its name and the end of its text"""
    output = io.StringIO()
    data = text.encode("utf8")
    parser_class(output, 1, len(data), known_metadata={"filename": "t"}, metadata_sql_types={}, **options).parse(
        io.StringIO(text)
    )
    words, ends = {}, {}
    for line in output.getvalue().split("\n")[:-1]:
        kind, name, philo_id, attrib = line.split("\t")
        attrib = json.loads(attrib)
        key = tuple(philo_id.split()[:6])
        if kind == "word":
            words.setdefault(key, []).append(data[attrib["start_byte"] : attrib["end_byte"]].decode("utf8"))
        elif kind == "sent":
            ends[key] = (name, data[: attrib["end_byte"]].decode("utf8"))
    return [(" ".join(words[key]), ends[key][0], ends[key][1]) for key in words]


def xml_sentences(body, **options):
    return [
        (words, name)
        for words, name, _ in sentences(XMLParser, f"<text><body><div>{body}</div></body></text>", **options)
    ]


@pytest.mark.unit
@pytest.mark.parametrize("options", MODES)
def test_a_virgule_after_lowercase_before_a_capital_ends_a_sentence(options):
    body = "<p>he wente to rome / And thempe<g ref='char:EOLhyphen'/>rour had despyte / I wote not / for why</p>"
    words = [w.lower() for w, _ in xml_sentences(body, **options)]
    assert words == ["he wente to rome", "and thempe<g ref='char:eolhyphen'/>rour had despyte", "i wote not for why"]
    assert [name for _, name in xml_sentences(body, **options)] == ["/", "/", "__philo_virtual"]


@pytest.mark.unit
@pytest.mark.parametrize("options", MODES)
def test_slashes_which_join_words_or_capitals_dont(options):
    body = "<p>and/or I/O Wendy/Janet cotton/Lycra THÉOPHILE GAUTIER / NOUVELLES / Paris 1 / 2 rome /And</p>"
    assert len(xml_sentences(body, **options)) == 1


@pytest.mark.unit
@pytest.mark.parametrize("options", MODES)
def test_a_pilcrow_ends_a_sentence_which_has_words(options):
    """In the text between words, or at the start of the next word's token (spaCy's tokenizers keep it with the word):
    the sentence ends with its last word then. At the start of a paragraph, it ends nothing."""
    body = "<p>¶ Of saynt Marcell he wente ¶ For what cause / ¶ It was so ¶Thirdly he dyed</p><p>¶And after</p>"
    result = xml_sentences(body, **options)
    assert [w.lower() for w, _ in result] == [
        "of saynt marcell he wente",
        "for what cause",
        "it was so",
        "thirdly he dyed",
        "and after",
    ]
    assert [name for _, name in result][:3] == ["¶", "/ ¶", "¶"]
    ends = [end for _, _, end in sentences(XMLParser, f"<text><body><div>{body}</div></body></text>", **options)]
    assert ends[2].endswith("so ¶" if not options else "so")  # a pilcrow kept with the next word starts its sentence


@pytest.mark.unit
@pytest.mark.parametrize("options", MODES)
def test_list_items_end_sentences(options):
    body = "<list><item>Saynt andrewe four score</item><item>Saynt nycholas</item></list><p>And thenne</p>"
    result = sentences(XMLParser, f"<text><body><div>{body}</div></body></text>", **options)
    assert [(w.lower(), name) for w, name, _ in result] == [
        ("saynt andrewe four score", "__philo_virtual"),
        ("saynt nycholas", "__philo_virtual"),
        ("and thenne", "__philo_virtual"),
    ]
    assert result[0][2].endswith("</item>")


@pytest.mark.unit
def test_tagged_sentences_go_on_across_items():
    body = "<list><item><s>Saynt andrewe</item><item>four score</s></item></list>"
    assert len(xml_sentences(body)) == 1


@pytest.mark.unit
@pytest.mark.parametrize("options", MODES)
def test_plain_text(options):
    text = "¶ Of saynt Marcell he wente to rome / And had despyte / ¶ For what cause ¶Thirdly he dyed / and so forth\n"
    assert [(w.lower(), name) for w, name, _ in sentences(PlainTextParser, text, **options)] == [
        ("of saynt marcell he wente to rome", "/"),
        ("and had despyte", "/ ¶"),
        ("for what cause", "¶"),
        ("thirdly he dyed and so forth", "__philo_virtual"),
    ]


@pytest.mark.unit
def test_the_rules():
    assert virgule_ends_sentence(" / ", "rome", "And")
    assert virgule_ends_sentence(" / ¶ ", "rome", "For")
    assert virgule_ends_sentence(" / ", "rome", "¶Thirdly")  # the first letter of the next word
    assert not virgule_ends_sentence(" / ", "Rome", "And")
    assert not virgule_ends_sentence(" / ", "rome", "and")
    assert not virgule_ends_sentence("/", "and", "Or")
    assert not virgule_ends_sentence(" /", "rome", "And")
    assert not virgule_ends_sentence(" / ", "1483", "And")
    assert not virgule_ends_sentence(" / ", "rome", None)
    assert pilcrow_between(" ¶ ", "For") and pilcrow_between(" ", "¶Thirdly") and not pilcrow_between(" ", "For")
