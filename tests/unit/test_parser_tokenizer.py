"""XMLParser with a language, whose words spaCy's tokenizer for it splits (see XMLParser.flush_run): tags inside words
and entities don't split them, words leave out the punctuation at the edges of their token, which they keep in their
spacy_token attribute, every record has the exact bytes of its text in the file, and everything else is as without a
tokenizer."""

import io
import json
import sys
from pathlib import Path

import pytest
import regex as re

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.loadtime.Loader import Loader
from philologic.loadtime.Parser import XMLParser
from philologic.loadtime.spacy_tokenizer import spacy_tokenizer

pytest.importorskip("spacy")


def records(xml, **options):
    """(type, name, attributes) of the parser's output records for a document"""
    output = io.StringIO()
    parser = XMLParser(
        output, 1, len(xml.encode("utf8")), known_metadata={"filename": "snippet.xml"}, metadata_sql_types={}, **options
    )
    parser.parse(io.StringIO(xml))
    lines = output.getvalue().split("\n")[:-1]
    return [(line.split("\t")[0], line.split("\t")[1], json.loads(line.split("\t")[3])) for line in lines]


def words(xml, **options):
    """(word, its bytes in the document) of the words of a document"""
    data = xml.encode("utf8")
    return [
        (name, data[attrib["start_byte"] : attrib["end_byte"]].decode("utf8"))
        for kind, name, attrib in records(xml, **options)
        if kind == "word"
    ]


def document(text):
    return f"<text><body><div><p>{text}</p></div></body></text>"


@pytest.mark.unit
def test_inline_tags_dont_split_words():
    xml = document("wo<hi>r</hi>d <hi rend='initial'>L</hi>e roi, <hi>it</hi>alic M<sup>r</sup>")
    assert words(xml, language="en") == [
        ("word", "wo<hi>r</hi>d"),
        ("le", "L</hi>e"),
        ("roi", "roi"),
        ("italic", "it</hi>alic"),
        ("mr", "M<sup>r"),
    ]


@pytest.mark.unit
def test_entities_and_joined_hyphens_dont_split_words():
    xml = document("caf&eacute; na&shy;<lb/>ture l&apos;homme")
    assert words(xml, language="fr") == [
        ("café", "caf&eacute;"),
        ("nature", "na&shy;<lb/>ture"),
        ("l", "l"),
        ("homme", "homme"),
    ]


@pytest.mark.unit
def test_language_tokens():
    """Words leave out the punctuation at the edges of their token, not inside it; spacy_token keeps the token as it
    is in the text, for the tagger, when the word isn't that"""
    xml = document("L'homme dit-il aujourd'hui, ¶M. Smith")
    tokens = [(name, attrib.get("spacy_token")) for kind, name, attrib in records(xml, language="fr") if kind == "word"]
    assert tokens == [
        ("l", "L'"),
        ("homme", None),
        ("dit", None),
        ("il", "-il"),
        ("aujourd'hui", None),
        ("m", "¶M."),
        ("smith", "Smith"),
    ]
    assert [word for word, _ in words(xml, language="fr")] == [word for word, _ in tokens]
    assert [word for word, _ in words(xml)] == ["l", "homme", "dit", "il", "aujourd", "hui", "m", "smith"]


@pytest.mark.unit
def test_compounds_and_brackets():
    """Words with hyphens or apostrophes inside them which the tokenizer keeps whole are whole (as searches keep them,
    see Query.query_parse); characters not to index ([ ]) split words, as token_regex splits them"""
    xml = document("Il est peut-être lui-même, à Saint-Just. [L']Homme qu'[il] vit, pa[r]ole.")
    tokens = [(name, attrib.get("spacy_token")) for kind, name, attrib in records(xml, language="fr") if kind == "word"]
    assert tokens == [
        ("il", "Il"),
        ("est", None),
        ("peut-être", None),
        ("lui-même", None),
        ("à", None),
        ("saint-just", "Saint-Just"),
        ("l", "L"),
        ("homme", "Homme"),
        ("qu", None),
        ("il", None),
        ("vit", None),
        ("pa", None),
        ("r", None),
        ("ole", None),
    ]
    assert [text for _, text in words(xml, language="fr")][2:4] == ["peut-être", "lui-même"]


@pytest.mark.unit
def test_words_whatever_their_case():
    """spaCy's exceptions, which it has in lowercase, apply to the other cases too: Peut-être is whole as peut-être is,
    Est-il split as est-il is"""
    xml = document("Peut-être PEUT-ÊTRE peut-être. Est-il là ? EST-CE vrai ? Est-ce vrai ?")
    expected = ["peut-être"] * 3 + ["est", "il", "là"] + ["est", "ce", "vrai"] * 2
    assert [word for word, _ in words(xml, language="fr")] == expected


@pytest.mark.unit
def test_words_split_as_on_their_own():
    """A token the tokenizer keeps whole only where it is (est-ce-, before a dash) is split as the tokenizer splits its
    word on its own: the index has a word with a hyphen or apostrophe whole only if a search for it would keep it whole
    (see Query.query_parse)"""
    xml = document("est-ce- que vous")
    tokens = [(name, attrib.get("spacy_token")) for kind, name, attrib in records(xml, language="fr") if kind == "word"]
    assert tokens == [("est", None), ("ce", "-ce-"), ("que", None), ("vous", None)]
    assert [text for _, text in words(xml, language="fr")] == ["est", "ce", "que", "vous"]


@pytest.mark.unit
def test_tags_inside_words():
    """Proper names, glyphs and line breaks marked as inside words don't split words; other line breaks do"""
    xml = document(
        'à <pn>Saint</pn>-<pn>Germain</pn>, ex<g ref="char:EOLhyphen"/>ecution, Govern<lb type="hyphenInWord"/>ment, '
        'pa<lb break="no"/>role, <seg rend="decorInit">W</seg>E, expedient<lb/>demourer'
    )
    assert [word for word, _ in words(xml, language="fr")] == [
        "à",
        "saint-germain",
        "execution",
        "government",
        "parole",
        "we",
        "expedient",
        "demourer",
    ]


@pytest.mark.unit
def test_same_records_where_the_tokenizer_agrees():
    """Words the tokenizer and token_regex split alike give the same records with and without a tokenizer (but for the
    tokens of words in uppercase, kept for the tagger), with exact bytes: those of their text in the document,
    sentences from their first word to their punctuation (a sentence without punctuation ends with its paragraph)"""
    xml = document("He went home. He left! Did he? Yes, 1.5 times. It is a b cd. End") + "<p>Two «quoted» ; words</p>"
    with_tokenizer = records(xml, language="en")

    def without_tokens(records):
        return [(kind, name, {k: v for k, v in a.items() if k != "spacy_token"}) for kind, name, a in records]

    assert without_tokens(with_tokenizer) == records(xml)
    tokens = {attrib["spacy_token"] for kind, _, attrib in with_tokenizer if "spacy_token" in attrib}
    assert tokens == {"He", "Did", "Yes", "It", "End", "Two"}
    data = xml.encode("utf8")
    text = [
        (kind, data[attrib["start_byte"] : attrib["end_byte"]].decode("utf8"))
        for kind, name, attrib in with_tokenizer
        if kind == "punct" or (kind == "sent" and name != "__philo_virtual")
    ]
    assert text == [
        ("sent", "He went home."),
        ("punct", "."),
        ("sent", "He left!"),
        ("punct", "!"),
        ("sent", "Did he?"),
        ("punct", "?"),
        ("punct", ","),
        ("punct", "."),
        ("sent", "Yes, 1.5 times."),
        ("punct", "."),
        ("sent", "It is a b cd."),
        ("punct", "."),
        ("punct", "«"),
        ("punct", "»"),
        ("punct", ";"),
    ]


@pytest.mark.unit
def test_inline_tags_with_a_handler():
    """An inline tag with a handler is handled where it is between words, and after the word it's inside, so that
    its object starts where it is, as without a tokenizer"""
    xml = document('one <pb n="2"/>two thr<pb n="3"/>ee four')
    tag_exceptions = [r"<pb[^>]*>"]
    with_tokenizer = records(xml, language="en", tag_exceptions=tag_exceptions)
    word_pages = [(name, attrib.get("page", 0)) for kind, name, attrib in with_tokenizer if kind == "word"]
    assert word_pages == [("one", 0), ("two", 1), ("three", 1), ("four", 2)]
    pages = [(kind, name, attrib) for kind, name, attrib in with_tokenizer if kind == "page"]
    assert [attrib["start_byte"] for _, _, attrib in pages] == [xml.index('<pb n="2"/>'), xml.index('<pb n="3"/>')]
    assert pages == [(kind, name, attrib) for kind, name, attrib in records(xml) if kind == "page"]


@pytest.mark.unit
def test_object_and_suppressed_tags_end_runs():
    xml = document("kept<gap>dropped</gap>again <hi>x<note>in note</note>y</hi>")
    assert [word for word, _ in words(xml, language="en")] == ["kept", "again", "x", "in", "note", "y"]


@pytest.mark.unit
def test_word_tags():
    xml = document('<w lemma="be" pos="VERB">is</w> <w lemma="a">a</w> test')
    word_records = [(name, attrib) for kind, name, attrib in records(xml, language="en") if kind == "word"]
    assert [(name, attrib.get("lemma")) for name, attrib in word_records] == [("is", "be"), ("a", "a"), ("test", None)]


@pytest.mark.unit
def test_speaker_with_inline_tags():
    xml = "<text><body><div><sp><speaker>M<hi>r</hi>. Smith.</speaker><p>Hello</p></sp></div></body></text>"
    speakers = [attrib.get("speaker") for kind, name, attrib in records(xml, language="en") if name == "sp"]
    assert speakers == ["Mr. Smith"]


@pytest.mark.unit
def test_bytes_in_a_real_file():
    """Each word and punctuation of an ELTeC novel is what its bytes in the file are, tags left out and normalized"""
    path = sorted((REPO_ROOT / "tests" / "collections" / "ELTeC-eng").glob("*.xml"))[0]
    output = io.StringIO()
    parser = XMLParser(output, 1, path.stat().st_size, metadata_sql_types={}, language="en")
    with open(path, encoding="utf8", newline="") as input_file:
        parser.parse(input_file)
    data = path.read_bytes()
    last_end = 0
    count = 0
    for line in output.getvalue().split("\n")[:-1]:
        kind, name, _, attrib = line.split("\t")
        attrib = json.loads(attrib)
        text = data[attrib["start_byte"] : attrib["end_byte"]].decode("utf8")
        if kind == "word":
            assert attrib["start_byte"] >= last_end
            last_end = attrib["end_byte"]
            text = parser.convert_entities(re.sub(r"<[^>]*>", "", text))
            assert parser.chars_not_to_index.sub("", text).lower().replace("_", "") == name
            count += 1
        elif kind == "punct":
            assert parser.convert_entities(text).replace("\t", " ") == name
    assert count > 10000


@pytest.mark.unit
def test_tokens_for_a_model_without_tokenizer():
    """With a spaCy model but token_regex splitting words, they keep their form in the text, for the tagger, too"""
    xml = document("L'Homme de Paris est là.")
    tokens = [
        (name, attrib.get("spacy_token")) for kind, name, attrib in records(xml, spacy_model="x") if kind == "word"
    ]
    assert tokens == [("l", "L"), ("homme", "Homme"), ("de", None), ("paris", "Paris"), ("est", None), ("là", None)]
    assert all("spacy_token" not in attrib for _, _, attrib in records(xml))  # nothing to tag: as without a model


@pytest.mark.unit
def test_language_without_tokenizer(capsys):
    """The loader splits words with token_regex when spaCy has no tokenizer for the language"""
    assert Loader.spacy_tokenization("not-a-language") is None
    assert "split by token_regex" in capsys.readouterr().err
    assert Loader.spacy_tokenization("en") == "en"
    assert spacy_tokenizer("en") is spacy_tokenizer("en")  # loaded once per process


@pytest.mark.unit
def test_plain_text():
    """Plain text files: words split by the tokenizer, with the exact bytes of their text, and of their punctuation
    (none for what ends a sentence, as without a tokenizer)"""
    from philologic.loadtime.PlainTextParser import PlainTextParser

    text = "L'homme dit-il : « oui ».\n\nAujourd'hui, peut-être !\n"
    output = io.StringIO()
    PlainTextParser(output, 1, len(text.encode("utf8")), language="fr").parse(io.StringIO(text))
    data = text.encode("utf8")
    found = []
    for line in output.getvalue().split("\n")[:-1]:
        kind, name, _, attrib = line.split("\t")
        attrib = json.loads(attrib)
        if kind in ("word", "punct"):
            text_bytes = data[attrib["start_byte"] : attrib["end_byte"]].decode("utf8")
            found.append((kind, name, text_bytes, attrib.get("spacy_token")))
    assert found == [
        ("word", "l", "L", "L'"),
        ("word", "homme", "homme", None),
        ("word", "dit", "dit", None),
        ("word", "il", "il", "-il"),
        ("punct", ":", ":", None),
        ("punct", "«", "«", None),
        ("word", "oui", "oui", None),
        ("word", "aujourd'hui", "Aujourd'hui", "Aujourd'hui"),
        ("punct", ",", ",", None),
        ("word", "peut-être", "peut-être", None),
    ]


@pytest.mark.unit
def test_tagger_gets_tokens(tmp_path):
    """The tagger tags the token of a word, as the tokenizer split it: l' and -il, not the l and il of the index"""
    import spacy
    from philologic.loadtime.LoadFilters import spacy_tagger

    class Tagger:
        """A pipeline which tags nothing, and keeps the words it's given"""

        vocab = spacy.blank("fr").vocab
        words = []

        def pipe(self, documents, as_tuples, batch_size):
            for document, context in documents:
                self.words.extend(token.text for token in document)
                yield document, context

    raw = tmp_path / "raw"
    with open(raw, "w", encoding="utf8") as output:
        xml = document("L'homme dit-il peut-être oui.")
        XMLParser(output, 1, len(xml.encode("utf8")), metadata_sql_types={}, language="fr").parse(io.StringIO(xml))
    loader = type("Loader", (), {"nlp": Tagger()})
    spacy_tagger(loader, {"raw": str(raw)})
    assert [word for word in loader.nlp.words if word.isalpha() or "'" in word or "-" in word] == [
        "L'",
        "homme",
        "dit",
        "-il",
        "peut-être",
        "oui",
    ]
    tagged = [line.split("\t") for line in raw.read_text(encoding="utf8").split("\n")[:-1]]
    assert [(name, json.loads(attrib).get("spacy_token")) for kind, name, _, attrib in tagged if kind == "word"] == [
        ("l", "L'"),
        ("homme", None),
        ("dit", None),
        ("il", "-il"),
        ("peut-être", None),
        ("oui", None),
    ]
    assert all("pos" in json.loads(attrib) for kind, _, _, attrib in tagged if kind == "word")


@pytest.mark.unit
@pytest.mark.parametrize("language", [None, "fr"])
def test_tagger_gets_sentences(tmp_path, language):
    """The tagger tags a sentence at a time: its words and punctuation, with what ends it once (its sent record has it
    too), and nothing for a sentence without punctuation (its sent record is __philo_virtual)"""
    import spacy
    from philologic.loadtime.LoadFilters import spacy_tagger

    class Tagger:
        """A pipeline which tags nothing, and keeps the sentences it's given"""

        vocab = spacy.blank("fr").vocab
        sentences = []

        def pipe(self, documents, as_tuples, batch_size):
            for document, context in documents:
                self.sentences.append([token.text for token in document])
                yield document, context

    raw = tmp_path / "raw"
    xml = "<text><body><div><p>Il vient. « Elle part ! » Oui, dit-il.</p><p>Sans point</p></div></body></text>"
    with open(raw, "w", encoding="utf8") as output:
        XMLParser(output, 1, len(xml.encode("utf8")), metadata_sql_types={}, language=language).parse(io.StringIO(xml))
    parsed = [line.split("\t")[:2] for line in raw.read_text(encoding="utf8").split("\n")[:-1]]
    loader = type("Loader", (), {"nlp": Tagger()})
    spacy_tagger(loader, {"raw": str(raw)})
    sentences = {
        "fr": [["Il", "vient", ". «"], ["Elle", "part", "! »"], ["Oui", ",", "dit", "-il", "."], ["Sans", "point"]],
        None: [["il", "vient", ". «"], ["elle", "part", "! »"], ["oui", ",", "dit", "-", "il", "."], ["sans", "point"]],
    }
    assert loader.nlp.sentences == sentences[language]
    tagged = [line.split("\t") for line in raw.read_text(encoding="utf8").split("\n")[:-1]]
    kept = [record[:2] for record in parsed if record[0] in ("word", "sent", "punct")]
    assert [record[:2] for record in tagged] == kept  # in the same order
    assert all(("pos" in json.loads(attrib)) == (kind != "sent") for kind, _, _, attrib in tagged)
