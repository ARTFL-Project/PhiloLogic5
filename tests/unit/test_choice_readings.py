"""Unit tests for the readings of TEI <choice> elements (see philologic.utils.choice): the parser indexes one of those
which are its direct children, at the bytes of the one the formatters show, and the formatters of a database loaded so
(choice_readings in its db.locals.py) show only that one. Those of the databases loaded before show every reading."""

import io
import json
import sqlite3
import sys
import urllib.parse
from pathlib import Path
from types import SimpleNamespace

import pytest
import regex as re

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.loadtime.Parser import XMLParser
from philologic.runtime import WebConfig, WSGIHandler
from philologic.runtime import get_text as get_text_module
from philologic.runtime.DB import DB
from philologic.runtime.get_text import get_concordance_text
from philologic.runtime.ObjectFormatter import format_concordance, format_strip
from philologic.runtime.reports import kwic as kwic_module
from philologic.runtime.reports.kwic import hit_bounds, kwic_hit_object
from philologic.runtime.reports.navigation import generate_text_object
from philologic.utils.choice import choose_readings

from .test_plain_text_parser import new_load  # noqa: F401

TOKEN_REGEX = r"[\p{L}\p{M}\p{N}]+|[&\p{L};]+"
TAGS = re.compile(r"<[^>]+>")


def tei(body):
    return f'<TEI><text><body><div type="act"><p>{body}</p></div></body></text></TEI>\n'


def records(text):
    """The records the parser writes for a TEI text, as (type, name, attributes)"""
    output = io.StringIO()
    parser = XMLParser(output, 1, len(text.encode()), known_metadata={"filename": "t.xml"}, metadata_sql_types={})
    parser.parse(io.StringIO(text))
    for line in output.getvalue().splitlines():
        philo_type, name, _, attributes = line.split("\t")
        yield philo_type, name, json.loads(attributes)


def indexed(body):
    """The words the parser indexes in a paragraph of body, each with the text at its bytes"""
    data = tei(body).encode()
    return [
        (name, data[attributes["start_byte"] : attributes["end_byte"]].decode())
        for philo_type, name, attributes in records(tei(body))
        if philo_type == "word"
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    "names, expected",
    [
        (("orig", "reg"), (0, 1)),
        (("reg", "orig"), (1, 0)),
        (("sic", "corr"), (1, 1)),
        (("corr", "sic"), (0, 0)),
        (("orig", "corr"), (1, 1)),
        (("sic", "reg"), (0, 1)),
        (("abbr", "expan"), (0, 1)),
        (("expan", "abbr"), (1, 0)),
        (("orig",), (0, 0)),
        (("reg",), (0, 0)),
        (("expan",), (0, 0)),
        # A corr shown, but a reg or expan indexed
        (("sic", "corr", "reg"), (1, 2)),
        (("abbr", "corr", "expan"), (1, 2)),
    ],
)
def test_choose_readings(names, expected):
    """The reading shown is the first there is of corr, orig, abbr and sic, else the first; the reading indexed the
    first there is of reg, expan and corr, else the one shown"""
    assert choose_readings(names) == expected


@pytest.mark.unit
class TestIndexedReading:
    @pytest.mark.parametrize(
        "choice, expected",
        [
            ("<orig>ſçait</orig><reg>sait</reg>", [("sait", "ſçait")]),
            ("<reg>faible</reg><orig>foible</orig>", [("faible", "foible")]),
            ("<sic>pût</sic><corr>pu</corr>", [("pu", "pu")]),
            ("<corr>pu</corr><sic>pût</sic>", [("pu", "pu")]),
            ("<orig>prétens</orig><corr>prétends</corr>", [("prétends", "prétends")]),
            ("<sic>défauts</sic><reg>défaut</reg>", [("défaut", "défauts")]),
            ("<abbr>frō</abbr><expan>from</expan>", [("from", "frō")]),
            ("<orig>paroître</orig>", [("paroître", "paroître")]),
            ("<reg>bientôt</reg>", [("bientôt", "bientôt")]),
            ("<expan>from</expan>", [("from", "from")]),
            ("\n  <sic>ti l</sic>\n  <corr>till</corr>\n", [("till", "till")]),
            ("<sic>a</sic><corr>bb</corr><reg>c</reg>", [("c", "bb")]),
        ],
    )
    def test_one_reading_indexed(self, choice, expected):
        """Only one reading is indexed: at its own bytes when it is shown, else at those of the one shown"""
        assert indexed(f"Il <choice>{choice}</choice> fin") == [("il", "Il"), *expected, ("fin", "fin")]

    def test_words_of_a_reading_not_shown_all_at_the_text_shown(self):
        """At the bytes of its text: the shown one's first and last characters other than spaces"""
        assert indexed("Il <choice><orig>bien-tôt</orig><reg>bien tôt</reg></choice>.") == [
            ("il", "Il"), ("bien", "bien-tôt"), ("tôt", "bien-tôt"),
        ]  # fmt: skip
        assert indexed("Il <choice><orig> Régneroit </orig><reg>Règnerait </reg></choice>ſur") == [
            ("il", "Il"), ("règnerait", "Régneroit"), ("sur", "ſur"),
        ]  # fmt: skip

    def test_the_same_bytes_for_each_word(self):
        data = tei("<choice><orig>UN COMEDIEN</orig><reg>UN COMÉDIEN</reg></choice>").encode()
        words = [a for t, _, a in records(data.decode()) if t == "word"]
        orig = data.index(b"UN COMEDIEN")
        assert [(w["start_byte"], w["end_byte"]) for w in words] == [(orig, orig + len(b"UN COMEDIEN"))] * 2

    def test_punctuation_and_entities(self):
        """The "&amp;" shown is no word: "et" is indexed there; the punctuation of a reading not indexed is not"""
        assert indexed("<choice><orig>&amp;</orig><reg>et</reg></choice>c. fin") == [
            ("et", "&amp;"), ("c", "c"), ("fin", "fin"),
        ]  # fmt: skip
        data = tei("A <choice><orig>tems,</orig><reg>temps</reg></choice> B").encode()
        assert [n for t, n, _ in records(data.decode()) if t == "punct"] == []

    def test_readings_with_markup(self):
        assert indexed("<choice><orig>Liſette ta</orig><reg><persName>Lisette</persName> ta</reg></choice> x") == [
            ("lisette", "Liſette ta"), ("ta", "Liſette ta"), ("x", "x"),
        ]  # fmt: skip
        assert indexed('<choice><sic>orĕlŭ</sic><corr><foreign xml:lang="chu">orĭlŭ</foreign></corr></choice>') == [
            ("orĭlŭ", "orĭlŭ"),
        ]  # fmt: skip
        assert indexed('<choice><orig><c type="lettrine">T</c>Ien</orig><reg>Tiens</reg></choice>, x') == [
            ("tiens", 'T</c>Ien'), ("x", "x"),
        ]  # fmt: skip

    def test_readings_outside_a_choice_are_text(self):
        """As any other element: there are hundreds of <sic> alone"""
        assert indexed("Un <sic>pût</sic> et <corr>pu</corr>, <reg>a</reg><orig>b</orig>") == [
            ("un", "Un"), ("pût", "pût"), ("et", "et"), ("pu", "pu"), ("a", "a"), ("b", "b"),
        ]  # fmt: skip
        # Nor are those of a choice inside another element
        assert indexed("<choice><orig>SÇAI<reg>SAIS</reg></orig></choice> QUOI") == [
            ("sçai", "SÇAI"), ("sais", "SAIS"), ("quoi", "QUOI"),
        ]  # fmt: skip

    def test_reading_ending_in_a_tag_moved_out_of_a_word(self):
        """DTA's initials: the parser moves the </hi> and </sic> after "U" and "Unſere" out of the word "UUnſere", but
        the choice is read with them, as the formatters read it; and the choices after it too"""
        sic = '<sic><hi rendition="#in">U</hi>Unſere</sic><corr><hi rendition="#in">U</hi>Nſere</corr>'
        assert indexed(f"<choice>{sic}</choice> <choice><orig>ſçait</orig><reg>sait</reg></choice>") == [
            ("u", "U"), ("nsere", "Nſere"), ("sait", "ſçait"),
        ]  # fmt: skip

    def test_an_unclosed_choice_is_text(self):
        assert indexed("<choice><orig>a</orig><reg>b</reg> c") == [("a", "a"), ("b", "b"), ("c", "c")]


@pytest.mark.unit
class TestNestedChoices:
    def test_choice_in_the_reading_shown(self):
        """The rule again inside the reading shown: the words of the reading indexed instead go to what it shows"""
        assert indexed("<choice><orig><choice><orig>&amp;</orig><reg>et</reg></choice></orig><reg>et</reg></choice> j") == [
            ("et", "&amp;"), ("j", "j"),
        ]  # fmt: skip
        assert indexed("<choice><orig><choice><orig>tems</orig><reg>temps</reg></choice></orig><reg>temps</reg></choice>") == [
            ("temps", "tems"),
        ]  # fmt: skip

    def test_choice_in_an_abbreviation(self):
        """Bodleian's: the "y" and superscript "r" shown, "the" indexed from the first of them to the last"""
        choice = '<choice><abbr>y<choice><orig>d</orig><corr><c rend="superscript">r</c></corr></choice></abbr><expan>the</expan></choice>'
        assert indexed(f"a {choice} b") == [
            ("a", "a"), ("the", 'y<choice><orig>d</orig><corr><c rend="superscript">r'), ("b", "b"),
        ]  # fmt: skip

    def test_choice_in_the_reading_indexed(self):
        """The rule again inside the reading indexed: all of it at the text shown"""
        assert indexed("<choice><orig>foo</orig><reg>a <choice><sic>b</sic><corr>c</corr></choice></reg></choice>") == [
            ("a", "foo"), ("c", "foo"),
        ]  # fmt: skip

    def test_bare_sic_and_corr_in_the_reading_shown(self):
        """Boissy's: both are text in the orig, which is not indexed"""
        assert indexed("<choice><orig>lo<sic>u</sic><corr>n</corr>g-tems</orig><reg>longtemps</reg></choice> x") == [
            ("longtemps", "lo<sic>u</sic><corr>n</corr>g-tems"), ("x", "x"),
        ]  # fmt: skip


@pytest.mark.unit
def test_div_head_has_the_reading_shown():
    text = '<TEI><text><body><div type="act"><head><hi><choice><orig>COMEDIE</orig><reg>COMÉDIE</reg></choice></hi> EN UN ACTE</head><p>x</p></div></body></text></TEI>\n'
    [head] = [attributes["head"] for philo_type, _, attributes in records(text) if philo_type == "div1"]
    assert head == "COMEDIE EN UN ACTE"


@pytest.mark.unit
@pytest.mark.parametrize(
    "config, file_type, expected",
    [
        ("words_to_index = ''\n", "xml", True),
        ("words_to_index = ''\n", "plain_text", False),
        # A parser of its own, as Walt's fork of the XMLParser, indexes every reading
        ("class Fork:\n    pass\nparser_factory = Fork\n", "xml", False),
    ],
)
def test_db_locals_get_choice_readings(new_load, config, file_type, expected):  # noqa: F811
    from philologic.Config import DB_LOCALS_DEFAULTS, DB_LOCALS_HEADER, Config

    loader = new_load(config, file_type)
    loader.write_db_config()
    db_locals_path = Path(loader.destination) / "db.locals.py"
    assert Config(str(db_locals_path), DB_LOCALS_DEFAULTS, DB_LOCALS_HEADER)["choice_readings"] is expected
    written = {}
    exec(db_locals_path.read_text(encoding="utf8"), {}, written)
    assert written["choice_readings"] is expected


@pytest.mark.unit
def test_db_locals_without_choice_readings(tmp_path):
    """The databases loaded before have every reading indexed, and shown"""
    from philologic.Config import DB_LOCALS_DEFAULTS, DB_LOCALS_HEADER, Config

    (tmp_path / "db.locals.py").write_text('metadata_fields = []\ntoken_regex = "\\\\w+"\n', encoding="utf8")
    assert Config(str(tmp_path / "db.locals.py"), DB_LOCALS_DEFAULTS, DB_LOCALS_HEADER)["choice_readings"] is False


HIT = '<span class="highlight">{}</span>'

LINE = (
    "<l>Mais Monſieur <choice><orig>ſçait</orig><reg>sait</reg></choice> que vous aimez à rire, "
    "<choice><sic>pût</sic><corr>pu</corr></choice> <choice><orig>&amp;</orig><reg>et</reg></choice> "
    "<choice><orig>bien-tôt</orig><reg>bien tôt</reg></choice> la fin.</l>"
)


def hits(text, *words):
    data = text.encode()
    return data, [data.index(word.encode()) for word in words]


@pytest.mark.unit
@pytest.mark.parametrize("formatter", [format_concordance, format_strip], ids=["concordance", "kwic"])
class TestConcordanceAndKwic:
    def test_readings_not_shown_left_out(self, formatter):
        data, offsets = hits(LINE, "rire")
        output = TAGS.sub("", formatter(data, TOKEN_REGEX, offsets, choice_readings=True))
        assert output == "Mais Monſieur ſçait que vous aimez à rire, pu &amp; bien-tôt la fin."

    def test_every_reading_shown_without_choice_readings(self, formatter):
        """As before"""
        data, offsets = hits(LINE, "rire")
        expected = "Mais Monſieur ſçaitsait que vous aimez à rire, pûtpu &amp;et bien-tôtbien tôt la fin."
        assert TAGS.sub("", formatter(data, TOKEN_REGEX, offsets)) == expected
        assert TAGS.sub("", formatter(data, TOKEN_REGEX, offsets, choice_readings=False)) == expected

    def test_hit_on_the_reading_indexed_highlights_the_one_shown(self, formatter):
        """All of it, past the end of the word the token regex finds"""
        for shown in ("ſçait", "bien-tôt"):
            data, offsets = hits(LINE, shown)
            assert HIT.format(shown) in formatter(data, TOKEN_REGEX, offsets, choice_readings=True)
        data, offsets = hits(LINE, "pu<")
        assert HIT.format("pu") in formatter(data, TOKEN_REGEX, offsets, choice_readings=True)

    def test_hit_on_a_reading_shown_and_indexed_highlights_its_word(self, formatter):
        """Its words are indexed where they are"""
        data, offsets = hits("<l><choice><sic>Tantpis</sic><corr>Tant pis</corr></choice>, x</l>", "pis</corr")
        output = formatter(data, TOKEN_REGEX, offsets, choice_readings=True)
        assert "Tant " + HIT.format("pis") in output

    def test_hits_at_the_same_bytes(self, formatter):
        """The two words of "bien tôt", indexed at "bien-tôt": one highlight"""
        data, offsets = hits(LINE, "bien-tôt", "bien-tôt")
        output = formatter(data, TOKEN_REGEX, offsets, choice_readings=True)
        assert output.count('<span class="highlight">') == 1
        assert HIT.format("bien-tôt") in output

    def test_markup_of_the_reading_shown_kept(self, formatter):
        data, offsets = hits('<l><choice><orig><c type="lettrine">T</c>Ien</orig><reg>Tiens</reg></choice>, x</l>', "T<")
        output = formatter(data, TOKEN_REGEX, offsets, choice_readings=True)
        assert TAGS.sub("", output) == "TIen, x"
        if formatter is format_concordance:
            assert '<span class="highlight"><span type="lettrine">T</span>Ien</span>' in output

    def test_partial_choices_cut(self, formatter):
        """The text may start or end in a choice, of which it has only some readings: those are cut, as partial words"""
        text = "tôt</orig><reg>bientôt</reg></choice> mot avant le hit et <choice><orig>tems</orig><reg>te"
        data, offsets = hits(text, "hit")
        output = formatter(data, TOKEN_REGEX, offsets, choice_readings=True)
        assert TAGS.sub("", output).strip() == "mot avant le hit et"
        assert "bientôt" in TAGS.sub("", formatter(data, TOKEN_REGEX, offsets))

    def test_partial_choice_with_the_hit_kept(self, formatter):
        text = "pût</sic><corr>pu</corr></choice> et"
        data, offsets = hits(text, "pu<")
        output = formatter(data, TOKEN_REGEX, offsets, choice_readings=True)
        assert HIT.format("pu") in output


def make_database(root, text, choice_readings):
    """A database of a TEI text: a document, a div1 and its paragraphs, with or without choice_readings"""
    data = root / "data"
    (data / "TEXT").mkdir(parents=True)
    (data / "TEXT" / "t.xml").write_bytes(text.encode())
    columns = ["philo_type", "philo_name", "philo_id", "start_byte", "end_byte", "filename", "title", "author"]
    (data / "db.locals.py").write_text(
        'metadata_fields = ["title", "author", "filename"]\n'
        'metadata_types = {"title": "doc", "author": "doc", "filename": "doc"}\n'
        f"metadata_sql_types = {dict.fromkeys(columns, 'text')!r}\n"
        f"token_regex = {TOKEN_REGEX!r}\n" + (f"choice_readings = {choice_readings!r}\n" if choice_readings is not None else "")
    )
    (data / "web_config.cfg").write_text('dbname = "choice"\n')

    def byte(position):
        return len(text[:position].encode())

    rows = [
        ("doc", "TEI", "1 0 0 0 0 0 0", 0, len(text.encode()), "t.xml", "Pièce", "Boissy"),
        ("div1", "div", "1 1 0 0 0 0 0", byte(text.index("<div")), byte(text.index("</body>")), None, None, None),
        ("div2", "__philo_virtual", "1 1 1 0 0 0 0", byte(text.index("<div")), byte(text.index("</body>")), None, None, None),
        ("div3", "__philo_virtual", "1 1 1 1 0 0 0", byte(text.index("<div")), byte(text.index("</body>")), None, None, None),
    ]
    for number, match in enumerate(re.finditer("<p>", text), 1):
        end = text.index("</p>", match.start()) + 4
        rows.append(("para", "p", f"1 1 1 1 {number} 0 0", byte(match.start()), byte(end), None, None, None))
    dbh = sqlite3.connect(data / "toms.db")
    dbh.execute(f"CREATE TABLE toms ({', '.join(c + ' text' for c in columns)})")
    dbh.execute("CREATE INDEX toms_philo_id_index ON toms (philo_id)")
    dbh.executemany(f"INSERT INTO toms VALUES ({', '.join('?' * len(columns))})", [tuple(None if v is None else str(v) for v in r) for r in rows])
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


TEXT = tei(LINE)


def navigate(root, **query):
    config = WebConfig(str(root))
    request = WSGIHandler({"QUERY_STRING": urllib.parse.urlencode(query, doseq=True), "PHILOLOGIC_DBURL": ""}, config)
    return generate_text_object(request, config)["text"]


class Hit(SimpleNamespace):
    def __getitem__(self, key):
        return getattr(self, key)


def hit_on(text, *words):
    data = text.encode()
    return Hit(
        bytes=[data.index(word.encode()) for word in words], doc=SimpleNamespace(filename="t.xml"),
        philo_id=[1, 1, 1, 1, 1, 1, 1], title="Pièce", author="Boissy", filename="t.xml",
    )  # fmt: skip


@pytest.mark.unit
class TestDatabases:
    """The formatters of a database show one reading only if its db.locals.py has choice_readings"""

    def test_text_page(self, tmp_path):
        byte = str(TEXT.encode().index("bien-tôt".encode()))
        shown = navigate(make_database(tmp_path / "new", TEXT, True), philo_id="1 1", byte=[byte, byte])
        assert TAGS.sub("", shown) == "Mais Monſieur ſçait que vous aimez à rire, pu &amp; bien-tôt la fin."
        assert '<span class="xml-orig"><span class="highlight">bien-tôt</span></span>' in shown
        assert shown.count('<span class="highlight">') == 1

    def test_text_page_without_choice_readings(self, tmp_path):
        """As before: every reading shown"""
        expected = "Mais Monſieur ſçaitsait que vous aimez à rire, pûtpu &amp;et bien-tôtbien tôt la fin."
        for name, value in (("old", None), ("false", False)):
            assert TAGS.sub("", navigate(make_database(tmp_path / name, TEXT, value), philo_id="1 1")) == expected

    def test_passage_bounds_in_a_reading_not_shown_kept(self, tmp_path):
        data = TEXT.encode()
        start, end = data.index(b"sait</reg>") + 1, data.index("aimez".encode())
        shown = navigate(make_database(tmp_path, TEXT, True), philo_id="1 1", start_byte=start, end_byte=end)
        assert 'class="passage-marker"' in shown
        assert re.search(r'<span class="passage-0" n="0"> que vous </span>', shown)

    @pytest.mark.parametrize(
        "choice_readings, expected",
        [
            (True, "Mais Monſieur ſçait que vous aimez à rire, pu &amp; bien-tôt la fin."),
            (None, "Mais Monſieur ſçaitsait que vous aimez à rire, pûtpu &amp;et bien-tôtbien tôt la fin."),
        ],
    )
    def test_concordance(self, tmp_path, choice_readings, expected):
        root = make_database(tmp_path, TEXT, choice_readings)
        context = get_concordance_text(DB(str(root / "data")), hit_on(TEXT, "ſçait"), str(root), 300)
        assert TAGS.sub("", context).strip() == expected
        assert '<span class="highlight">ſçait</span>' in context

    @pytest.mark.parametrize("choice_readings, after", [(True, " que vous"), (None, "sait que vous")])
    def test_kwic(self, tmp_path, monkeypatch, choice_readings, after):
        root = make_database(tmp_path, TEXT, choice_readings)
        monkeypatch.setattr(kwic_module, "citation_links", lambda *args, **kwargs: {})
        monkeypatch.setattr(kwic_module, "citations", lambda *args, **kwargs: [])
        result = kwic_hit_object(hit_on(TEXT, "ſçait"), WebConfig(str(root)), DB(str(root / "data")))
        assert '<span class="kwic-highlight"><span class="highlight">ſçait</span></span>' in result["context"]
        assert f'<span class="kwic-after">{after}' in result["context"]

    def test_bare_sic_and_corr_in_the_reading_shown(self, tmp_path):
        """Text of the orig: both shown, with no space between them"""
        text = tei("<choice><orig>lo<sic>u</sic><corr>n</corr>g-tems</orig><reg>longtemps</reg></choice> attendre")
        shown = navigate(make_database(tmp_path, text, True), philo_id="1 1")
        assert TAGS.sub("", shown) == "loung-tems attendre"
