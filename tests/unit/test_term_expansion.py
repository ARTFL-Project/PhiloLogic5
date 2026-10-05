"""Unit tests for query term expansion: regex tokens, .terms files, and the width of hits."""

import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime.Query import get_word_groups, resolve_method, words_per_hit
from philologic.runtime import term_expansion
from philologic.runtime.QuerySyntax import group_terms, parse_query
from philologic.runtime.Query import split_terms
from philologic.runtime.term_expansion import (
    _expand_positive,
    _forms_pattern,
    _is_regex_pattern,
    _lmdb_expand_term,
    _normalize_pattern,
    cut_terms,
)


@pytest.mark.unit
class TestRegexTokens:
    """Tests for telling regex tokens from words, and how they are scanned for."""

    @pytest.mark.parametrize("token", ["sens.*", "couleu?r", "[aeiou]rt", "lov.*:pos:NOUN", "lemma:constitut.*"])
    def test_regex(self, token):
        assert _is_regex_pattern(token)

    @pytest.mark.parametrize("token", ["hamlet", "Art)", "l'art"])
    def test_word(self, token):
        assert not _is_regex_pattern(token)

    @pytest.mark.parametrize("token", ["(Art", "art[", "*nvit*", "italie\\", "du).*"])
    def test_invalid_regex_is_a_word(self, token):
        """A token that does not compile is looked up as a word, rather than failing the search."""
        assert not _is_regex_pattern(token)

    def test_prefix_and_pattern(self):
        assert _normalize_pattern("Sens.*") == (b"sens", "sens.*")

    def test_quantified_character_is_not_in_prefix(self):
        """A quantifier makes the character before it optional: "couleu?r" also matches "couler"."""
        prefix, pattern = _normalize_pattern("couleu?r")
        assert prefix == b"coule"
        assert pattern == "coule(?:u)?r"
        prefix, pattern = _normalize_pattern("ab+c")
        assert prefix == b"ab"

    def test_quantified_character_is_normalized(self):
        prefix, pattern = _normalize_pattern("CŒ?ur")
        assert prefix == b"c"
        assert pattern == "c(?:oe)?ur"

    def test_literal_is_escaped(self):
        """Characters of the literal part that are not metacharacters here still match only themselves."""
        assert _normalize_pattern("x-y.*") == (b"x-y", r"x\-y.*")

    def test_backslash_ends_literal(self):
        assert _normalize_pattern(r"a\.b.*") == (b"a", r"a\.b.*")

    def test_forms_pattern_is_not_normalized(self):
        assert _forms_pattern("lemma:Être.*") == ("lemma:Être".encode("utf-8"), "lemma:Être.*")


@pytest.mark.unit
class TestRegexNormalization:
    """Plain terms are accent- and case-insensitive, regexes too: all their literal characters are normalized as the
    keys of norm_word.lmdb are, not only those before the first metacharacter."""

    @pytest.mark.parametrize(
        "token, expected",
        [
            (".*té", (b"", ".*te")),
            ("pr.mière", (b"pr", "pr.miere")),
            ("Lib.rTé", (b"lib", "lib.rte")),
            ("a[MN]our", (b"a", "a[mn]our")),
            ("[éè]t.*", (b"et", "et.*")),  # a class of one letter once normalized is that letter, to scan from
            ("libert[éèê]s?", (b"liberte", "liberte(?:s)?")),
            ("[œa]uvre", (b"", "(?:[a]|oe)uvre")),  # œ is two letters, which a class can't hold
            ("cœ?ur", (b"c", "c(?:oe)?ur")),
            ("œuvre.*", (b"oeuvre", "oeuvre.*")),
            ("[^é]tat", (b"", "[^e]tat")),
            ("[A-Z]tat", (b"", "[a-z]tat")),
            (r"\w+ité", (b"", r"\w+ite")),  # escapes stay as they are
            (r"\p{L}+É", (b"", r"\p{L}+e")),
            ("x(?:é|È)s", (b"x", "x(?:e|e)s")),
            ("é{2}", (b"", "(?:e){2}")),
        ],
    )
    def test_normalized(self, token, expected):
        assert _normalize_pattern(token) == expected


@pytest.fixture
def norm_words(tmp_path):
    """A norm_word.lmdb: normalized keys, and the forms of each."""
    import lmdb

    env = lmdb.open(str(tmp_path / "norm_word.lmdb"), map_size=1 << 20)
    forms = {"etat": ["état", "etat", "ètat"], "etats": ["états"], "ete": ["été", "ete"], "etes": ["étés", "êtes"]}
    with env.begin(write=True) as txn:
        for key, values in forms.items():
            txn.put(key.encode(), "\x00".join(values).encode())
    yield env
    env.close()


@pytest.mark.unit
class TestQuotedRegex:
    """Quoted terms are accent-sensitive, regexes too: their forms are matched as they are."""

    @pytest.mark.parametrize(
        "token, forms",
        [
            ('"été.*"', ["été", "étés"]),
            ('"ete.*"', ["ete"]),
            ('"[ée]tat"', ["état", "etat"]),
            ('"[éê]t.s"', ["étés", "êtes"]),
        ],
    )
    def test_quoted(self, norm_words, token, forms):
        with norm_words.begin(buffers=True) as txn:
            assert sorted(_expand_positive("QUOTE", token, txn, True, True)) == sorted(forms)

    def test_empty_term(self, norm_words):
        """A term normalized to nothing (an emoji) matches nothing: LMDB fails on an empty key."""
        with norm_words.begin(buffers=True) as txn:
            assert _expand_positive("TERM", "\U0001F600", txn, True, True) == []
            assert _expand_positive("QUOTE", '""', txn, True, True) == []

    def test_unclosed_quote(self, norm_words):
        with norm_words.begin(buffers=True) as txn:
            assert _expand_positive("QUOTE", '"état', txn, True, True) == ["état"]

    def test_plain(self, norm_words):
        with norm_words.begin(buffers=True) as txn:
            assert sorted(_expand_positive("TERM", "ét.s", txn, True, True)) == sorted(["étés", "êtes"])
            assert sorted(_expand_positive("TERM", "ÉT.T.*", txn, True, True)) == sorted(["état", "etat", "ètat", "états"])


@pytest.mark.unit
class TestExpansionCap:
    """A regex with no literal start expands to REGEX_EXPANSION_CAP forms at most, and says when it left some out:
    the results summary tells the user (.*ez found 946,874 hits of 1,430,547, silently)."""

    FORMS = 8  # those of "et.*" in norm_words: état etat ètat états été ete étés êtes

    def test_all_forms(self, norm_words):
        with norm_words.begin(buffers=True) as txn:
            forms = _lmdb_expand_term(txn, b"", "et.*", max_results=self.FORMS)
        assert len(forms) == self.FORMS and not forms.cut

    def test_cut(self, norm_words):
        with norm_words.begin(buffers=True) as txn:
            forms = _lmdb_expand_term(txn, b"", "et.*", max_results=self.FORMS - 1)
        assert len(forms) == self.FORMS - 1 and forms.cut

    @pytest.fixture
    def frequency_file(self, tmp_path):
        """A normalized_word_frequencies.lmdb, closed for cut_terms to open it."""
        import lmdb

        env = lmdb.open(str(tmp_path / "normalized_word_frequencies.lmdb"), map_size=1 << 20)
        with env.begin(write=True) as txn:
            for key, forms in {"amour": ["amour"], "etat": ["état", "etat"], "ete": ["été"], "etes": ["étés"]}.items():
                txn.put(key.encode(), "\x00".join(forms).encode())
        env.close()
        return str(tmp_path / "normalized_word_frequencies")

    @pytest.mark.parametrize(
        "query, cut",
        [
            (".*", [(".*", False)]),
            ("amour NOT .*e.*", [(".*e.*", True)]),
            ('".*t.*"', [('".*t.*"', False)]),
            ("et.*", []),  # a literal start: no cap
            (".*s", []),  # under the cap
            ("amour", []),
        ],
    )
    def test_cut_terms(self, frequency_file, monkeypatch, query, cut):
        monkeypatch.setattr(term_expansion, "REGEX_EXPANSION_CAP", 2)
        split = split_terms(group_terms(parse_query(query)))
        assert cut_terms(split, frequency_file, True, True) == cut


@pytest.mark.unit
class TestWordGroups:
    """Tests for reading the word groups of a .terms file."""

    @pytest.mark.parametrize(
        "content, groups",
        [
            ("a\nb\n", [["a", "b"]]),
            ("a\n\nb\n", [["a"], ["b"]]),
            ("a\n\n\nc\n", [["a"], [], ["c"]]),
            ("\nb\n", [[], ["b"]]),
            ("a\n\n", [["a"], []]),
            ("", [[]]),
        ],
    )
    def test_groups(self, tmp_path, content, groups):
        """Groups that expand to no word are kept, so that the search sees as many as the query has."""
        terms_file = tmp_path / "hitlist.terms"
        terms_file.write_text(content, encoding="utf8")
        assert get_word_groups(str(terms_file)) == groups


@pytest.mark.unit
class TestHitWidth:
    """Tests for the number of words per hit, which readers of a hitlist take its width from."""

    def test_single_term_hits_have_one_word(self):
        split = [(("QUOTE", '"liberté"'),), (("QUOTE", '"amis"'),)]
        assert words_per_hit("single_term", split) == 1
        assert words_per_hit("phrase_ordered", split) == 2

    @pytest.mark.parametrize(
        "q, method",
        [
            ("hamlet", "single_term"),
            ("hamlet | macbeth", "single_term"),
            ("a.* NOT abalone", "single_term"),
            ("my lord", "phrase_unordered"),
            ('"my lord"', "phrase_unordered"),
            ('"républicain""vertu"', "phrase_unordered"),
        ],
    )
    def test_resolve_method_counts_groups(self, q, method):
        """single_term is for queries of one group, whatever their whitespace."""
        assert resolve_method(q, "proxy", "0", "no")[0] == method


@pytest.fixture
def autocomplete_db(tmp_path, monkeypatch):
    """A database's frequency files and hits, the loader's way: words by number of hits, the same number of them from
    the last word ("lit" before "libre"); lemmas the same, from the first lemma ("lemma:libre" before "lemma:lier").
    "la" has its hits in an overflow file. Its tables keep 2 suggestions a prefix, built by build_tables()."""
    import hashlib

    import lmdb

    monkeypatch.setattr(term_expansion, "AUTOCOMPLETE_SUGGESTIONS", 2)
    data = tmp_path / "data"
    frequencies = data / "frequencies"
    frequencies.mkdir(parents=True)
    words = [
        ("le", "le", 50),
        ("la", "la", 40),
        ("le", "lé", 30),
        ("les", "les", 30),
        ("lit", "lit", 5),
        ("libre", "libre", 5),
        ("lac", "lac", 2),
    ]  # (normalized, form, hits)
    lemmas = [("lemma:le", 60), ("lemma:libre", 5), ("lemma:lier", 5), ("lemma:lys", 1)]
    (frequencies / "normalized_word_frequencies").write_text("".join(f"{n}\t{f}\n" for n, f, _ in words), "utf-8")
    (frequencies / "lemmas").write_text("".join(f"{lemma}\n" for lemma, _ in lemmas), "utf-8")

    def write_lmdb(path, items):
        env = lmdb.open(str(path), map_size=1 << 24)
        with env.begin(write=True) as txn:
            for key, value in items:
                txn.put(key.encode("utf-8"), value)
        env.close()

    norm = {}
    for n, f, _ in words:
        norm.setdefault(n, []).append(f)
    write_lmdb(
        frequencies / "normalized_word_frequencies.lmdb", [(n, "\x00".join(f).encode()) for n, f in norm.items()]
    )
    hits = [(f, b"\x00" * 36 * h) for _, f, h in words if f != "la"] + [(k, b"\x00" * 36 * h) for k, h in lemmas]
    write_lmdb(data / "words.lmdb", hits)
    (data / "overflow_words").mkdir()
    (data / "overflow_words" / f"{hashlib.sha256(b'la').hexdigest()}.bin").write_bytes(b"\x00" * 36 * 40)
    forms = [lemma for lemma, _ in lemmas] + ["lemma:libre:pos:ADJ", "lemma:lier:pos:VERB"]
    write_lmdb(frequencies / "word_forms.lmdb", [(key, b"") for key in forms])

    def suggest(kind, token, max_results=10):
        return term_expansion.expand_autocomplete(
            kind, token, str(frequencies / "normalized_word_frequencies"), str(data), True, True, max_results, {"la"}
        )

    def build_tables():
        return term_expansion.build_autocomplete_tables(str(data))

    def table(name):
        env = lmdb.open(str(frequencies / name), readonly=True, lock=False)
        with env.begin() as txn:
            content = {bytes(k).decode(): bytes(v).decode().split("\x00") for k, v in txn.cursor()}
        env.close()
        return content

    return suggest, build_tables, table, frequencies


@pytest.mark.unit
class TestAutocompleteOrder:
    """Autocomplete suggests the most frequent words and lemmas first, as the frequency files order them: from tables
    built at load time for the prefixes more start with than it suggests, ordering the few of any other prefix."""

    def test_tables(self, autocomplete_db):
        _, build_tables, table, _ = autocomplete_db
        assert build_tables() == 4
        assert table("autocomplete_words.lmdb") == {"l": ["le", "la"], "le": ["le", "lé"]}
        # from "lemma:" on, what a LEMMA term is: "lemma:" alone suggests the most frequent lemmas
        assert table("autocomplete_lemmas.lmdb") == {
            "lemma:": ["lemma:le", "lemma:libre"],
            "lemma:l": ["lemma:le", "lemma:libre"],
        }

    def test_from_the_table(self, autocomplete_db):
        suggest, build_tables, _, _ = autocomplete_db
        build_tables()
        assert suggest("TERM", "l") == ["le", "la"]
        assert suggest("TERM", "Lé") == ["le", "lé"]  # normalized, as the words are
        assert suggest("QUOTE", '"le') == ["le", "lé"]
        assert suggest("TERM", "l", max_results=1) == ["le"]
        assert suggest("LEMMA", "lemma:l") == ["lemma:le", "lemma:libre"]

    def test_few_ordered(self, autocomplete_db):
        """A prefix the table keeps nothing for has few enough words to order them: by hits, overflow ones too, the
        same number of them from the last word, as in the frequency file."""
        suggest, build_tables, _, _ = autocomplete_db
        build_tables()
        assert suggest("TERM", "li") == ["lit", "libre"]
        assert suggest("TERM", "la") == ["la", "lac"]  # "la" in its overflow file
        assert suggest("TERM", "x") == []

    def test_few_lemmas_ordered(self, autocomplete_db):
        """The lemmas only, not their attributes, the same number of hits from the first lemma, as in the lemmas file"""
        suggest, build_tables, _, _ = autocomplete_db
        build_tables()
        assert suggest("LEMMA", "lemma:li") == ["lemma:libre", "lemma:lier"]

    def test_regex(self, autocomplete_db):
        suggest, build_tables, _, _ = autocomplete_db
        build_tables()
        assert suggest("TERM", "l.*", max_results=4) == ["le", "la", "lé", "les"]
        assert suggest("LEMMA", "lemma:li.*") == ["lemma:libre", "lemma:lier"]

    def test_without_tables(self, autocomplete_db):
        """A database loaded before the tables suggests as before: the first words in alphabetical order."""
        suggest, _, _, _ = autocomplete_db
        assert suggest("TERM", "l", max_results=3) == ["la", "lac", "le"]
        assert suggest("LEMMA", "lemma:li") == [
            "lemma:libre",
            "lemma:libre:pos:ADJ",
            "lemma:lier",
            "lemma:lier:pos:VERB",
        ]

    def test_rebuilt(self, autocomplete_db):
        """Built again, as for a database loaded before them, a table is replaced whole."""
        _, build_tables, table, frequencies = autocomplete_db
        build_tables()
        (frequencies / "normalized_word_frequencies").write_text("la\tla\nle\tle\nles\tles\n", "utf-8")
        build_tables()
        assert table("autocomplete_words.lmdb") == {"l": ["la", "le"]}
        assert sorted(p.name for p in frequencies.iterdir() if "autocomplete" in p.name) == [
            "autocomplete_lemmas.lmdb",
            "autocomplete_words.lmdb",
        ]

    def test_prefixes_of_whole_characters(self, tmp_path, monkeypatch):
        """Prefixes are of bytes, but only those of whole characters can be asked for."""
        monkeypatch.setattr(term_expansion, "AUTOCOMPLETE_SUGGESTIONS", 2)
        entries = [(word.encode(), word.encode()) for word in ("étape", "étoile", "été")]
        path = str(tmp_path / "table.lmdb")
        assert term_expansion._write_prefix_table(iter(entries), path, 1) == 2  # "é" and "ét", not half an "é"
