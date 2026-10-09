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
    _expand_exclude,
    _expand_positive,
    _forms_pattern,
    _is_regex_pattern,
    _lmdb_expand_term,
    _normalize_pattern,
    cut_terms,
    expand_autocomplete,
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


# norm_word.lmdb of a database loaded with the long s as an s, and of one loaded before, with it as it is
LOADED_WITH_S = {"son": ["son"], "sont": ["sont"]}
LOADED_BEFORE = {"son": ["son", "ſon"], "sont": ["sont", "ſont"]}


def norm_index(tmp_path, forms):
    """A norm_word.lmdb with the forms of each key, closed: its path without the .lmdb"""
    import lmdb

    env = lmdb.open(str(tmp_path / "norm_word.lmdb"), map_size=1 << 20)
    with env.begin(write=True) as txn:
        for key, values in forms.items():
            txn.put(key.encode(), "\x00".join(values).encode())
    env.close()
    return str(tmp_path / "norm_word")


@pytest.fixture
def long_s_txn(tmp_path, request):
    import lmdb

    env = lmdb.open(norm_index(tmp_path, request.param) + ".lmdb", readonly=True, lock=False)
    with env.begin(buffers=True) as txn:
        yield txn
    env.close()


@pytest.mark.unit
class TestLongS:
    """A word, quoted or not, or a pattern, typed with a long s is looked up with an s, as loads index it, and as it
    is, as databases loaded before have it."""

    def test_quoted(self, norm_words):
        with norm_words.begin(buffers=True) as txn:
            assert _expand_positive("QUOTE", '"ſont"', txn, True, True) == ["ſont", "sont"]
            assert _expand_positive("QUOTE", '"sont"', txn, True, True) == ["sont"]

    def test_without_ascii_conversion(self, norm_words):
        with norm_words.begin(buffers=True) as txn:
            assert _expand_positive("TERM", "chriſtiens", txn, False, True) == ["chriſtiens", "christiens"]

    @pytest.mark.parametrize(
        "long_s_txn, forms",
        [(LOADED_WITH_S, ["son", "sont"]), (LOADED_BEFORE, ["ſon", "ſont", "son", "sont"])],
        indirect=["long_s_txn"],
    )
    def test_quoted_pattern(self, long_s_txn, forms):
        """Quoted patterns are matched against the forms as they are: ſ in the pattern matches the s of loads too"""
        assert _expand_positive("QUOTE", '"ſon.*"', long_s_txn, True, True) == forms

    @pytest.mark.parametrize("long_s_txn", [LOADED_BEFORE], indirect=True)
    def test_excluded(self, long_s_txn):
        assert _expand_exclude("QUOTE", '"ſon.*"', long_s_txn, True, True) == {"ſon", "ſont", "son", "sont"}
        assert _expand_exclude("TERM", "ſont", long_s_txn, False, True) == {"ſont", "sont"}

    @pytest.mark.parametrize("long_s_txn", [LOADED_WITH_S], indirect=True)
    def test_lemmas_as_they_are(self, long_s_txn):
        """The parser leaves attributes, lemmas among them, as the source has them"""
        assert _expand_positive("LEMMA", "lemma:ſont", long_s_txn, True, True) == ["lemma:ſont"]

    def test_autocomplete(self, tmp_path):
        frequency_file = norm_index(tmp_path, LOADED_WITH_S)
        assert expand_autocomplete("TERM", "ſon", frequency_file, str(tmp_path), False, True) == ["son", "sont"]
        assert expand_autocomplete("QUOTE", '"ſon"', frequency_file, str(tmp_path), True, True) == ["son", "sont"]


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
