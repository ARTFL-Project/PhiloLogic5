"""Unit tests for query term expansion: regex tokens, .terms files, and the width of hits."""

import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime.Query import get_word_groups, resolve_method, words_per_hit
from philologic.runtime.term_expansion import _expand_positive, _forms_pattern, _is_regex_pattern, _normalize_pattern


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

    def test_plain(self, norm_words):
        with norm_words.begin(buffers=True) as txn:
            assert sorted(_expand_positive("TERM", "ét.s", txn, True, True)) == sorted(["étés", "êtes"])
            assert sorted(_expand_positive("TERM", "ÉT.T.*", txn, True, True)) == sorted(["état", "etat", "ètat", "états"])


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
