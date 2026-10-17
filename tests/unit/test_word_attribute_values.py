"""The word attributes and their values the web config lists, from the word_attributes frequency file"""

import pytest

from philologic.loadtime.Loader import word_attribute_values


@pytest.mark.unit
def test_words_and_values_with_colons(tmp_path):
    """Words split by spaCy's tokenizer (3:16) and values (the Penn Treebank tag ":") can have colons"""
    frequency_file = tmp_path / "word_attributes"
    frequency_file.write_text(
        "the:pos:DET\nthe:tag:DT\n13:12b:pos:NUM\na:tag::\n3:16:tag:CD\nword:pos:\n", encoding="utf8"
    )
    assert word_attribute_values(str(frequency_file), {"pos", "tag", "lemma"}) == {
        "pos": {"DET", "NUM", ""},
        "tag": {"DT", ":", "CD"},
    }
    assert list(word_attribute_values(str(frequency_file), {"pos", "tag"})) == ["pos", "tag"]  # in the file's order


@pytest.mark.unit
def test_lines_without_a_known_attribute_are_left_out(tmp_path):
    frequency_file = tmp_path / "word_attributes"
    frequency_file.write_text("the:pos:DET\nthe:type:square\n", encoding="utf8")
    assert word_attribute_values(str(frequency_file), {"pos"}) == {"pos": {"DET"}}
