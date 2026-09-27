"""The parser's output for small documents, each showing some of its behaviors (objects, pages, sentences, notes,
suppressed tags...), compared with the output recorded in parser_snippets.json: a readable record of what the
parser does, to check changes against. After an intended change of the output, record the new one with:
python tests/unit/test_parser_snippets.py --update"""

import io
import json
import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.loadtime.Parser import XMLParser

EXPECTED = Path(__file__).with_name("parser_snippets.json")

SNIPPETS = {
    "divs_and_heads": '<TEI><text><body><div type="chapter"><head>One</head><p>First part.</p><div><head>Sub'
    "</head><p>Deep</p></div></div><div><p>Two</p></div></body></text></TEI>",
    "numbered_divs": "<text><body><div1><head>A</head><div2><head>B</head><div3><p>C</p></div3></div2></div1>"
    "<div1><p>D</p></div1></body></text>",
    "body_without_divs": "<text><body><p>no divs here</p></body></text>",
    "front_and_body": "<text><front><div><head>Preface</head><p>pre</p></div></front><body><div><p>main</p></div>"
    "</body></text>",
    "unclosed_paragraphs": "<text><body><div><p>one<p>two<p>three</div></body></text>",
    "notes": '<text><body><div><p>text<note n="1">a note <p>inside</p></note> more</p></div></body></text>',
    "paragraphs_after_notes": "<text><body><div><p>one<note>n <p>in note</p></note> after</p><p>two</p><p>three</p>"
    "</div></body></text>",
    "speech_and_stage": "<text><body><div><sp><speaker>Hamlet.</speaker><p>To be</p></sp><stage>Exit.</stage>"
    "</div></body></text>",
    "line_groups": "<text><body><div><lg><l>first line</l><l>second line</l></lg></div></body></text>",
    "pages": '<text><body><div><pb n="12 b"/><p>page text</p><pb/><p>more</p><pb n="Ⅳ-2"/></div></body></text>',
    "sentences": "<text><body><div><p>Mr. Smith went home. He left! Did he? Yes, 1.5 times. It is a.b c. End"
    "</p></div></body></text>",
    "tagged_sentences": "<text><body><div><p><s>One two.</s><s>Three</s></p></div></body></text>",
    "suppressed_tags": "<text><body><div><p>kept <gap>dropped</gap> again <gap/> and <gap>open</p></div></body>"
    "</text>",
    "inline_tags_in_words": "<text><body><div><p>wo<hi>r</hi>d and <hi>it</hi>alic, <sup>e</sup>nd</p></div>"
    "</body></text>",
    "word_tags": '<text><body><div><p><w lemma="be" pos="VERB">is</w> <w lemma="a">a</w> test</p></div></body>'
    "</text>",
    "entities_and_hyphens": "<text><body><div><p>caf&eacute; na&shy;<lb/>ture &amp; co&mdash;op</p></div></body>"
    "</text>",
    "apostrophes_and_punctuation": "<text><body><div><p>L'homme d’État « dit » : non ; (oui) [sic]</p></div>"
    "</body></text>",
    "refs_and_graphics": '<text><body><div><p>see <ref target="#n1">note</ref> <graphic url="a.png"/></p></div>'
    "</body></text>",
    "text_outside_body": "<TEI><teiHeader><title>Not indexed</title></teiHeader><text><body><p>indexed</p>"
    "</body></text></TEI>",
}


def parse(xml):
    output = io.StringIO()
    XMLParser(
        output, 1, len(xml.encode("utf8")), known_metadata={"filename": "snippet.xml"}, metadata_sql_types={}
    ).parse(io.StringIO(xml))
    return output.getvalue().splitlines()


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(SNIPPETS))
def test_parser_output(name):
    expected = json.loads(EXPECTED.read_text(encoding="utf8"))
    assert parse(SNIPPETS[name]) == expected[name]


if __name__ == "__main__":
    if "--update" in sys.argv:
        EXPECTED.write_text(
            json.dumps({name: parse(xml) for name, xml in SNIPPETS.items()}, indent=1, ensure_ascii=False) + "\n",
            encoding="utf8",
        )
        print(f"recorded the output of {len(SNIPPETS)} snippets in {EXPECTED}")
