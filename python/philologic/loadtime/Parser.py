#!/var/lib/philologic5/philologic_env/bin/python3
"""PhiloLogic5 main parser"""

import html
import os
import string
import sys
from bisect import bisect_right

import regex as re

from philologic.loadtime.OHCOVector import CompoundStack
from philologic.loadtime.spacy_tokenizer import TextBytes, spacy_tokenizer, split_words, tokenizer_input
from philologic.utils import convert_entities, extract_full_date, extract_integer

DEFAULT_TAG_TO_OBJ_MAP = {
    "div": "div",
    "div1": "div",
    "div2": "div",
    "div3": "div",
    "hyperdiv": "div",
    "front": "div",
    "note": "para",
    "p": "para",
    "sp": "para",
    "lg": "para",
    "epigraph": "para",
    "argument": "para",
    "postscript": "para",
    "opener": "para",
    "closer": "para",
    "stage": "para",
    "castlist": "para",
    "list": "para",
    "q": "para",
    "add": "para",
    "pb": "page",
    "ref": "ref",
    "graphic": "graphic",
    "l": "line",
    "ab": "line",
}

DEFAULT_METADATA_TO_PARSE = {
    "div": ["head", "type", "n", "id", "vol", "div_date"],
    "para": ["who", "speaker", "resp", "id"],  # for <sp> and <add> tags
    "page": ["n", "id", "facs"],
    "ref": ["target", "n", "type"],
    "graphic": ["facs"],
    "line": ["n", "id"],
}

DEFAULT_DOC_XPATHS = {
    "author": [
        ".//sourceDesc/bibl/author[@type='marc100']",
        ".//sourceDesc/bibl/author[@type='artfl']",
        ".//sourceDesc/bibl/author",
        ".//titleStmt/author",
        ".//sourceDesc/biblStruct/monogr/author/name",
        ".//sourceDesc/biblFull/titleStmt/author",
        ".//sourceDesc/biblFull/titleStmt/respStmt/name",
        ".//sourceDesc/biblFull/titleStmt/author",
        ".//sourceDesc/bibl/titleStmt/author",
    ],
    "title": [
        ".//sourceDesc/bibl/title[@type='marc245']",
        ".//sourceDesc/bibl/title[@type='artfl']",
        ".//sourceDesc/bibl/title",
        ".//titleStmt/title",
        ".//sourceDesc/bibl/titleStmt/title",
        ".//sourceDesc/biblStruct/monogr/title",
        ".//sourceDesc/biblFull/titleStmt/title",
    ],
    "author_dates": [".//sourceDesc/bibl/author/date", ".//titlestmt/author/date"],
    "create_date": [
        ".//profileDesc/creation/date",
        ".//fileDesc/sourceDesc/bibl/imprint/date",
        ".//sourceDesc/biblFull/publicationStmt/date",
        ".//profileDesc/dummy/creation/date",
        ".//fileDesc/sourceDesc/bibl/creation/date",
    ],
    "publisher": [
        ".//sourceDesc/bibl/imprint[@type='artfl']",
        ".//sourceDesc/bibl/imprint[@type='marc534']",
        ".//sourceDesc/bibl/imprint/publisher",
        ".//sourceDesc/biblStruct/monogr/imprint/publisher/name",
        ".//sourceDesc/biblFull/publicationStmt/publisher",
        ".//sourceDesc/bibl/publicationStmt/publisher",
        ".//sourceDesc/bibl/publisher",
        ".//publicationStmt/publisher",
        ".//publicationStmp",
    ],
    "pub_place": [
        ".//sourceDesc/bibl/imprint/pubPlace",
        ".//sourceDesc/biblFull/publicationStmt/pubPlace",
        ".//sourceDesc/biblStruct/monog/imprint/pubPlace",
        ".//sourceDesc/bibl/pubPlace",
        ".//sourceDesc/bibl/publicationStmt/pubPlace",
    ],
    "pub_date": [
        ".//sourceDesc/bibl/imprint/date",
        ".//sourceDesc/biblStruct/monog/imprint/date",
        ".//sourceDesc/biblFull/publicationStmt/date",
        ".//sourceDesc/bibFull/imprint/date",
        ".//sourceDesc/bibl/date",
        ".//text/front/docImprint/acheveImprime",
    ],
    "extent": [".//sourceDesc/bibl/extent", ".//sourceDesc/biblStruct/monog//extent", ".//sourceDesc/biblFull/extent"],
    "editor": [
        ".//sourceDesc/bibl/editor",
        ".//sourceDesc/biblFull/titleStmt/editor",
        ".//sourceDesc/bibl/title/Stmt/editor",
    ],
    "text_genre": [".//profileDesc/textClass/keywords[@scheme='genre']/term", ".//SourceDesc/genre"],
    "keywords": [".//profileDesc/textClass/keywords/list/item"],
    "language": [".//profileDesc/language/language"],
    "notes": [".//fileDesc/notesStmt/note", ".//publicationStmt/notesStmt/note"],
    "auth_gender": [".//publicationStmt/notesStmt/note"],
    "collection": [".//seriesStmt/title"],
    "period": [
        ".//profileDesc/textClass/keywords[@scheme='period']/list/item",
        ".//SourceDesc/period",
        ".//sourceDesc/period",
    ],
    "text_form": [".//profileDesc/textClass/keywords[@scheme='form']/term"],
    "structure": [".//SourceDesc/structure", ".//sourceDesc/structure"],
    "idno": [".//publicationStmt/idno/", ".//seriesStmt/idno/"],
}

TAG_EXCEPTIONS = [
    r"<hi[^>]*>",
    r"<emph[^>]*>",
    r"<\/hi>",
    r"<\/emph>",
    r"<orig[^>]*>",
    r"<\/orig>",
    r"<sic[^>]*>",
    r"<\/sic>",
    r"<abbr[^>]*>",
    r"<\/abbr>",
    r"<i>",
    r"</i>",
    r"<sup>",
    r"</sup>",
    r"<sub(?:\s[^>]*)?>",
    r"<\/sub>",
    r"<sc(?:\s[^>]*)?>",
    r"<\/sc>",
    r"<b(?:\s[^>]*)?>",
    r"<\/b>",
    r"<seg(?:\s[^>]*)?>",  # as the decorated initials of EEBO: <seg rend="decorInit">W</seg>E
    r"<\/seg>",
    r"<supplied(?:\s[^>]*)?>",
    r"<\/supplied>",
    r"<pn(?:\s[^>]*)?>",  # proper names, as frantext marks them: <pn>Saint</pn>-<pn>Germain</pn>
    r"<\/pn>",
    r"<g(?:\s[^>]*)?>",  # glyphs, as EEBO's end-of-line hyphens: ex<g ref="char:EOLhyphen"/>ecution
    r"<\/g>",
    r"<lb\s[^>]*(?:break=[\"']no|hyphenInWord|rend=[\"']hyphen)[^>]*>",  # line breaks inside words, not the others
    r"<pc\s[^>]*force=[\"']weak[^>]*>",  # punctuation which doesn't split words (<pc type="hyphen" force="weak"/>)
]

# For Asian script, try using this slightly-modified regex: r"[\p{L}\p{M}\p{N}\p{Po}]+|[&\p{L};]+"
TOKEN_REGEX = r"[\p{L}\p{M}\p{N}]+|[&\p{L};]+"

PUNCTUATION = r"\p{P}+"

# Tags whose content is never indexed. These are added to the suppress_tags defined in load configs.
DEFAULT_SUPPRESS_TAGS = {"gap"}

# Tags which open or close para and div level objects (or their containers). Used to detect when we
# leave the object containing an unclosed suppressed tag.
OBJECT_TAGS = {
    "p",
    "note",
    "epigraph",
    "list",
    "sp",
    "argument",
    "opener",
    "closer",
    "stage",
    "castlist",
    "add",
    "lg",
    "div",
    "div1",
    "div2",
    "div3",
    "hyperdiv",
    "front",
    "body",
    "back",
    "text",
}

CHARS_NOT_TO_INDEX = r"[\[\{\]\}]"

# Pre-compiled regexes used for parsing
join_hyphen_with_lb = re.compile(r"(\&shy;[\n \t]*<lb\/>)", re.I | re.M)
join_hyphen = re.compile(r"(\&shy;[\n \t]*)", re.I | re.M)
text_tag = re.compile(r"<text\W", re.I)
doc_body_tag = re.compile(r"<docbody", re.I)
body_tag = re.compile(r"<body\W", re.I)
div_tag = re.compile(r"<div", re.I)
closed_div_tag = re.compile(r"<\/div", re.I)
page_tag = re.compile(r"<pb\W", re.I)
attrib_matcher = re.compile(r"""(\S+)="?((?:.(?!"?\s+(?:\S+)=|[>"]))+.)"?""", re.I)
tag_matcher = re.compile(r"<(\/?\w+)[^>]*>?", re.I)
head_self_close_tag = re.compile(r"<head\/>", re.I)
head_tag = re.compile(r"<head", re.I)
closed_head_tag = re.compile(r"<\/head>", re.I)
apost_ent = re.compile(r"\&apos;", re.I)
macr_ent = re.compile(r"\&([A-Za-z])macr;", re.I)
inverted_ent = re.compile(r"\&inverted([a-zA-Z0-9]);", re.I)
supp_ent = re.compile(r"&supp([a-z0-9]);", re.I)
ligatures_ent = re.compile(r"\&([A-Za-z][A-Za-z])lig;", re.I)
type_attrib = re.compile(r'type="([^"]*)"', re.I)
char_ents = re.compile(r"\&[a-zA-Z0-9\#][a-zA-Z0-9]*;", re.I)
check_if_char_word = re.compile(r"\p{L}", re.I)
cap_char_or_num = re.compile(r"[A-Z0-9]")  # Capitals
ending_punctuation = re.compile(r"[%s]$" % string.punctuation.replace(")", "").replace("]", ""))
abbreviation_tag = re.compile(r"""<abbr\b[^>]*?\bexpan=(["'])(.*?)\1[^>]*>(.*?)</abbr>""", re.I | re.S)
any_tag = re.compile(r"<[^>]*>")
semi_colon_strip = re.compile(r"\A;?(\w+);?\Z")
# What the text the tokenizer splits leaves out or changes (see XMLParser.tokenizer_text): entities, and the "_"
# of cleanup_content
tokenizer_special_chars = re.compile(rf"{char_ents.pattern}|_+", re.I)
UNKNOWN_ENTITY_CHARACTER = chr(0xE000)  # a private use character, which spaCy's tokenizers keep in words

## Build a list of control characters to remove
## http://stackoverflow.com/questions/92438/stripping-non-printable-characters-from-a-string-in-python/93029#93029
tab_newline = re.compile(r"[\n|\t]")
control_chars = "".join(
    [i for i in [chr(x) for x in list(range(0, 32)) + list(range(127, 160))] if not tab_newline.search(i)]
)
control_char_re = re.compile(r"[%s]" % re.escape(control_chars))

# Entities regexes
entity_regex = [
    re.compile(r"(\&space;)", re.I),
    re.compile(r"(\&mdash;)", re.I),
    re.compile(r"(\&nbsp;)", re.I),
    re.compile(r"(\&para;)", re.I),
    re.compile(r"(\&sect;)", re.I),
    re.compile(r"(\&ast;)", re.I),
    re.compile(r"(\&commat;)", re.I),
    re.compile(r"(\&ldquo;)", re.I),
    re.compile(r"(\&laquo;)", re.I),
    re.compile(r"(\&rdquo;)", re.I),
    re.compile(r"(\&raquo;)", re.I),
    re.compile(r"(\&lsquo;)", re.I),
    re.compile(r"(\&rsquo;)", re.I),
    re.compile(r"(\&quot;)", re.I),
    re.compile(r"(\&sup[0-9]*;)", re.I),
    re.compile(r"(\&mdash;)", re.I),
    re.compile(r"(\&amp;)", re.I),
    re.compile(r"(\&deg;)", re.I),
    re.compile(r"(\&ndash;)", re.I),
    re.compile(r"(\&copy;)", re.I),
    re.compile(r"(\&gt;)", re.I),
    re.compile(r"(\&lt;)", re.I),
    re.compile(r"(\&frac[0-9]*;)", re.I),
    re.compile(r"(\&pound;)", re.I),
    re.compile(r"(\&colon;)", re.I),
    re.compile(r"(\&hyphen;)", re.I),
    re.compile(r"(\&dash;)", re.I),
    re.compile(r"(\&excl;)", re.I),
    re.compile(r"(\&dagger;)", re.I),
    re.compile(r"(\&ddagger;)", re.I),
    re.compile(r"(\&times;)", re.I),
    re.compile(r"(\&blank;)", re.I),
    re.compile(r"(\&dollar;)", re.I),
    re.compile(r"(\&cent;)", re.I),
    re.compile(r"(\&verbar;)", re.I),
    re.compile(r"(\&quest;)", re.I),
    re.compile(r"(\&hellip;)", re.I),
    re.compile(r"(\&percnt;)", re.I),
    re.compile(r"(\&middot;)", re.I),
    re.compile(r"(\&plusmn;)", re.I),
    re.compile(r"(\&sqrt;)", re.I),
    re.compile(r"(\&sol;)", re.I),
    re.compile(r"(\&sdash;)", re.I),
    re.compile(r"(\&equals;)", re.I),
    re.compile(r"(\&ornament;)", re.I),
    re.compile(r"(\&rule;)", re.I),
    re.compile(r"(\&prime;)", re.I),
    re.compile(r"(\&rsqb;)", re.I),
    re.compile(r"(\&lsqb;)", re.I),
    re.compile(r"(\&punc;)", re.I),
    re.compile(r"(\&cross;)", re.I),
    re.compile(r"(\&diamond;)", re.I),
    re.compile(r"(\&lpunctel;)", re.I),
    re.compile(r"(\&lsemicol;)", re.I),
    re.compile(r"(\&plus;)", re.I),
    re.compile(r"(\&minus;)", re.I),
    re.compile(r"(\&ounce;)", re.I),
    re.compile(r"(\&rindx;)", re.I),
    re.compile(r"(\&lindx;)", re.I),
    re.compile(r"(\&leaf;)", re.I),
    re.compile(r"(\&radic;)", re.I),
    re.compile(r"(\&dram;)", re.I),
    re.compile(r"(\&sun;)", re.I),
]


# Handlers of XMLParser.tag_handler: tag name (lowercased; "div" and "/div" stand for all the names starting with them)
# -> (regex which the tag must match, handler method). The regexes match tags starting with different names: a tag has
# one handler at most.
TAG_HANDLERS = {
    "p": (re.compile(r"<p[ >]", re.I), "handle_paragraph"),
    "note": (re.compile(r"<note\W", re.I), "handle_note"),
    "/note": (re.compile(r"</note\W", re.I), "handle_note_end"),
    "castlist": (re.compile(r"<castlist\W", re.I), "handle_paragraph_object"),
    "/castlist": (re.compile(r"</castlist\W", re.I), "handle_paragraph_object_end"),
    "sp": (re.compile(r"<sp\W", re.I), "handle_blocking_paragraph_object"),
    "/sp": (re.compile(r"</sp\W", re.I), "handle_blocking_paragraph_object_end"),
    "epigraph": (re.compile(r"<epigraph\W", re.I), "handle_blocking_paragraph_object"),
    "/epigraph": (re.compile(r"</epigraph\W", re.I), "handle_blocking_paragraph_object_end"),
    "argument": (re.compile(r"<argument\W", re.I), "handle_blocking_paragraph_object"),
    "/argument": (re.compile(r"</argument\W", re.I), "handle_blocking_paragraph_object_end"),
    "opener": (re.compile(r"<opener\W", re.I), "handle_blocking_paragraph_object"),
    "/opener": (re.compile(r"</opener\W", re.I), "handle_blocking_paragraph_object_end"),
    "closer": (re.compile(r"<closer\W", re.I), "handle_blocking_paragraph_object"),
    "/closer": (re.compile(r"</closer\W", re.I), "handle_blocking_paragraph_object_end"),
    "add": (re.compile(r"<add\W", re.I), "handle_blocking_paragraph_object"),
    "/add": (re.compile(r"</add\W", re.I), "handle_blocking_paragraph_object_end"),
    "list": (re.compile(r"<list\W", re.I), "handle_inner_paragraph_object"),
    "/list": (re.compile(r"</list\W", re.I), "handle_inner_paragraph_object_end"),
    "stage": (re.compile(r"<stage\W", re.I), "handle_inner_paragraph_object"),
    "/stage": (re.compile(r"</stage\W", re.I), "handle_inner_paragraph_object_end"),
    "speaker": (re.compile(r"<speaker\W", re.I), "handle_speaker"),
    "pb": (page_tag, "handle_page"),
    "lg": (re.compile(r"<lg\W", re.I), "handle_line_group"),
    "/lg": (re.compile(r"</lg\W", re.I), "handle_line_group_end"),
    "l": (re.compile(r"<l\W", re.I), "handle_line"),
    "/l": (re.compile(r"</l\W", re.I), "handle_line_end"),
    "ab": (re.compile(r"<ab\W", re.I), "handle_ab"),
    "/ab": (re.compile(r"</ab\W", re.I), "handle_ab_end"),
    "s": (re.compile(r"<s\W", re.I), "handle_sentence"),
    "/s": (re.compile(r"</s\W", re.I), "handle_sentence_end"),
    "front": (re.compile(r"<front\W", re.I), "handle_front"),
    "/front": (re.compile(r"</front\W", re.I), "handle_front_end"),
    "body": (body_tag, "handle_body"),
    "hyperdiv": (re.compile(r"<hyperdiv\W", re.I), "handle_hyperdiv"),
    "div": (div_tag, "handle_div"),
    "/div": (closed_div_tag, "handle_div_end"),
    "index": (re.compile(r"<index(?!\w)"), "handle_index"),
    "date": (re.compile(r"<date(?!\w)"), "handle_date"),
    "ref": (re.compile(r"<ref(?!\w)"), "handle_ref"),
    "graphic": (re.compile(r"<graphic(?!\w)"), "handle_graphic"),
}

# Characters matched by the ending_punctuation character class (all ASCII), used to strip it without a regex call
ENDING_PUNCTUATION_CHARS = frozenset(chr(i) for i in range(128) if ending_punctuation.match(chr(i)))

TOKEN_CHAR = re.compile(r"[\p{L}\p{M}\p{N}&;]", re.I)  # a character of a TOKEN_REGEX token
WORD_CHAR = re.compile(r"[\p{L}\p{M}\p{N}]")  # one of its characters other than those of entities


def byte_positions(text):
    """A function giving the byte (in UTF-8) of character positions of text, asked for in increasing order"""
    if text.isascii():
        return lambda position: position
    last = [0, 0]  # the position last asked for, and its byte

    def byte(position):
        last[1] += len(text[last[0] : position].encode("utf8"))
        last[0] = position
        return last[1]

    return byte


def replace_tag_exceptions(content, tag_exceptions, tag_exception_starts, replacement):
    """Same as tag_exceptions.sub(replacement, content), faster. Matches start with a TOKEN_REGEX token, then a tag
    found by tag_exception_starts (which the token ends before): only the token before such a tag, from its start
    (and after the last match), can start one."""
    output, pos = [], 0
    tag_start = tag_exception_starts.search(content)
    while tag_start:
        tag_position = tag_start.start()
        token_start = tag_position - 1  # the token's last character
        if token_start >= pos:
            while token_start > pos and TOKEN_CHAR.match(content, token_start - 1):
                token_start -= 1
            for match_start in range(token_start, tag_position):
                match = tag_exceptions.match(content, match_start)
                if match:
                    output.append(content[pos:match_start])
                    output.append(replacement(match))
                    pos = match.end()
                    break
        tag_start = tag_exception_starts.search(content, max(tag_position + 1, pos))
    output.append(content[pos:])
    return "".join(output)


class XMLParser:
    """Parses clean or dirty XML.
    This is a port of the PhiloLogic3 parser originally written by Mark Olsen in Perl.
    """

    def __init__(
        self,
        output,
        docid,
        filesize,
        words_to_index=None,
        known_metadata={},
        tag_to_obj_map=DEFAULT_TAG_TO_OBJ_MAP,
        metadata_to_parse=DEFAULT_METADATA_TO_PARSE,
        lowercase_index=True,
        lemmas=None,
        **parse_options,
    ):
        """Initialize class"""
        self.types = ["doc", "div1", "div2", "div3", "para", "sent", "word"]
        self.parallel_type = "page"
        self.docid = docid
        self.tag_to_obj_map = tag_to_obj_map
        self.metadata_to_parse = {}
        for obj in metadata_to_parse:
            self.metadata_to_parse[obj] = set(metadata_to_parse[obj])

        # Initialize an OHCOVector Stack. operations on this stack produce all parser output.
        self.v = CompoundStack(
            self.types,
            self.parallel_type,
            docid=docid,
            out=output,
            ref="ref",
            line="line",
            graphic="graphic",
            punctuation="punct",
        )

        self.filesize = filesize
        self.known_metadata = known_metadata

        if words_to_index:
            self.words_to_index = words_to_index
            self.defined_words_to_index = True
        else:
            self.defined_words_to_index = False

        # List of global variables used for the tag handler
        if "token_regex" in parse_options:
            self.token_regex = re.compile(r"(%s)" % parse_options["token_regex"], re.I)
        else:
            self.token_regex = re.compile(r"(%s)" % TOKEN_REGEX, re.I)

        if "sentence_breakers" in parse_options:
            self.sentence_breakers = parse_options["sentence_breakers"]
        else:
            self.sentence_breakers = []

        if "punctuation" in parse_options:
            self.punct_regex = re.compile(rf"""{parse_options["punctuation"]}""")
        else:
            self.punct_regex = re.compile(rf"{PUNCTUATION}")

        # Tags in DEFAULT_SUPPRESS_TAGS are always suppressed: load configs can add tags, but not remove these.
        self.suppress_tags = DEFAULT_SUPPRESS_TAGS | {i for i in parse_options.get("suppress_tags", []) if i}
        self.in_suppressed_tag = False
        self.suppressed_tag_stack = []  # suppressed tags and object tags opened inside a suppressed tag

        if "break_apost" in parse_options:
            self.break_apost = parse_options["break_apost"]
        else:
            self.break_apost = True

        if "chars_not_to_index" in parse_options:
            self.chars_not_to_index = re.compile(r"%s" % parse_options["chars_not_to_index"], re.I)
        else:
            self.chars_not_to_index = re.compile(r"%s" % CHARS_NOT_TO_INDEX, re.I)

        if "break_sent_in_line_group" in parse_options:
            self.break_sent_in_line_group = parse_options["break_sent_in_line_group"]
        else:
            self.break_sent_in_line_group = False

        if "tag_exceptions" in parse_options:
            tag_exceptions = parse_options["tag_exceptions"]
        else:
            tag_exceptions = TAG_EXCEPTIONS

        tags_start_with_bracket = all(tag.startswith("<") for tag in tag_exceptions)
        tag_exceptions = "|".join(tag_exceptions)
        tag_token_regex = parse_options.get("token_regex", TOKEN_REGEX)
        self.tag_exceptions = self.tag_runs = None  # none: every match would be replaced by itself
        if tag_exceptions:
            # Words with tag exceptions inside them: tokens joined by runs of these tags, and the tag right after them
            # (see cleanup_content)
            self.tag_exceptions = re.compile(
                rf"(?:{tag_token_regex})(?:(?:{tag_exceptions})+(?:{tag_token_regex}))+(?:{tag_exceptions})?",
                re.I | re.M,
            )
            self.tag_runs = re.compile(rf"(?:{tag_exceptions})+", re.I | re.M)
        # With TOKEN_REGEX and tags starting with "<", matches start in the token before one of the tags: finding these
        # tags first saves trying a match everywhere (see replace_tag_exceptions)
        self.tag_exception_starts = None
        if tag_token_regex == TOKEN_REGEX and tags_start_with_bracket and tag_exceptions:
            self.tag_exception_starts = re.compile(rf"(?<={TOKEN_CHAR.pattern})(?:{tag_exceptions})", re.I | re.M)

        # With a language, words are split by spaCy's tokenizer for it rather than by token_regex (the loader only
        # gives a language spaCy has a tokenizer for). Tag exceptions are then inline tags, which the text it splits
        # leaves out, rather than tags cleanup_content takes out of words (see flush_run).
        # With a spaCy model, which tags the words as they are in the text (see LoadFilters.spacy_tagger): words keep
        # their form, before index_form makes it the indexed one, in their spacy_token attribute
        self.spacy_model = parse_options.get("spacy_model")
        self.tokenizer = None
        self.inline_tags = None
        self.run = []  # with a tokenizer, the text lines and inline tags not indexed yet (see flush_run)
        if parse_options.get("language"):
            self.tokenizer = spacy_tokenizer(parse_options["language"], parse_options.get("spacy_model"))
            if tag_exceptions:
                self.inline_tags = re.compile(rf"(?:{tag_exceptions})", re.I)
            self.tag_exceptions = self.tag_exception_starts = None

        if "join_hyphen_in_words" in parse_options:
            self.join_hyphen_in_words = parse_options["join_hyphen_in_words"]
        else:
            self.join_hyphen_in_words = True

        if "abbrev_expand" in parse_options:
            self.abbrev_expand = parse_options["abbrev_expand"]
        else:
            self.abbrev_expand = True

        if "long_word_limit" in parse_options:
            self.long_word_limit = parse_options["long_word_limit"]
        else:
            self.long_word_limit = 200

        if "flatten_ligatures" in parse_options:
            self.flatten_ligatures = parse_options["flatten_ligatures"]
        else:
            self.flatten_ligatures = True

        self.lemmas = lemmas

        self.metadata_sql_types = parse_options["metadata_sql_types"]

        self.lowercase_index = lowercase_index
        self.in_the_text = False
        self.in_front_matter = False
        self.in_a_note = False
        self.no_deeper_objects = False
        self.in_line_group = False
        self.open_div1 = False
        self.open_div2 = False
        self.open_div3 = False
        self.open_para = False
        self.open_sent = False
        self.open_page = False
        self.in_tagged_sentence = False
        self.got_a_div = False
        self.context_div_level = 0
        self.current_tag = "doc"
        self.current_div_id = ""

        self.last_sentence_marker = ""  # Caching the last sentence marker in case we need it for <s> tags

    def parse(self, text_input):
        """Top level function for reading a file and printing out the output."""
        self.input = text_input
        self.content = text_input.read()
        if div_tag.search(self.content):
            self.got_a_div = True
        self.cleanup_content()

        # Begin by creating a document level object, just call it "text" for now.
        self.v.push("doc", "text", 0)

        # if the parser was created with known_metadata,
        # we can attach it to the newly created doc object here.
        # you can attach metadata to an object at any time between push() and pull().
        for k, v in self.known_metadata.items():
            self.v["doc"][k] = v

        # Split content into a list on newlines.
        self.content = self.content.split("\n")
        self.closed_suppressed_tags = self.find_closed_suppressed_tags()

        self.bytes_read_in = 0
        self.line_count = 0
        for line in self.content:
            # Let's start indexing words and objects at either the <text
            # of the <docbody tag.  We can add more.
            if self.in_the_text is False:
                if text_tag.search(line) or doc_body_tag.search(line) or body_tag.search(line):
                    self.in_the_text = True

            self.line_count += 1

            if line.startswith("<"):
                self.bytes_read_in += len(line.encode("utf8"))
                # TODO : implement DUMPXPATHS?
                if self.in_the_text:
                    if self.tokenizer is None:
                        self.tag_handler(line)
                    elif self.is_inline_tag(line):
                        self.run.append((line, True, self.line_count, self.bytes_read_in))
                    else:
                        self.flush_run()
                        self.tag_handler(line)
            elif self.tokenizer is None:
                self.word_handler(line)
                self.bytes_read_in += len(line.encode("utf8"))
            else:
                if self.in_the_text:
                    self.run.append((line, False, self.line_count, self.bytes_read_in))
                self.bytes_read_in += len(line.encode("utf8"))
        self.flush_run()

        # if we have an open page, make sure we close it so it is properly stored
        if self.open_page:
            self.v.pull("page", self.bytes_read_in)
            self.open_page = False
        if self.open_div1 is True:
            self.close_div1(self.bytes_read_in)
        self.v["doc"].attrib["philo_doc_id"] = str(self.v["doc"].id[0])
        self.v.pull("doc", self.filesize)

    def cleanup_content(self):
        """Run various clean-ups before parsing."""
        # Replace carriage returns with spaces since they can give us all kinds of difficulties.
        self.content = self.content.replace("\r", " ")

        # What cleanup_content takes out (soft hyphens, tag exceptions), it replaces with a "_" for each byte, which
        # isn't text: start byte of each run of them -> its end byte (see index_words)
        self.placeholders = {}
        self.word_ends = {}  # end byte of a word joined across tag exceptions -> the end byte of its text in the file

        def placeholder(content):
            """A function replacing a match in content with a "_" for each of its bytes, which it records"""
            byte = byte_positions(content)

            def replace(match):
                start = byte(match.start())
                length = len(match.group().encode("utf8"))
                self.placeholders[start] = start + length
                return "_" * length

            return replace

        # Join hyphens
        if self.join_hyphen_in_words:
            self.content = join_hyphen_with_lb.sub(placeholder(self.content), self.content)
            self.content = join_hyphen.sub(placeholder(self.content), self.content)

        # Replace newlines with spaces.  Remember that we have seen lots of
        # tags with newlines, which can make a mess of things.
        self.content = self.content.replace("\n", " ")

        # Abbreviation expander
        self.expansions = []  # (start, end) bytes of each expansion in the content, and the bytes of its parts
        self.expansion_starts = []
        if self.abbrev_expand:
            self.content = self.expand_abbreviations(self.content)

        # An experimental inword tag spanner. For selected tags between letters, this takes the tags out of the word,
        # putting a "_" for each of their bytes after it (in order to keep the byte count).  This is to allow indexing
        # of words broken by tags. The word's text ends in the file after the tags inside it.
        byte = byte_positions(self.content)

        def word_part(token):
            """The word at the start of a token of the match, as split_with_token_regex splits it: "ter" in the "ter;"
            the entity alternative of TOKEN_REGEX lets the match have, none in ";"."""
            match = self.token_regex.match(token)
            return match.group() if match and WORD_CHAR.search(match.group()) else ""

        def replace_tag(match):
            text = match.group()
            runs = [(run.start(), run.end()) for run in self.tag_runs.finditer(text)]  # of tags, between tokens
            if word_part(text[: runs[0][0]]) != text[: runs[0][0]]:  # the tags aren't inside a word
                return text
            word, length, tags_length = text[: runs[0][0]], runs[0][0], 0  # the word, and what it takes of the match
            taken = None  # with the tag after it, if it has one
            for number, (tags_start, tags_end) in enumerate(runs):
                tags_length += len(text[tags_start:tags_end].encode("utf8"))
                if tags_end == len(text):  # the tag after the word
                    taken = tags_end
                    break
                token = text[tags_end : runs[number + 1][0] if number + 1 < len(runs) else len(text)]
                part = word_part(token)
                if not part:
                    tags_length -= len(text[tags_start:tags_end].encode("utf8"))
                    break
                word += part
                length = tags_end + len(part)
                if part != token:  # the word ends in this token (Laugh<g/>ter;): the rest is left as it is
                    break
            if not tags_length:  # no tag inside a word
                return text
            start = byte(match.start())
            word_end = start + len(word.encode("utf8"))
            self.word_ends[word_end] = start + len(text[:length].encode("utf8"))
            self.placeholders[word_end] = word_end + tags_length
            return word + "_" * tags_length + text[taken or length :]

        if self.tag_exception_starts is not None:
            self.content = replace_tag_exceptions(
                self.content, self.tag_exceptions, self.tag_exception_starts, replace_tag
            )
        elif self.tag_exceptions is not None:
            self.content = self.tag_exceptions.sub(replace_tag, self.content)

        # Add newlines to the beginning and end of all tags
        self.content = self.content.replace("<", "\n<").replace(">", ">\n")

    def expand_abbreviations(self, content):
        """content with its abbreviations, as <abbr expan="en">&emacr;</abbr> or <abbr expan="Valerius Maximus">Val.
        Max.</abbr>, replaced by their expansion, which the index has: the letters following one, as in
        pres<abbr expan="en">&emacr;</abbr>t, move after its expansion, to make one word (present), and a "_" for each
        byte left keeps the byte count. The words of an expansion get the bytes of the abbreviation in the file (see
        file_bytes): of the word it expands, if it has as many words (Valerius gets those of Val.), else all of it."""
        output, position = [], 0
        byte = byte_positions(content)
        for match in abbreviation_tag.finditer(content):
            expansion, abbreviation = match.group(2), match.group(3)
            if match.start() < position or not expansion.strip() or not self.abbreviation_words(abbreviation):
                continue
            if any(self.handler_of(tag, tag_matcher.findall(tag)[0]) for tag in any_tag.findall(abbreviation)):
                continue  # tags inside it which make objects (as pages): left as it is
            following_end = match.end()
            while following_end < len(content) and TOKEN_CHAR.match(content, following_end):
                following_end += 1
            text = expansion + content[match.end() : following_end]
            start = byte(match.start())
            abbreviation_start = byte(match.start(3))
            replaced_end = byte(following_end)
            text_end = start + len(text.encode("utf8"))
            # The parts of the expansion, and the bytes in the file of what each stands for
            expansion_words = [m.span() for m in re.finditer(r"\S+", expansion)]
            abbreviation_words = self.abbreviation_words(abbreviation)
            if len(expansion_words) != len(abbreviation_words):  # all of the abbreviation for each word
                expansion_words = [(expansion_words[0][0], expansion_words[-1][1])]
                abbreviation_words = [(abbreviation_words[0][0], abbreviation_words[-1][1])]
            parts = []
            for (x_start, x_end), (a_start, a_end) in zip(expansion_words, abbreviation_words):
                parts.append(
                    (
                        start + len(expansion[:x_start].encode("utf8")),
                        start + len(expansion[:x_end].encode("utf8")),
                        abbreviation_start + len(abbreviation[:a_start].encode("utf8")),
                        abbreviation_start + len(abbreviation[:a_end].encode("utf8")),
                    )
                )
            self.expansions.append((start, start + len(expansion.encode("utf8")), parts))
            self.expansion_starts.append(start)
            if replaced_end > text_end:
                self.placeholders[text_end] = replaced_end
            if following_end > match.end():  # a word joined across the abbreviation ends after the letters following it
                self.word_ends[text_end] = replaced_end
            output += [content[position : match.start()], text, "_" * (replaced_end - text_end)]
            position = following_end
        output.append(content[position:])
        return "".join(output)

    @staticmethod
    def abbreviation_words(abbreviation):
        """(start, end) of the words of an abbreviation, between spaces, its tags left out (as those of <w>)"""
        words, start, end = [], None, None
        for match in re.finditer(r"<[^>]*>|\s+|[^<\s]+", abbreviation):
            if match.group()[0] == "<":
                continue
            if match.group()[0].isspace():
                if start is not None:
                    words.append((start, end))
                start = None
            else:
                start = match.start() if start is None else start
                end = match.end()
        if start is not None:
            words.append((start, end))
        return words

    def file_bytes(self, start_byte, end_byte):
        """The bytes in the file of a record from start_byte to end_byte in the content: in an expansion of an
        abbreviation (see expand_abbreviations), those of what it stands for"""
        if self.expansions:
            number = bisect_right(self.expansion_starts, start_byte) - 1
            if number >= 0 and start_byte < self.expansions[number][1]:
                parts = self.expansions[number][2]
                start_byte = next((part[2] for part in parts if start_byte < part[1]), parts[-1][2])
            number = bisect_right(self.expansion_starts, end_byte - 1) - 1
            if number >= 0 and end_byte <= self.expansions[number][1]:
                parts = self.expansions[number][2]
                end_byte = next((part[3] for part in parts if end_byte <= part[1]), parts[-1][3])
        return start_byte, end_byte

    def is_tracked_tag(self, name, is_self_closing):
        """Suppressed and object open tags, which we match with their closing tags to find the extent of suppressed tags."""
        return not is_self_closing and (name in self.suppress_tags or name.lower() in OBJECT_TAGS)

    def find_closed_suppressed_tags(self):
        """Return the line numbers of suppressed tags which are closed before their parent object ends.
        Tags left unclosed have no content to suppress: this is typically dirty XML using them as empty tags."""
        closed_suppressed_tags = set()
        open_tags = []
        open_tag_counts = {}
        for line_num, line in enumerate(self.content):
            if not line.startswith("<"):
                continue
            try:
                name = tag_matcher.findall(line)[0]
            except IndexError:
                continue
            if name.startswith("/"):
                name = name[1:]
                if open_tag_counts.get(name):
                    # Close the most recent matching tag: any tags opened after it were left unclosed
                    while True:
                        open_name, open_line_num = open_tags.pop()
                        open_tag_counts[open_name] -= 1
                        if open_name == name:
                            if name in self.suppress_tags:
                                closed_suppressed_tags.add(open_line_num)
                            break
            elif self.is_tracked_tag(name, line.rstrip().endswith("/>")):
                open_tags.append((name, line_num))
                open_tag_counts[name] = open_tag_counts.get(name, 0) + 1
        return closed_suppressed_tags

    def skip_suppressed_tag(self, tag, tag_name):
        """Track suppressed tags. Returns True if the tag is a suppressed tag or is inside one, and should be skipped."""
        is_closing = tag_name.startswith("/")
        name = tag_name.lstrip("/")
        is_self_closing = tag.rstrip().endswith("/>")
        if self.in_suppressed_tag is False:
            if is_closing or name not in self.suppress_tags:
                return False
            # Self-closing and unclosed tags have no content to suppress
            if self.line_count - 1 in self.closed_suppressed_tags:
                self.suppressed_tag_stack.append(name)
                self.in_suppressed_tag = True
            return True

        if is_closing and name in self.suppressed_tag_stack:
            # Close the most recent matching tag, along with any unclosed tags opened inside it
            while self.suppressed_tag_stack.pop() != name:
                pass
            self.in_suppressed_tag = bool(self.suppressed_tag_stack)
            return True
        if not is_closing and self.is_tracked_tag(name, is_self_closing):
            self.suppressed_tag_stack.append(name)

        # Keep page breaks so that page numbers stay correct after the suppressed tag
        return not page_tag.search(tag)

    def tag_handler(self, tag):
        """Tag handler for parser: runs the handler of the tag (see TAG_HANDLERS), if it has one."""
        start_byte = self.bytes_read_in - len(tag.encode("utf8"))
        try:
            tag_name = tag_matcher.findall(tag)[0]
        except IndexError:
            tag_name = "unparsable_tag"
        if self.skip_suppressed_tag(tag, tag_name):
            return
        if not tag_name.startswith("/"):
            self.current_tag = tag_name

        # Word tags: store attributes to be attached to the actual word in word_handler. Tags closed after a word
        # tag clear them (current_tag only changes with opening tags).
        if self.current_tag == "w":
            self.word_tag_attributes = self.get_attributes(tag)

        method = self.handler_of(tag, tag_name)
        if method is not None:
            getattr(self, method)(tag, tag_name, start_byte)

    def handler_of(self, tag, tag_name):
        """The handler method of a tag (see TAG_HANDLERS): the one for its name, if the tag matches its regex"""
        if tag_name.isascii():
            name = tag_name.lower()
            if name.startswith("div"):  # div_tag and closed_div_tag match all names starting with div
                name = "div"
            elif name.startswith("/div"):
                name = "/div"
            handler = TAG_HANDLERS.get(name)
            if handler is not None:
                pattern, method = handler
                if pattern.search(tag):
                    return method
        else:  # ignoring case, letters other than ASCII ones can match the regexes: try them all
            for pattern, method in TAG_HANDLERS.values():
                if pattern.search(tag):
                    return method
        return None

    # Tag handlers (see TAG_HANDLERS): each gets the tag, its name and its start byte

    def handle_paragraph(self, tag, tag_name, start_byte):
        """Paragraphs, but in notes and objects blocking deeper objects"""
        if self.in_a_note or self.no_deeper_objects:
            return
        if self.open_para:  # account for unclosed paragraph tags
            self.close_para(start_byte)
        self.v.push("para", tag_name, start_byte)
        self.get_object_attributes(tag, tag_name, "para")
        self.open_para = True

    def handle_paragraph_object(self, tag, tag_name, start_byte):
        """Tags other than <p> made paragraph objects"""
        if self.open_para:  # account for unclosed paragraph tags
            self.close_para(start_byte)
        self.open_para = True
        self.v.push("para", tag_name, start_byte)
        self.get_object_attributes(tag, tag_name, "para")

    def handle_paragraph_object_end(self, tag, tag_name, start_byte):
        self.close_para(self.bytes_read_in)

    def handle_blocking_paragraph_object(self, tag, tag_name, start_byte):
        """Paragraph objects in which there are no deeper objects (sp, epigraph...)"""
        self.handle_paragraph_object(tag, tag_name, start_byte)
        self.no_deeper_objects = True

    def handle_blocking_paragraph_object_end(self, tag, tag_name, start_byte):
        self.close_para(self.bytes_read_in)
        self.no_deeper_objects = False
        self.open_para = False

    def handle_inner_paragraph_object(self, tag, tag_name, start_byte):
        """Paragraph objects unless inside blocking ones (list, stage)"""
        if not self.no_deeper_objects:
            self.handle_paragraph_object(tag, tag_name, start_byte)

    def handle_inner_paragraph_object_end(self, tag, tag_name, start_byte):
        if not self.no_deeper_objects:
            self.close_para(self.bytes_read_in)
            self.open_para = False

    def handle_note(self, tag, tag_name, start_byte):
        """Notes: paragraph objects, in which paragraphs are not objects"""
        self.handle_paragraph_object(tag, tag_name, start_byte)
        self.in_a_note = True

    def handle_note_end(self, tag, tag_name, start_byte):
        if self.in_a_note:  # paragraphs are objects again after the note
            self.close_para(self.bytes_read_in)
            self.in_a_note = False

    def handle_speaker(self, tag, tag_name, start_byte):
        """Speaker tags: we attach them to the parent <sp>"""
        if self.open_para:
            next_line_num = self.line_count
            speaker_name = ""
            while True:
                next_line = self.content[next_line_num]
                if next_line.startswith("<"):  # we've reached the next tag
                    if self.tokenizer is not None and self.is_inline_tag(next_line):  # left in the text (see flush_run)
                        next_line_num += 1
                        continue
                    break
                speaker_name += next_line.strip() if self.tokenizer is None else next_line
                next_line_num += 1
            if self.tokenizer is not None:  # lines around inline tags: the spaces between them kept, once
                speaker_name = " ".join(speaker_name.split())
            speaker_name = re.sub(r"\.$", "", speaker_name)  # trailing period often found in speaker names
            self.v["para"]["speaker"] = speaker_name.title()

    def handle_page(self, tag, tag_name, start_byte):
        """Page breaks: this updates the current page tag, its n attribute "na" if not found"""
        if self.open_page:
            self.v.pull("page", start_byte)
            self.open_page = False
        self.v.push("page", tag_name, start_byte)
        self.get_object_attributes(tag, tag_name, "page")
        try:
            n = self.v["page"]["n"]
            n = n.replace(" ", "_").replace("-", "_").lower()
            if not n:
                n = "na"
        except KeyError:
            n = "na"
        self.v["page"]["n"] = n
        self.open_page = True

    def handle_line_group(self, tag, tag_name, start_byte):
        """Line groups: paragraph objects, in which lines can break sentences (see handle_line)"""
        if self.no_deeper_objects:
            return
        if self.break_sent_in_line_group:
            self.in_line_group = True
        self.handle_paragraph_object(tag, tag_name, start_byte)

    def handle_line_group_end(self, tag, tag_name, start_byte):
        self.in_line_group = False
        self.close_para(self.bytes_read_in)

    def handle_line(self, tag, tag_name, start_byte):
        """Lines: line objects, and sentences in line groups when break_sent_in_line_group"""
        if self.in_line_group and self.break_sent_in_line_group:
            self.v.push("sent", tag_name, start_byte)
            self.v.pull("sent", self.bytes_read_in)
        # Create line parallel object
        self.v.push("line", tag_name, start_byte)
        self.get_object_attributes(tag, tag_name, "line")
        self.v["line"].attrib["doc_id"] = self.docid

    def handle_line_end(self, tag, tag_name, start_byte):
        if self.in_line_group and self.break_sent_in_line_group:
            self.v.pull("sent", self.bytes_read_in)
        self.v.pull("line", self.bytes_read_in)

    def handle_ab(self, tag, tag_name, start_byte):
        """Anonymous blocks: line objects"""
        self.v.push("line", tag_name, start_byte)
        self.get_object_attributes(tag, tag_name, "line")
        self.v["line"].attrib["doc_id"] = self.docid

    def handle_ab_end(self, tag, tag_name, start_byte):
        self.v.pull("line", self.bytes_read_in)

    def handle_sentence(self, tag, tag_name, start_byte):
        """Sentence tags: a new sentence, which turns off automatic sentence tagging. The last one ends where it starts."""
        if self.open_sent or self.in_tagged_sentence:
            self.close_sent(start_byte)
        self.v.push("sent", tag_name, start_byte)
        self.in_tagged_sentence = True
        self.open_sent = True

    def handle_sentence_end(self, tag, tag_name, start_byte):
        self.close_sent(self.bytes_read_in)
        self.in_tagged_sentence = False
        self.open_sent = False

    def handle_front(self, tag, tag_name, start_byte):
        """Front: treated as a div1"""
        if self.open_div1:
            self.close_div1(start_byte)
        self.in_front_matter = True
        self.v.push("div1", "front", start_byte)
        self.current_div_id = self.v["div1"].id
        self.get_object_attributes(tag, tag_name, "div1")
        self.context_div_level = 1
        self.open_div1 = True

    def handle_front_end(self, tag, tag_name, start_byte):
        self.in_front_matter = False
        self.context_div_level = 0
        self.close_div1(self.bytes_read_in)

    def handle_body(self, tag, tag_name, start_byte):
        """Body: a div1 when the document has no divs"""
        if self.got_a_div:
            return
        self.v.push("div1", tag_name, start_byte)
        self.current_div_id = self.v["div1"].id
        self.context_div_level = 1
        self.v["div1"].attrib["philo_div1_id"] = " ".join(str(i) for i in self.v["div1"].id[:2])
        self.open_div1 = True
        div_head = self.get_div_head(tag)
        if "[NA]" in div_head or "[na]" in div_head:
            div_head = "Document Body"
        self.v["div1"]["head"] = div_head

    def handle_hyperdiv(self, tag, tag_name, start_byte):
        """HyperDiv: a Brown WWP construct, a place to put information related to the body of the text which doesn't
        appear directly within its flow (footnotes, acrostics, castlist information required for the who attribute
        of <speaker>...): a div1"""
        if self.open_div1:
            self.close_div1(start_byte)
        self.context_div_level = 1
        self.open_div1 = True
        self.v.push("div1", tag_name, start_byte)
        self.current_div_id = self.v["div1"].id
        self.v["div1"]["head"] = "[HyperDiv]"

    def handle_div_end(self, tag, tag_name, start_byte):
        if "div1" in tag_name:
            if self.in_front_matter:
                self.close_div2(self.bytes_read_in)
            else:
                self.close_div1(self.bytes_read_in)
        elif "div2" in tag_name:
            if self.in_front_matter:
                self.close_div3(self.bytes_read_in)
            else:
                self.close_div2(self.bytes_read_in)
        elif "div3" in tag_name:
            self.close_div3(self.bytes_read_in)
        self.context_div_level -= 1
        self.no_deeper_objects = False

    def handle_div(self, tag, tag_name, start_byte):
        """Divs: division levels, from their number (1, 2 or 3) or else their depth, with the <head> found in them"""
        self.context_div_level += 1
        if self.context_div_level > 3:
            if self.open_div3:
                self.close_div3(start_byte)
            self.context_div_level = 3
        if self.context_div_level < 1:
            self.context_div_level = 1

        div_level = tag_name[-1]
        if not div_level.isdigit():
            div_level = self.context_div_level
        elif div_level == "0" or int(div_level) > 3:
            div_level = self.context_div_level
        else:
            div_level = int(div_level)
        # <Front is top level div1, so inner divs should be div2s or div3s
        if self.in_front_matter:
            div_level = self.context_div_level
        self.context_div_level = div_level

        # TODO: ignore divs inside of internal text tags.  Setable
        # from configuration.  But we will bump the para and sent args
        div_type = "div%d" % div_level
        if div_type == "div1":
            if self.open_div1:
                self.close_div1(start_byte)
            self.open_div1 = True
            self.v.push("div1", tag_name, start_byte)
            self.current_div_id = self.v["div1"].id
            self.v["div1"]["head"] = self.get_div_head(tag)
            self.get_object_attributes(tag, tag_name, object_type="div1")
        elif div_type == "div2":
            if self.open_div2:
                self.close_div2(start_byte)
            self.open_div2 = True
            self.v.push("div2", tag_name, start_byte)
            self.current_div_id = self.v["div2"].id
            self.v["div2"]["head"] = self.get_div_head(tag)
            self.get_object_attributes(tag, tag_name, object_type="div2")
        else:
            if self.open_div3:
                self.close_div3(start_byte)
            self.open_div3 = True
            self.v.push("div3", tag_name, start_byte)
            self.current_div_id = self.v["div3"].id
            self.v["div3"]["head"] = self.get_div_head(tag)
            self.get_object_attributes(tag, tag_name, object_type="div3")

        if "type" in self.v[div_type]:
            if self.v[div_type]["type"] == "notes":
                self.no_deeper_objects = True

    def handle_index(self, tag, tag_name, start_byte):
        """Index tags inside a div: attributes describing it (the type attribute's value in the value attribute)"""
        if self.context_div_level == 0:
            return
        attrib = dict(self.get_attributes(tag))
        div = f"div{self.context_div_level}"
        if "type" in attrib:
            if attrib["type"] in self.metadata_to_parse["div"]:
                try:
                    self.set_metadata_value(div, attrib["type"], attrib["value"])
                except KeyError:
                    pass
        else:
            for metadata_name, metadata_value in attrib.items():
                self.set_metadata_value(div, metadata_name, metadata_value)

    def handle_date(self, tag, tag_name, start_byte):
        """Dates: attributes of the current div"""
        div = f"div{self.context_div_level}"
        for attrib_name, attrib_value in self.get_attributes(tag):
            if attrib_name == "value" or attrib_name == "when":
                if "div_date" not in self.v[div].attrib:
                    self.set_metadata_value(div, "div_date", attrib_value)
            else:
                attrib_name = f"div_{attrib_name}"
                if attrib_name not in self.v[div].attrib:
                    self.v[div].attrib[attrib_name] = attrib_value

    def handle_ref(self, tag, tag_name, start_byte):
        self.v.push("ref", tag_name, start_byte)
        self.get_object_attributes(tag, tag_name, "ref")
        self.v["ref"].attrib["parent"] = " ".join([str(i) for i in self.current_div_id])
        self.v.pull("ref", self.bytes_read_in)

    def handle_graphic(self, tag, tag_name, start_byte):
        self.v.push("graphic", tag_name, start_byte)
        self.get_object_attributes(tag, tag_name, "graphic")
        self.v["graphic"].attrib["parent"] = " ".join([str(i) for i in self.current_div_id])
        self.v.pull("graphic", self.bytes_read_in)

    def set_metadata_value(self, object_level, metadata, value):
        try:
            if self.metadata_sql_types[metadata] == "date":
                self.v[object_level].attrib[metadata] = extract_full_date(value)
            elif self.metadata_sql_types[metadata] == "int":
                self.v[object_level].attrib[metadata] = extract_integer(value)
            else:
                self.v[object_level].attrib[metadata] = value
        except KeyError:
            try:
                self.v[object_level].attrib[metadata] = value
            except Exception:
                pass

    def word_handler(self, words):
        """
        Word handler. It takes an artbitrary string or words between two tags and
        splits them into words. It also identifies sentence breaks.  I am not checking for existing sentence
        tags, but could and conditionalize it. Also note that it does not check for
        sentences inside of linegroups ... which are typically broken on lines.
        I am not currently handling ";" at the end of words because of confusion with
        character ents. Easily fixed.
        """

        if self.in_suppressed_tag is False:
            # We don't like many character entities, so let's change them
            # into spaces to get a clean break.
            if char_ents.search(words):
                words = self.clear_char_ents(words)
            word_list = self.split_with_token_regex(words)
            if self.in_the_text:
                self.index_words(word_list, self.bytes_read_in)

    def split_with_token_regex(self, words):
        """The items of words (see index_words), split by token_regex"""
        # we're splitting the line of words into distinct words
        # separated by "\n"
        if self.token_regex.groups == 1:
            # Same result as the sub below (the pattern's only group is the whole match), but faster
            words = "\n".join(self.token_regex.split(words))
        else:
            words = self.token_regex.sub(r"\n\1\n", words)

        if self.break_apost:
            words = words.replace("'", "\n'\n")
            words = words.replace("’", "\n’\n")

        word_list = words.split("\n")
        if len(word_list) > 2:
            # Drop empty inner items: same as collapsing runs of newlines before splitting
            word_list = [word_list[0], *[w for w in word_list[1:-1] if w], word_list[-1]]
        return word_list

    def index_words(self, word_list, current_pos):
        """Index the items of word_list, split by token_regex, which follow each other from byte current_pos on: words,
        and what's between them (spaces, punctuation), which may end a sentence. Each gets the bytes of its text in the
        file: a word joined across tag exceptions ends after the tags inside it, and what's between words leaves out
        the spaces around it and the "_" standing for what cleanup_content took out."""
        count = 0
        last_word = ""
        next_word = ""
        word_list_length = len(word_list)
        for word in word_list:
            word_in_utf8 = word.encode("utf8")
            word_length = len(word_in_utf8)
            if count + 1 < word_list_length:
                next_word = word_list[count + 1]
            count += 1

            # Keep track of your bytes since this is where you are getting
            # the byte offsets for words.
            current_pos += word_length
            item_start = current_pos - word_length
            if self.placeholders and item_start in self.placeholders:  # not text: left out
                placeholder_length = self.placeholders[item_start] - item_start
                word, item_start = word[placeholder_length:], item_start + placeholder_length

            # Do we have a word? At least one of these characters.
            converted_word = word
            if "&" in word:  # convert entities before checking if there are word chars
                word = self.latin1_ents_to_utf8(word)
                word = self.convert_other_ents(word)
                word_match = check_if_char_word.search(word)
                converted_word = word
            else:
                word_match = check_if_char_word.search(word)
            if word_match:
                last_word = word
                word_pos = current_pos - len(word_in_utf8)
                if "&" in word:
                    word = converted_word  # use the converted word from above
                word = self.index_form(word)

                # Check to see if the word is longer than we want.  More than 235
                # characters appear to cause problems in the indexer.
                if len(word_in_utf8) > self.long_word_limit:
                    self.skip_long_word(word)
                    continue
                if word:

                    if self.defined_words_to_index:
                        if word not in self.words_to_index:
                            continue
                    token = converted_word if self.spacy_model and converted_word != word else None
                    self.push_word(word, word_pos, self.word_ends.get(current_pos, current_pos), token)
                is_sentence_breaker = self.is_word_sentence_breaker(word, last_word, next_word)
            else:
                is_sentence_breaker = self.is_word_sentence_breaker(word, last_word, next_word)
                # Sentence break handler
                if is_sentence_breaker and not self.in_line_group and not self.in_tagged_sentence:
                    # a little hack--we don't know the punctuation mark that will end a sentence
                    # until we encounter it--so instead, we let the push on "word" create a
                    # implicit philo_virtual sentence, then change its name once we actually encounter
                    # the punctuation token.
                    self.end_sentence(word.replace("\t", " ").strip(), *self.text_bytes(word, item_start, current_pos))
            if is_sentence_breaker:
                self.last_sentence_marker = word.replace("\t", " ").strip()
            if self.punct_regex.search(word):
                self.push_punct(word, *self.text_bytes(word, item_start, current_pos))

    def index_tokens(self, items):
        """Index the items of a run split by the tokenizer (see flush_run), each with its exact bytes in the file:
        (text, start byte, end byte, whether it's a word, token). A word's token is the tokenizer's, as it has it in the
        text (which a spaCy model tags): when the indexed word isn't that (it's in lowercase, or leaves out the
        punctuation at the edges of the token), the token is kept in the spacy_token attribute of the word. Between
        words, items are split as without a tokenizer, and end sentences as they do without one."""
        last_word = ""
        for number, (text, start_byte, end_byte, is_word, token) in enumerate(items):
            converted = self.convert_entities(text) if "&" in text else text
            if is_word:
                last_word = converted
                word = self.index_form(converted)
                if len(text.encode("utf8")) > self.long_word_limit:
                    self.skip_long_word(word)
                elif word and (not self.defined_words_to_index or word in self.words_to_index):
                    end_byte = self.word_ends.get(end_byte, end_byte)  # (joined across an abbreviation)
                    self.push_word(word, start_byte, end_byte, token if token != word else None)
                continue
            if self.placeholders and start_byte in self.placeholders:  # not text: left out
                placeholder_length = self.placeholders[start_byte] - start_byte
                text, converted = text[placeholder_length:], converted[placeholder_length:]
                start_byte += placeholder_length
            # The bytes of what's between words, without the spaces around it
            start_byte, end_byte = self.text_bytes(text, start_byte, end_byte)
            next_word = items[number + 1][0] if number + 1 < len(items) else ""
            if self.is_word_sentence_breaker(converted, last_word, next_word):
                sentence_name = converted.replace("\t", " ").strip()
                if not self.in_line_group and not self.in_tagged_sentence:
                    self.end_sentence(sentence_name, start_byte, end_byte)
                self.last_sentence_marker = sentence_name
            if self.punct_regex.search(converted):
                self.push_punct(converted, start_byte, end_byte)

    @staticmethod
    def text_bytes(item, start_byte, end_byte):
        """The start and end bytes of the text of an item from start_byte to end_byte, without the spaces around it"""
        stripped = item.strip()
        if not stripped:
            return start_byte, end_byte
        start_byte += len(item[: len(item) - len(item.lstrip())].encode("utf8"))
        return start_byte, end_byte - len(item[len(item.rstrip()) :].encode("utf8"))

    def index_form(self, word):
        """The form under which a word, its entities converted, is indexed"""
        # You may have some semi-colons...
        if ";" in word:
            if "&" in word:
                pass  # TODO
            else:
                word = semi_colon_strip.sub(r"\1", word)  # strip on both ends

        # Get rid of certain characters that don't break words, but don't index.
        # These are defined in a compiled regex below: chars_not_to_index
        word = self.chars_not_to_index.sub("", word)

        # TODO: Call a function to distinguish between words beginning with an
        # upper case and lower case character.  This USED to be a proper
        # name split in ARTFL, but we don't see many databases with proper
        # names tagged.

        # Switch everything to lower case
        if self.lowercase_index is True:
            word = word.lower()

        word = self.remove_control_chars(word)
        word = word.replace("_", "").strip()
        return word.replace(" ", "")

    def skip_long_word(self, word):
        """Words longer than long_word_limit bytes, which appear to cause problems in the indexer, aren't indexed"""
        print("Long word in {}: {}".format(self.input.name, word), file=sys.stderr)
        print(f"Removing word from for index since over {self.long_word_limit} bytes long...", file=sys.stderr)

    def push_word(self, word, start_byte, end_byte, token=None):
        """Store a word, with the attributes of its <w> tag or its lemma, and its token (as a spaCy model tags it), if
        given"""
        start_byte, end_byte = self.file_bytes(start_byte, end_byte)
        self.v.push("word", word, start_byte)
        if self.current_tag == "w":
            word_record = self.v["word"]
            for attrib, value in self.word_tag_attributes:
                word_record[attrib] = value
        elif self.lemmas is not None:
            if word not in self.lemmas:
                lower_word = word.lower()
                try:
                    self.v["word"]["lemma"] = self.lemmas[lower_word]
                except KeyError:
                    self.v["word"]["lemma"] = word.lower()
            else:
                self.v["word"]["lemma"] = self.lemmas[word]
        if token is not None:
            self.v["word"]["spacy_token"] = token
        self.v.pull("word", end_byte)

    def end_sentence(self, name, start_byte, end_byte):
        """End the sentence (one starting at start_byte, if words haven't started one), named after its punctuation"""
        start_byte, end_byte = self.file_bytes(start_byte, end_byte)
        if "sent" not in self.v:
            self.v.push("sent", name, start_byte)
        self.v["sent"].name = name
        self.v.pull("sent", end_byte)

    def push_punct(self, punct, start_byte, end_byte):
        """Store punctuation, from start_byte to end_byte"""
        punct = punct.strip()
        punct = punct.replace("\t", " ")
        punct = self.remove_control_chars(punct)
        if punct and not punct.isspace():
            start_byte, end_byte = self.file_bytes(start_byte, end_byte)
            self.v.push("punct", punct, start_byte)
            self.v.pull("punct", end_byte)

    def convert_entities(self, text):
        """Entities as index_words converts them in words"""
        return self.convert_other_ents(self.latin1_ents_to_utf8(text))

    def is_inline_tag(self, tag):
        """Whether a tag is an inline tag, which doesn't end the text the tokenizer splits (see flush_run): a tag
        exception, unless it's a suppressed tag or an object tag, which the words after it depend on"""
        if self.inline_tags is None or not self.inline_tags.fullmatch(tag):
            return False
        try:
            name = tag_matcher.findall(tag)[0]
        except IndexError:
            return False
        return not self.is_tracked_tag(name.lstrip("/"), False)

    def flush_run(self):
        """Index the run of text lines and inline tags in self.run (lines in order, with whether each is a tag, its
        line count and its start byte, or end byte for a tag), with the tokenizer. Its words are split from the text
        of its lines, the inline tags left out, so that a tag inside a word doesn't split it. Inline tags between words
        are handled where they are, as tags outside runs are. Inside a word, those with a handler (see TAG_HANDLERS)
        are handled after it; the others are dropped, as cleanup_content drops tag exceptions without a tokenizer."""
        run, self.run = self.run, []
        if not run:
            return
        line_count, bytes_read_in = self.line_count, self.bytes_read_in
        pieces, tags = [], []  # (start byte, text) of the lines; (position in the text, tag, line count, end byte)
        text_length = 0
        for line, is_tag, count, byte in run:
            if is_tag:
                tags.append((text_length, line, count, byte))
            elif line:
                if char_ents.search(line):
                    line = self.clear_char_ents(line)
                pieces.append((byte, line))
                text_length += len(line)
        text = "".join(line for _, line in pieces)
        text_bytes = TextBytes(pieces)
        words = []
        if not self.in_suppressed_tag and check_if_char_word.search(text):  # (letters of entities too)
            words = self.split_words(text)

        # Lists of items to index (see index_tokens), and the tags to handle between them, in order
        steps, items = [], []
        position = tag_number = 0

        def add_items_between_words(end):
            """The items of the text from position to end, between words and tags, split as without a tokenizer"""
            nonlocal position
            for item in self.split_with_token_regex(text[position:end]) if end > position else ():
                if item:
                    item_end = position + len(item)
                    items.append((item, text_bytes.start(position), text_bytes.end(item_end), False, None))
                    position = item_end

        for token_start, token_end, word_start, word_end, token in words + [(text_length,) * 4 + (None,)]:
            while tag_number < len(tags) and tags[tag_number][0] <= token_start:  # between words: handled there
                add_items_between_words(tags[tag_number][0])
                steps += [items, tags[tag_number]]
                items = []
                tag_number += 1
            add_items_between_words(token_start)
            if token_end > token_start:  # a word
                handled_after = []  # tags inside it with a handler, handled after it; the others are dropped
                while tag_number < len(tags) and tags[tag_number][0] < token_end:
                    tag = tags[tag_number][1]
                    if self.handler_of(tag, tag_matcher.findall(tag)[0]) is not None:
                        handled_after.append(tags[tag_number])
                    tag_number += 1
                word_bytes = (text_bytes.start(word_start), text_bytes.end(word_end))
                items.append((text[word_start:word_end], *word_bytes, True, token))
                if handled_after:
                    steps += [items, *handled_after]
                    items = []
            position = token_end
        steps.append(items)

        for step in steps:
            if isinstance(step, list):
                if step and not self.in_suppressed_tag:
                    self.index_tokens(step)
            else:
                _, tag, self.line_count, self.bytes_read_in = step
                self.tag_handler(tag)
        self.line_count, self.bytes_read_in = line_count, bytes_read_in

    def split_words(self, text):
        """The words the tokenizer finds in text, with letters once their entities are converted (as index_tokens has
        them): see spacy_tokenizer.split_words. Characters not to index split words, as token_regex splits them."""
        separated = tokenizer_input(text, self.chars_not_to_index)
        mapped = self.tokenizer_text(separated) if "&" in separated or "_" in separated else (separated,)
        return split_words(self.tokenizer, text, self.has_letter, *mapped)

    def has_letter(self, word):
        """Whether a word has a letter, once its entities are converted"""
        return check_if_char_word.search(self.convert_entities(word) if "&" in word else word) is not None

    def tokenizer_text(self, text):
        """text as the tokenizer splits it: entities as the characters they stand for, and without "_", which
        cleanup_content puts in to join words (as at &shy;); and for each of its characters, the start and end in text
        of what it stands for"""
        parts, starts, ends = [], [], []
        position = 0
        for match in tokenizer_special_chars.finditer(text):
            start, end = match.span()
            if start > position:
                parts.append(text[position:start])
                starts.extend(range(position, start))
                ends.extend(range(position + 1, start + 1))
            if text[start] == "&":
                characters = self.entity_characters(match.group())
                parts.append(characters)
                starts.extend([start] * len(characters))
                ends.extend([end] * len(characters))
            position = end
        if position < len(text):
            parts.append(text[position:])
            starts.extend(range(position, len(text)))
            ends.extend(range(position + 1, len(text) + 1))
        return "".join(parts), starts, ends

    def entity_characters(self, entity):
        """The characters an entity stands for, for the tokenizer: as index_words converts it, or as HTML has it, or
        else a character spaCy keeps in words, as token_regex does entities"""
        characters = self.convert_entities(entity)
        if characters.startswith("&"):
            characters = html.unescape(entity)
        if not characters or characters.startswith("&"):
            characters = UNKNOWN_ENTITY_CHARACTER
        return characters

    def is_word_sentence_breaker(self, word, last_word, next_word):
        is_sent = False
        # Break on ! and ?, unless a word in lowercase follows, which goes on with the sentence: an interjection
        # (Ah! mon Dieu!) or the words introducing what was said (« Viens-tu ? » dit-il)
        # TODO: why test if word > 2 in p3?
        if "!" in word or "?" in word:
            is_sent = not next_word.islower()

        # Periods are messy. Let's try by length of previous word and
        # capital letters to avoid hitting abbreviations.
        elif "." in word:
            is_sent = True
            if len(last_word) < 3:
                if cap_char_or_num.search(last_word):
                    is_sent = False

            # Periods in numbers don't break sentences.
            if next_word.islower() or next_word.isdigit():
                is_sent = False

        elif word in self.sentence_breakers:
            is_sent = True
        return is_sent

    def close_sent(self, end_byte):
        """Close sentence objects."""
        try:
            self.v["sent"].name = self.last_sentence_marker
        except IndexError:
            pass
        self.v.pull("sent", end_byte)
        self.open_sent = False

    def close_para(self, end_byte):
        """Close paragraph objects."""
        if self.open_sent:
            self.close_sent(end_byte)
        self.v.pull("para", end_byte)
        self.open_para = False

    def close_div3(self, end_byte):
        """Close div3 objects."""
        if self.open_para:
            self.close_para(end_byte)
        if self.open_div3 is True:
            if "div3" in self.v:
                self.v["div3"].attrib["philo_div3_id"] = " ".join(str(i) for i in self.v["div3"].id[:4])
                self.v.pull("div3", end_byte)
            self.open_div3 = False

    def close_div2(self, end_byte):
        """Close div2 objects."""
        if self.open_div3:
            self.close_div3(end_byte)
        if self.open_div2 is True:
            if "div2" in self.v:
                self.v["div2"].attrib["philo_div2_id"] = " ".join(str(i) for i in self.v["div2"].id[:3])
                self.v.pull("div2", end_byte)
            self.open_div2 = False

    def close_div1(self, end_byte):
        """Close div1 objects."""
        if self.open_div2:
            self.close_div2(end_byte)
        if self.open_div1 is True:
            if "div1" in self.v:
                self.v["div1"].attrib["philo_div1_id"] = " ".join(str(i) for i in self.v["div1"].id[:2])
                self.v.pull("div1", end_byte)
            self.open_div1 = False

    def camel_case_to_snake_case(self, word):
        word = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", word)
        word = re.sub("([a-z0-9])([A-Z])", r"\1_\2", word).lower()
        return word

    def get_attributes(self, tag):
        """Find all attributes for any given tag."""
        attribs = []
        for attrib, value in attrib_matcher.findall(tag):
            # Replace ":" with "_" for attribute names since they are illegal in SQLite
            attrib = attrib.replace(":", "_")
            value = self.remove_control_chars(value).strip()
            if value and value[-1] in ENDING_PUNCTUATION_CHARS:  # same as ending_punctuation.sub("", value)
                value = value[:-1]
            value = convert_entities(value)
            value = value.replace('"', "")
            attribs.append((attrib, value))
        return attribs

    def get_object_attributes(self, tag, tag_name, object_type):
        """Get all attributes for any given tag and attach metadata to element."""
        try:
            text_object = self.tag_to_obj_map[tag_name.lower()]
            for attrib, value in self.get_attributes(tag):
                attrib = self.camel_case_to_snake_case(attrib)
                if attrib in self.metadata_to_parse[text_object]:
                    self.set_metadata_value(object_type, attrib, value)
        except KeyError:
            pass

    def get_div_head(self, tag):
        """Get div head."""
        read_more = False
        look_ahead = self.line_count
        overflow_trap = 0
        div_head = ""
        while not read_more:
            look_ahead += 1
            try:
                next_line = self.content[look_ahead]
            except IndexError:
                break
            if "<" not in next_line:  # none of the tag regexes below can match
                continue
            next_line = head_self_close_tag.sub("", next_line)
            if div_tag.search(next_line) or closed_div_tag.search(next_line):
                break  # don't go past an open or close <div.
            if head_tag.search(next_line):
                read_more = True
            if read_more:
                while read_more:
                    look_ahead += 1
                    overflow_trap += 1
                    try:
                        next_line = self.content[look_ahead]
                    except IndexError:
                        break
                    if closed_head_tag.search(next_line):
                        read_more = False
                    elif overflow_trap > 50:  # Overflow trap in case you miss </head
                        read_more = False
                    else:
                        div_head += next_line
        if div_head:
            div_head = self.clear_char_ents(div_head)
            div_head = self.latin1_ents_to_utf8(div_head)
            div_head = self.convert_other_ents(div_head)
            div_head = re.sub(r"\n?<[^>]*>\n?", "", div_head)
            div_head = div_head.replace("_", "")
            div_head = div_head.replace("\t", "")
            div_head = " ".join(div_head.split())  # remove double or more spaces
            div_head = div_head.replace("[", "").replace("]", "")
            div_head = div_head.replace('"', "")
            div_head = div_head.strip()
        elif "type=" in tag:
            try:
                div_head = type_attrib.findall(tag)[0]
            except IndexError:
                pass

        if not div_head:
            div_head = "[NA]"

        # TODO: evaluate need for below...
        if div_head == "[>]" or div_head == "[<]":
            div_head = "[NA]"

        div_head = self.remove_control_chars(div_head)
        div_head = convert_entities(div_head)
        div_head = div_head.replace('"', "")
        return div_head

    def clear_char_ents(self, text):
        """Replaces a selected set of SGML character ents with spaces in order to keep the byte count right."""
        # We would want to read a list of character ents that should NOT be considered
        # valid for including in words or, more likely, a list of VALID characters from a general table.
        if self.break_apost:
            try:
                text = apost_ent.sub(lambda match: "," + " " * len(match.group(1) - 1), text)
            except IndexError:
                pass
        for regex in entity_regex:
            try:
                text = regex.sub(lambda match: " " * len(match.group(1)), text)
            except IndexError:
                pass
        return text

    def latin1_ents_to_utf8(self, text):
        """Converts ISO-LATIN-1 character entities in index words to UTF-8
        for standard word index search consistency. This is for SGML data sets
        and XML that have character ents rather than UTF-8 characters."""
        text_in_bytes = text.encode("utf8")
        if self.flatten_ligatures:
            text_in_bytes = text_in_bytes.replace(b"&AElig;", b"\xc3\x86")
            text_in_bytes = text_in_bytes.replace(b"&szlig;", b"\xc3\x9F")
            text_in_bytes = text_in_bytes.replace(b"&aelig;", b"\xc3\xA6")
        text_in_bytes = text_in_bytes.replace(b"&Agrave;", b"\xc3\x80")
        text_in_bytes = text_in_bytes.replace(b"&Aacute;", b"\xc3\x81")
        text_in_bytes = text_in_bytes.replace(b"&Acirc;", b"\xc3\x82")
        text_in_bytes = text_in_bytes.replace(b"&Atilde;", b"\xc3\x83")
        text_in_bytes = text_in_bytes.replace(b"&Auml;", b"\xc3\x84")
        text_in_bytes = text_in_bytes.replace(b"&Aring;", b"\xc3\x85")
        text_in_bytes = text_in_bytes.replace(b"&Ccedil;", b"\xc3\x87")
        text_in_bytes = text_in_bytes.replace(b"&Egrave;", b"\xc3\x88")
        text_in_bytes = text_in_bytes.replace(b"&Eacute;", b"\xc3\x89")
        text_in_bytes = text_in_bytes.replace(b"&Ecirc;", b"\xc3\x8A")
        text_in_bytes = text_in_bytes.replace(b"&Euml;", b"\xc3\x8B")
        text_in_bytes = text_in_bytes.replace(b"&Igrave;", b"\xc3\x8C")
        text_in_bytes = text_in_bytes.replace(b"&Iacute;", b"\xc3\x8D")
        text_in_bytes = text_in_bytes.replace(b"&Icirc;", b"\xc3\x8E")
        text_in_bytes = text_in_bytes.replace(b"&Iuml;", b"\xc3\x8F")
        text_in_bytes = text_in_bytes.replace(b"&ETH;", b"\xc3\x90")
        text_in_bytes = text_in_bytes.replace(b"&Ntilde;", b"\xc3\x91")
        text_in_bytes = text_in_bytes.replace(b"&Ograve;", b"\xc3\x92")
        text_in_bytes = text_in_bytes.replace(b"&Oacute;", b"\xc3\x93")
        text_in_bytes = text_in_bytes.replace(b"&Ocirc;", b"\xc3\x94")
        text_in_bytes = text_in_bytes.replace(b"&Otilde;", b"\xc3\x95")
        text_in_bytes = text_in_bytes.replace(b"&Ouml;", b"\xc3\x96")
        text_in_bytes = text_in_bytes.replace(b"&#215;", b"\xc3\x97")  # MULTIPLICATION SIGN
        text_in_bytes = text_in_bytes.replace(b"&Oslash;", b"\xc3\x98")
        text_in_bytes = text_in_bytes.replace(b"&Ugrave;", b"\xc3\x99")
        text_in_bytes = text_in_bytes.replace(b"&Uacute;", b"\xc3\x9A")
        text_in_bytes = text_in_bytes.replace(b"&Ucirc;", b"\xc3\x9B")
        text_in_bytes = text_in_bytes.replace(b"&Uuml;", b"\xc3\x9C")
        text_in_bytes = text_in_bytes.replace(b"&Yacute;", b"\xc3\x9D")
        text_in_bytes = text_in_bytes.replace(b"&THORN;", b"\xc3\x9E")
        text_in_bytes = text_in_bytes.replace(b"&agrave;", b"\xc3\xA0")
        text_in_bytes = text_in_bytes.replace(b"&aacute;", b"\xc3\xA1")
        text_in_bytes = text_in_bytes.replace(b"&acirc;", b"\xc3\xA2")
        text_in_bytes = text_in_bytes.replace(b"&atilde;", b"\xc3\xA3")
        text_in_bytes = text_in_bytes.replace(b"&auml;", b"\xc3\xA4")
        text_in_bytes = text_in_bytes.replace(b"&aring;", b"\xc3\xA5")
        text_in_bytes = text_in_bytes.replace(b"&ccedil;", b"\xc3\xA7")
        text_in_bytes = text_in_bytes.replace(b"&egrave;", b"\xc3\xA8")
        text_in_bytes = text_in_bytes.replace(b"&eacute;", b"\xc3\xA9")
        text_in_bytes = text_in_bytes.replace(b"&ecirc;", b"\xc3\xAA")
        text_in_bytes = text_in_bytes.replace(b"&euml;", b"\xc3\xAB")
        text_in_bytes = text_in_bytes.replace(b"&igrave;", b"\xc3\xAC")
        text_in_bytes = text_in_bytes.replace(b"&iacute;", b"\xc3\xAD")
        text_in_bytes = text_in_bytes.replace(b"&icirc;", b"\xc3\xAE")
        text_in_bytes = text_in_bytes.replace(b"&iuml;", b"\xc3\xAF")
        text_in_bytes = text_in_bytes.replace(b"&eth;", b"\xc3\xB0")
        text_in_bytes = text_in_bytes.replace(b"&ntilde;", b"\xc3\xB1")
        text_in_bytes = text_in_bytes.replace(b"&ograve;", b"\xc3\xB2")
        text_in_bytes = text_in_bytes.replace(b"&oacute;", b"\xc3\xB3")
        text_in_bytes = text_in_bytes.replace(b"&ocirc;", b"\xc3\xB4")
        text_in_bytes = text_in_bytes.replace(b"&otilde;", b"\xc3\xB5")
        text_in_bytes = text_in_bytes.replace(b"&ouml;", b"\xc3\xB6")
        text_in_bytes = text_in_bytes.replace(b"&#247;", b"\xc3\xB7")  # DIVISION SIGN
        text_in_bytes = text_in_bytes.replace(b"&oslash;", b"\xc3\xB8")
        text_in_bytes = text_in_bytes.replace(b"&ugrave;", b"\xc3\xB9")
        text_in_bytes = text_in_bytes.replace(b"&uacute;", b"\xc3\xBA")
        text_in_bytes = text_in_bytes.replace(b"&ucirc;", b"\xc3\xBB")
        text_in_bytes = text_in_bytes.replace(b"&uuml;", b"\xc3\xBC")
        text_in_bytes = text_in_bytes.replace(b"&yacute;", b"\xc3\xBD")
        text_in_bytes = text_in_bytes.replace(b"&thorn;", b"\xc3\xBE")
        text_in_bytes = text_in_bytes.replace(b"&yuml;", b"\xc3\xBF")

        # Greek Entities for HTML4 and Chadwock Healey -- Charles Cooney
        text_in_bytes = text_in_bytes.replace(b"&agr;", b"\xce\xb1")
        text_in_bytes = text_in_bytes.replace(b"&alpha;", b"\xce\xb1")
        text_in_bytes = text_in_bytes.replace(b"&bgr;", b"\xce\xb2")
        text_in_bytes = text_in_bytes.replace(b"&beta;", b"\xce\xb2")
        text_in_bytes = text_in_bytes.replace(b"&ggr;", b"\xce\xb3")
        text_in_bytes = text_in_bytes.replace(b"&gamma;", b"\xce\xb3")
        text_in_bytes = text_in_bytes.replace(b"&dgr;", b"\xce\xb4")
        text_in_bytes = text_in_bytes.replace(b"&delta;", b"\xce\xb4")
        text_in_bytes = text_in_bytes.replace(b"&egr;", b"\xce\xb5")
        text_in_bytes = text_in_bytes.replace(b"&epsilon;", b"\xce\xb5")
        text_in_bytes = text_in_bytes.replace(b"&zgr;", b"\xce\xb6")
        text_in_bytes = text_in_bytes.replace(b"&zeta;", b"\xce\xb6")
        text_in_bytes = text_in_bytes.replace(b"&eegr;", b"\xce\xb7")
        text_in_bytes = text_in_bytes.replace(b"&eta;", b"\xce\xb7")
        text_in_bytes = text_in_bytes.replace(b"&thgr;", b"\xce\xb8")
        text_in_bytes = text_in_bytes.replace(b"&theta;", b"\xce\xb8")
        text_in_bytes = text_in_bytes.replace(b"&igr;", b"\xce\xb9")
        text_in_bytes = text_in_bytes.replace(b"&iota;", b"\xce\xb9")
        text_in_bytes = text_in_bytes.replace(b"&kgr;", b"\xce\xba")
        text_in_bytes = text_in_bytes.replace(b"&kappa;", b"\xce\xba")
        text_in_bytes = text_in_bytes.replace(b"&lgr;", b"\xce\xbb")
        text_in_bytes = text_in_bytes.replace(b"&lambda;", b"\xce\xbb")
        text_in_bytes = text_in_bytes.replace(b"&mgr;", b"\xce\xbc")
        text_in_bytes = text_in_bytes.replace(b"&mu;", b"\xce\xbc")
        text_in_bytes = text_in_bytes.replace(b"&ngr;", b"\xce\xbd")
        text_in_bytes = text_in_bytes.replace(b"&nu;", b"\xce\xbd")
        text_in_bytes = text_in_bytes.replace(b"&xgr;", b"\xce\xbe")
        text_in_bytes = text_in_bytes.replace(b"&xi;", b"\xce\xbe")
        text_in_bytes = text_in_bytes.replace(b"&ogr;", b"\xce\xbf")
        text_in_bytes = text_in_bytes.replace(b"&omicron;", b"\xce\xbf")
        text_in_bytes = text_in_bytes.replace(b"&pgr;", b"\xcf\x80")
        text_in_bytes = text_in_bytes.replace(b"&pi;", b"\xcf\x80")
        text_in_bytes = text_in_bytes.replace(b"&rgr;", b"\xcf\x81")
        text_in_bytes = text_in_bytes.replace(b"&rho;", b"\xcf\x81")
        text_in_bytes = text_in_bytes.replace(b"&sfgr;", b"\xcf\x82")
        text_in_bytes = text_in_bytes.replace(b"&sigmaf;", b"\xcf\x82")
        text_in_bytes = text_in_bytes.replace(b"&sgr;", b"\xcf\x83")
        text_in_bytes = text_in_bytes.replace(b"&sigma;", b"\xcf\x83")
        text_in_bytes = text_in_bytes.replace(b"&tgr;", b"\xcf\x84")
        text_in_bytes = text_in_bytes.replace(b"&tau;", b"\xcf\x84")
        text_in_bytes = text_in_bytes.replace(b"&ugr;", b"\xcf\x85")
        text_in_bytes = text_in_bytes.replace(b"&upsilon;", b"\xcf\x85")
        text_in_bytes = text_in_bytes.replace(b"&phgr;", b"\xcf\x86")
        text_in_bytes = text_in_bytes.replace(b"&phi;", b"\xcf\x86")
        text_in_bytes = text_in_bytes.replace(b"&khgr;", b"\xcf\x87")
        text_in_bytes = text_in_bytes.replace(b"&chi;", b"\xcf\x87")
        text_in_bytes = text_in_bytes.replace(b"&psgr;", b"\xcf\x88")
        text_in_bytes = text_in_bytes.replace(b"&psi;", b"\xcf\x88")
        text_in_bytes = text_in_bytes.replace(b"&ohgr;", b"\xcf\x89")
        text_in_bytes = text_in_bytes.replace(b"&omega;", b"\xcf\x89")
        text_in_bytes = text_in_bytes.replace(b"&Agr;", b"\xce\x91")
        text_in_bytes = text_in_bytes.replace(b"&Alpha;", b"\xce\x91")
        text_in_bytes = text_in_bytes.replace(b"&Bgr;", b"\xce\x92")
        text_in_bytes = text_in_bytes.replace(b"&Beta;", b"\xce\x92")
        text_in_bytes = text_in_bytes.replace(b"&Ggr;", b"\xce\x93")
        text_in_bytes = text_in_bytes.replace(b"&Gamma;", b"\xce\x93")
        text_in_bytes = text_in_bytes.replace(b"&Dgr;", b"\xce\x94")
        text_in_bytes = text_in_bytes.replace(b"&Delta;", b"\xce\x94")
        text_in_bytes = text_in_bytes.replace(b"&Egr;", b"\xce\x95")
        text_in_bytes = text_in_bytes.replace(b"&Epsilon;", b"\xce\x95")
        text_in_bytes = text_in_bytes.replace(b"&Zgr;", b"\xce\x96")
        text_in_bytes = text_in_bytes.replace(b"&Zeta;", b"\xce\x96")
        text_in_bytes = text_in_bytes.replace(b"&EEgr;", b"\xce\x97")
        text_in_bytes = text_in_bytes.replace(b"&Eta;", b"\xce\x97")
        text_in_bytes = text_in_bytes.replace(b"&THgr;", b"\xce\x98")
        text_in_bytes = text_in_bytes.replace(b"&Theta;", b"\xce\x98")
        text_in_bytes = text_in_bytes.replace(b"&Igr;", b"\xce\x99")
        text_in_bytes = text_in_bytes.replace(b"&Iota;", b"\xce\x99")
        text_in_bytes = text_in_bytes.replace(b"&Kgr;", b"\xce\x9a")
        text_in_bytes = text_in_bytes.replace(b"&Kappa;", b"\xce\x9a")
        text_in_bytes = text_in_bytes.replace(b"&Lgr;", b"\xce\x9b")
        text_in_bytes = text_in_bytes.replace(b"&Lambda;", b"\xce\x9b")
        text_in_bytes = text_in_bytes.replace(b"&Mgr;", b"\xce\x9c")
        text_in_bytes = text_in_bytes.replace(b"&Mu;", b"\xce\x9c")
        text_in_bytes = text_in_bytes.replace(b"&Ngr;", b"\xce\x9d")
        text_in_bytes = text_in_bytes.replace(b"&Nu;", b"\xce\x9d")
        text_in_bytes = text_in_bytes.replace(b"&Xgr;", b"\xce\x9e")
        text_in_bytes = text_in_bytes.replace(b"&Xi;", b"\xce\x9e")
        text_in_bytes = text_in_bytes.replace(b"&Ogr;", b"\xce\x9f")
        text_in_bytes = text_in_bytes.replace(b"&Omicron;", b"\xce\x9f")
        text_in_bytes = text_in_bytes.replace(b"&Pgr;", b"\xce\xa0")
        text_in_bytes = text_in_bytes.replace(b"&Pi;", b"\xce\xa0")
        text_in_bytes = text_in_bytes.replace(b"&Rgr;", b"\xce\xa1")
        text_in_bytes = text_in_bytes.replace(b"&Rho;", b"\xce\xa1")
        text_in_bytes = text_in_bytes.replace(b"&Sgr;", b"\xce\xa3")
        text_in_bytes = text_in_bytes.replace(b"&Sigma;", b"\xce\xa3")
        text_in_bytes = text_in_bytes.replace(b"&Tgr;", b"\xce\xa4")
        text_in_bytes = text_in_bytes.replace(b"&Tau;", b"\xce\xa4")
        text_in_bytes = text_in_bytes.replace(b"&Ugr;", b"\xce\xa5")
        text_in_bytes = text_in_bytes.replace(b"&Upsilon;", b"\xce\xa5")
        text_in_bytes = text_in_bytes.replace(b"&PHgr;", b"\xce\xa6")
        text_in_bytes = text_in_bytes.replace(b"&Phi;", b"\xce\xa6")
        text_in_bytes = text_in_bytes.replace(b"&KHgr;", b"\xce\xa7")
        text_in_bytes = text_in_bytes.replace(b"&Chi;", b"\xce\xa7")
        text_in_bytes = text_in_bytes.replace(b"&PSgr;", b"\xce\xa8")
        text_in_bytes = text_in_bytes.replace(b"&Psi;", b"\xce\xa8")
        text_in_bytes = text_in_bytes.replace(b"&OHgr;", b"\xce\xa9")
        text_in_bytes = text_in_bytes.replace(b"&Omega;", b"\xce\xa9")
        return text_in_bytes.decode("utf8")

    def convert_other_ents(self, text):
        """handles character entities in index words.
        There should not be many of these."""
        text = text.replace("&apos;", "'")
        text = text.replace("&s;", "s")
        text = macr_ent.sub(r"\1", text)
        text = inverted_ent.sub(r"\1", text)
        text = supp_ent.sub(r"\1", text)
        if self.flatten_ligatures:
            text = ligatures_ent.sub(r"\1", text)
        return text

    def remove_control_chars(self, text):
        if text.isprintable():  # control characters aren't printable: nothing to remove
            return text
        return control_char_re.sub("", text)


if __name__ == "__main__":
    if os.path.isfile(sys.argv[1]):
        files = sys.argv[1:]
    else:
        import glob

        files = glob.glob(sys.argv[1] + "/*")
    for docid, fn in enumerate(files, 1):
        print(docid, fn, file=sys.stderr)
        size = os.path.getsize(fn)
        fh = open(fn, encoding="utf8")
        parser = XMLParser(
            sys.stdout,
            docid,
            size,
            known_metadata={"filename": fn},
            tag_to_obj_map=DEFAULT_TAG_TO_OBJ_MAP,
            metadata_to_parse=DEFAULT_METADATA_TO_PARSE,
            metadata_sql_types={},
            token_regex=r"[\p{L}\p{M}\p{N}]+|[&\p{L};]+",
        )
        parser.parse(fh)
