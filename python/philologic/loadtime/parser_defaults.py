"""The defaults of the parser's options, which load configs can change (see XMLParser and extras/load_config.py): the
objects tags make, the metadata of objects and of documents (in TEI headers), the tags which don't split words, and what
words and punctuation are"""

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

CHARS_NOT_TO_INDEX = r"[\[\{\]\}]"
