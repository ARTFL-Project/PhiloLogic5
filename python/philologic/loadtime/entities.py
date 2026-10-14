"""The character entities of texts, as the parser converts them in words (convert_entities) and blanks out the others
which stand for punctuation or symbols (clear_char_ents)"""

from html.entities import name2codepoint

import regex as re

# Entities standing for letters, which words get as these letters: the Latin-1 letters of HTML, multiplication and
# division signs, and Greek letters, by their HTML names or the SGML ones of Chadwyck-Healey (&agr; for &alpha;)
LATIN1_LETTERS = (
    "Agrave Aacute Acirc Atilde Auml Aring Ccedil Egrave Eacute Ecirc Euml Igrave Iacute Icirc Iuml ETH Ntilde Ograve "
    "Oacute Ocirc Otilde Ouml Oslash Ugrave Uacute Ucirc Uuml Yacute THORN agrave aacute acirc atilde auml aring ccedil "
    "egrave eacute ecirc euml igrave iacute icirc iuml eth ntilde ograve oacute ocirc otilde ouml oslash ugrave uacute "
    "ucirc uuml yacute thorn yuml"
).split()
GREEK_LETTERS = (  # SGML name, HTML name
    "agr alpha bgr beta ggr gamma dgr delta egr epsilon zgr zeta eegr eta thgr theta igr iota kgr kappa lgr lambda "
    "mgr mu ngr nu xgr xi ogr omicron pgr pi rgr rho sfgr sigmaf sgr sigma tgr tau ugr upsilon phgr phi khgr chi "
    "psgr psi ohgr omega Agr Alpha Bgr Beta Ggr Gamma Dgr Delta Egr Epsilon Zgr Zeta EEgr Eta THgr Theta Igr Iota "
    "Kgr Kappa Lgr Lambda Mgr Mu Ngr Nu Xgr Xi Ogr Omicron Pgr Pi Rgr Rho Sgr Sigma Tgr Tau Ugr Upsilon PHgr Phi "
    "KHgr Chi PSgr Psi OHgr Omega"
).split()
LIGATURES = ("AElig", "szlig", "aelig")  # converted with flatten_ligatures only

LETTERS = {name: chr(name2codepoint[name]) for name in LATIN1_LETTERS}
LETTERS.update({"#215": chr(215), "#247": chr(247)})
for sgml_name, html_name in zip(GREEK_LETTERS[0::2], GREEK_LETTERS[1::2]):
    LETTERS[sgml_name] = LETTERS[html_name] = chr(name2codepoint[html_name])
LETTERS_AND_LIGATURES = {**LETTERS, **{name: chr(name2codepoint[name]) for name in LIGATURES}}
LETTER_ENTITY = re.compile(r"&([A-Za-z0-9#]+);")

# Entities standing for punctuation and symbols, which clear_char_ents makes spaces
BLANKED_ENTITIES = re.compile(
    r"\&(?:%s);"
    % "|".join(
        "space mdash nbsp para sect ast commat ldquo laquo rdquo raquo lsquo rsquo quot sup[0-9]* amp deg ndash copy "
        "gt lt frac[0-9]* pound colon hyphen dash excl dagger ddagger times blank dollar cent verbar quest hellip "
        "percnt middot plusmn sqrt sol sdash equals ornament rule prime rsqb lsqb punc cross diamond lpunctel lsemicol "
        "plus minus ounce rindx lindx leaf radic dram sun".split()
    ),
    re.I,
)
macr_ent = re.compile(r"\&([A-Za-z])macr;", re.I)
inverted_ent = re.compile(r"\&inverted([a-zA-Z0-9]);", re.I)
supp_ent = re.compile(r"&supp([a-z0-9]);", re.I)
ligatures_ent = re.compile(r"\&([A-Za-z][A-Za-z])lig;", re.I)


def clear_char_ents(text):
    """text with the entities standing for punctuation and symbols replaced by spaces, which keep the byte count"""
    return BLANKED_ENTITIES.sub(lambda match: " " * len(match.group()), text)


def latin1_ents_to_utf8(text, flatten_ligatures):
    """text with the entities standing for letters (see LETTERS) as these letters, for the index's words: for SGML data
    sets and XML which have entities rather than UTF-8 characters"""
    letters = LETTERS_AND_LIGATURES if flatten_ligatures else LETTERS
    return LETTER_ENTITY.sub(lambda match: letters.get(match.group(1), match.group()), text)


def convert_other_ents(text, flatten_ligatures):
    """text with the other entities of index words converted: there should not be many of these"""
    text = text.replace("&apos;", "'")
    text = text.replace("&s;", "s")
    text = macr_ent.sub(r"\1", text)
    text = inverted_ent.sub(r"\1", text)
    text = supp_ent.sub(r"\1", text)
    if flatten_ligatures:
        text = ligatures_ent.sub(r"\1", text)
    return text


def convert_entities(text, flatten_ligatures):
    """text with entities converted as in index words"""
    return convert_other_ents(latin1_ents_to_utf8(text, flatten_ligatures), flatten_ligatures)
