"""Ligatures as their letters (œ as oe), as loads with flatten_ligatures index them, and as searches of their
databases then take them"""

# The Latin ligatures of Unicode (the characters it names ligatures, and æ, which it names a letter but SGML's
# entities a ligature, &aelig;), with their letters: the long s of ﬅ as an s, as loads index it
LIGATURE_LETTERS = {
    "Æ": "AE",
    "æ": "ae",
    "Ĳ": "IJ",
    "ĳ": "ij",
    "Œ": "OE",
    "œ": "oe",
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}
_TABLE = str.maketrans(LIGATURE_LETTERS)


def ligatures_as_letters(text):
    """text with its ligatures as their letters (see LIGATURE_LETTERS)"""
    return text if text.isascii() else text.translate(_TABLE)


def has_ligature(text):
    """Whether text has a ligature (see LIGATURE_LETTERS)"""
    return not text.isascii() and any(char in LIGATURE_LETTERS for char in text)
