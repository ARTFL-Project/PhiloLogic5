"""spaCy's tokenizers, which parsers split words with when a load sets a language (see XMLParser.flush_run), and what
they need to give these words their exact bytes"""

import functools
from bisect import bisect_right

import regex as re

# Characters a word keeps at its edges: those of token_regex (letters, marks and numbers), and the private use
# character XMLParser gives the tokenizer for an unknown entity
WORD_EDGE_CHARACTER = re.compile(rf"[\p{{L}}\p{{M}}\p{{N}}{chr(0xE000)}]")

# Characters which can join the words of a token: hyphens (peut-être, Saint-Just) and apostrophes (aujourd'hui), the "-"
# and "'" the query_parser_regex of web configs splits, and the typographic ones Query.query_parse makes "-" and "'"
# (tokenizer_input makes the apostrophes ’)
WORD_JOINER = re.compile("[%s]" % re.escape("".join(map(chr, (0x2D, 0x2010, 0x2011, 0x2012, 0x2013, 0x27, 0x2019)))))

# Typographic apostrophes the tokenizers don't take as one (qu‘il), as the one they do (qu’il)
APOSTROPHES = str.maketrans({chr(0x2018): chr(0x2019), chr(0x02BC): chr(0x2019)})


@functools.cache
def spacy_tokenizer(language, spacy_model=None):
    """The tokenizer of spacy_model (the one its tagging expects), or else of spaCy's blank model for language: one per
    process, loaded the first time it's asked for. Raises ImportError when spaCy has no tokenizer for the language, or
    lacks the libraries it needs (Japanese needs SudachiPy...), and OSError when spacy_model can't be found."""
    import spacy  # only imported when used: importing spacy takes about a second
    from spacy import util

    if spacy_model:
        path = util.get_package_path(spacy_model) if util.is_package(spacy_model) else spacy_model
        meta = util.get_model_meta(path)
        nlp = spacy.load(spacy_model, exclude=meta.get("components") or meta.get("pipeline", []))
    else:
        nlp = spacy.blank(language)
    return nlp.tokenizer


def token_spans(tokenizer, text):
    """(start, end) of each token spaCy's tokenizer finds in text, as character positions"""
    from spacy.attrs import IDX, LENGTH

    return [(start, start + length) for start, length in tokenizer(text).to_array([IDX, LENGTH]).tolist()]


ON_THEIR_OWN = {}  # (tokenizer, word) -> the spans of the tokens it makes of the word on its own


def spans_on_its_own(tokenizer, word):
    """(start, end) of the tokens the tokenizer makes of word on its own, in lowercase (the form searches look up)"""
    key = (id(tokenizer), word)
    spans = ON_THEIR_OWN.get(key)
    if spans is None:
        lowercase = word.lower()
        spans = ON_THEIR_OWN[key] = token_spans(tokenizer, lowercase if len(lowercase) == len(word) else word)
    return spans


def trimmed_span(text, start, end):
    """(start, end) of the token text[start:end] without the characters at its edges which can't be at a word's (see
    WORD_EDGE_CHARACTER), as the punctuation of l' or -il: an empty span if it has none of them"""
    while start < end and not WORD_EDGE_CHARACTER.match(text, start):
        start += 1
    while end > start and not WORD_EDGE_CHARACTER.match(text, end - 1):
        end -= 1
    return start, end


class TextBytes:
    """Bytes in the file of the positions of a text made of the lines of a run (see XMLParser.flush_run), with tags
    between them which it leaves out. Positions have to be asked for in increasing order."""

    def __init__(self, pieces):
        """pieces: (start byte, text) of the lines, in order, none empty"""
        self.pieces = pieces
        self.ascii = [line.isascii() for _, line in pieces]
        self.positions = []  # of the start of each line in the text
        position = 0
        for _, line in pieces:
            self.positions.append(position)
            position += len(line)
        self.line = -1  # the line, and the position in it and its byte, last asked for
        self.offset = self.byte = 0

    def byte_at(self, line, offset):
        """The byte of position offset of a line"""
        start_byte, text = self.pieces[line]
        if line != self.line:
            self.line, self.offset, self.byte = line, 0, start_byte
        if self.ascii[line]:
            self.byte = start_byte + offset
        else:
            self.byte += len(text[self.offset : offset].encode("utf8"))
        self.offset = offset
        return self.byte

    def start(self, position):
        """Byte where the character at position starts"""
        line = bisect_right(self.positions, position) - 1
        return self.byte_at(line, position - self.positions[line])

    def end(self, position):
        """Byte where the character before position ends"""
        line = bisect_right(self.positions, position - 1) - 1
        return self.byte_at(line, position - self.positions[line])


def tokenizer_input(text, separators=None):
    """text as the tokenizer gets it, of the same length: what separators matches (the chars_not_to_index of a load,
    as [ and ]) as spaces, so that it splits words as token_regex does ([L']Homme), and typographic apostrophes as one
    the tokenizers take as such"""
    if separators is not None:
        text = separators.sub(lambda match: " " * len(match.group()), text)
    return text.translate(APOSTROPHES)


def split_words(tokenizer, text, is_word, tokenizer_text=None, starts=None, ends=None):
    """The words the tokenizer finds in text, in order: (start, end) of their token, and of the word itself, which
    leaves out the punctuation at the edges of the token (as l' or -il has), when is_word(word) is true, and the text of
    the token as the tokenizer has it (which a spaCy model tags). A word with a hyphen or apostrophe inside it stays
    whole (peut-être, aujourd'hui) if the tokenizer keeps it whole on its own, else it's split as the tokenizer splits
    it on its own: searches keep such a word whole when the index has it (see Query.query_parse), and split it
    otherwise. The tokenizer splits tokenizer_text rather than text, if given (of the same length, or with starts and
    ends giving, for each of its characters, the start and end in text of what it stands for: see
    XMLParser.tokenizer_text)."""
    if tokenizer_text is None:
        tokenizer_text = text
    words = []
    last_end = 0
    for token_start, token_end in token_spans(tokenizer, tokenizer_text):
        word_start, word_end = trimmed_span(tokenizer_text, token_start, token_end)
        if word_start == word_end:  # punctuation, spaces...
            continue
        tokens = [(token_start, token_end)]
        if WORD_JOINER.search(tokenizer_text, word_start, word_end):
            alone = spans_on_its_own(tokenizer, tokenizer_text[word_start:word_end])
            if len(alone) > 1:  # (the tokens keep the punctuation at the edges of the one they split)
                tokens = [(word_start + start, word_start + end) for start, end in alone]
                tokens[0], tokens[-1] = (token_start, tokens[0][1]), (tokens[-1][0], token_end)
        for token_start, token_end in tokens:
            word_start, word_end = trimmed_span(tokenizer_text, token_start, token_end)
            if word_start == word_end:
                continue
            token = tokenizer_text[token_start:token_end]
            if starts is not None:  # (characters something of several stands for, the first gets)
                token_start, token_end = max(starts[token_start], last_end), ends[token_end - 1]
                word_start, word_end = max(starts[word_start], token_start), ends[word_end - 1]
                if word_start >= word_end:
                    continue
            if is_word(text[word_start:word_end]):
                words.append((token_start, token_end, word_start, word_end, token))
                last_end = token_end
    return words
