"""spaCy's tokenizers, which parsers split words with when a load sets a language (see XMLParser.flush_run), and what
they need to give these words their exact bytes"""

import functools
from bisect import bisect_right

import regex as re

# Characters a word keeps at its edges: those of token_regex (letters, marks and numbers), and the private use
# character XMLParser gives the tokenizer for an unknown entity
WORD_EDGE_CHARACTER = re.compile(rf"[\p{{L}}\p{{M}}\p{{N}}{chr(0xE000)}]")

# Characters at which the words of a token split, as searches split them: hyphens (peut-être, Saint-Just) and
# apostrophes (aujourd'hui), the "-" and "'" of the query_parser_regex of web configs, and the typographic ones
# Query.query_parse makes "-" and "'" (tokenizer_input makes the apostrophes ’)
HYPHENS = "".join(chr(c) for c in (0x2D, 0x2010, 0x2011, 0x2012, 0x2013))
APOSTROPHE_CHARACTERS = "'" + chr(0x2019)


def word_breaks(apostrophes=True):
    """A regex of the characters at which the words of a token split: hyphens, and apostrophes if they split words (as
    token_regex splits them, with break_apost)"""
    return re.compile("[%s]+" % re.escape(HYPHENS + (APOSTROPHE_CHARACTERS if apostrophes else "")))


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


def split_words(tokenizer, text, is_word, breaks, tokenizer_text=None, starts=None, ends=None):
    """The tokens with words the tokenizer finds in text, in order: (start, end) of each, (start, end) of each of its
    words, and its text as the tokenizer has it (which a spaCy model tags). Its words are the token without the
    punctuation at its edges (as l' or -il has), split at what breaks matches (see word_breaks: peut-être and
    aujourd'hui have two words), those for which is_word(word) is true. The tokenizer splits tokenizer_text rather
    than text, if given (of the same length, or with starts and ends giving, for each of its characters, the start and
    end in text of what it stands for: see XMLParser.tokenizer_text)."""
    if tokenizer_text is None:
        tokenizer_text = text
    tokens = []
    last_end = 0
    for token_start, token_end in token_spans(tokenizer, tokenizer_text):
        word_start, word_end = trimmed_span(tokenizer_text, token_start, token_end)
        if word_start == word_end:  # punctuation, spaces...
            continue
        token = tokenizer_text[token_start:token_end]
        parts, part_start = [], word_start
        for word_break in breaks.finditer(tokenizer_text, word_start, word_end):
            parts.append(trimmed_span(tokenizer_text, part_start, word_break.start()))
            part_start = word_break.end()
        parts.append(trimmed_span(tokenizer_text, part_start, word_end))
        if starts is not None:  # (characters something of several stands for, the first gets)
            token_start, token_end = max(starts[token_start], last_end), ends[token_end - 1]
            mapped, last_part_end = [], token_start
            for start, end in parts:
                if end > start:
                    start, end = max(starts[start], last_part_end), ends[end - 1]
                    if start < end:
                        mapped.append((start, end))
                        last_part_end = end
            parts = mapped
        words = [(start, end) for start, end in parts if end > start and is_word(text[start:end])]
        if words:
            tokens.append((token_start, token_end, words, token))
            last_end = token_end
    return tokens
