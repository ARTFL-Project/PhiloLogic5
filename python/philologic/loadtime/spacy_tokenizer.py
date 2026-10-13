"""spaCy's tokenizers, which parsers split words with when a load sets a language (see XMLParser.flush_run), and what
they need to give these words their exact bytes"""

import functools
from bisect import bisect_right

import regex as re

# Characters a word keeps at its edges: those of token_regex (letters, marks and numbers), and the private use
# character XMLParser gives the tokenizer for an unknown entity
WORD_EDGE_CHARACTER = re.compile(rf"[\p{{L}}\p{{M}}\p{{N}}{chr(0xE000)}]")


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


def split_words(tokenizer, text, is_word, tokenizer_text=None, starts=None, ends=None):
    """The words the tokenizer finds in text, in order: (start, end) of their token, and of the word itself, which
    leaves out the punctuation at the edges of the token (as l' or -il has), when is_word(word) is true. The tokenizer
    can split tokenizer_text rather than text, if starts and ends give, for each of its characters, the start and end in
    text of what it stands for (see XMLParser.tokenizer_text)."""
    if tokenizer_text is None:
        tokenizer_text = text
    words = []
    last_end = 0
    for token_start, token_end in token_spans(tokenizer, tokenizer_text):
        word_start, word_end = trimmed_span(tokenizer_text, token_start, token_end)
        if word_start == word_end:  # punctuation, spaces...
            continue
        if starts is not None:  # (characters something of several stands for, the first gets)
            token_start, token_end = max(starts[token_start], last_end), ends[token_end - 1]
            word_start, word_end = max(starts[word_start], token_start), ends[word_end - 1]
            if word_start >= word_end:
                continue
        if is_word(text[word_start:word_end]):
            words.append((token_start, token_end, word_start, word_end))
            last_end = token_end
    return words
