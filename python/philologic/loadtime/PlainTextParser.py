#!/var/lib/philologic5/philologic_env/bin/python3

import os
import sys
import unicodedata
from io import TextIOWrapper
from typing import List, Union

import regex as re
from philologic.loadtime import OHCOVector
from philologic.loadtime.Parser import CHARS_NOT_TO_INDEX
from philologic.loadtime.spacy_tokenizer import TextBytes, spacy_tokenizer, split_words, tokenizer_input, word_breaks

TOKEN_REGEX = r"[\p{L}\p{M}\p{N}]+|[&\p{L};]+"
PUNCTUATION = r"""[;,:=+()"]"""

check_if_char_word = re.compile(r"\p{L}", re.I)
cap_char_or_num = re.compile(r"[A-Z0-9]")  # Capitals


class PlainTextParser:
    """Plain tex parser"""

    def __init__(
        self,
        output,
        docid,
        filesize,
        token_regex: str = TOKEN_REGEX,
        xpaths=[("doc", "./")],
        metadata_xpaths=[],
        suppress_tags=[],
        pseudo_empty_tags=[],
        known_metadata={},
        words_to_index=None,
        paragraph_on_newline=False,  # TODO: make this configurable from load_config.py
        lemmas=None,
        **parse_options,
    ):
        self.types = ["doc", "div1", "div2", "div3", "para", "sent", "word"]
        self.parallel_type = "page"
        self.output = output
        self.docid = docid
        ## Initialize an OHCOVector Stack. operations on this stack produce all parser output.
        self.v = OHCOVector.CompoundStack(
            self.types,
            self.parallel_type,
            docid,
            output,
            ref="ref",
            line="line",
            graphic="graphic",
            punctuation="punct",
        )

        self.filesize = filesize
        self.token_regex = re.compile(rf"({token_regex})", re.I | re.M)
        if "sentence_breakers" in parse_options:
            self.sentence_breakers = parse_options["sentence_breakers"]
        else:
            self.sentence_breakers = []
        if "punctuation" in parse_options:
            self.punct_regex = re.compile(rf"""{parse_options["punctuation"]}""")
        else:
            self.punct_regex = re.compile(rf"{PUNCTUATION}")

        self.paragraph_on_newline = paragraph_on_newline

        self.xpaths = xpaths[:]
        self.metadata_xpaths = metadata_xpaths[:]

        self.suppress_xpaths = suppress_tags
        self.pseudo_empty_tags = pseudo_empty_tags
        self.known_metadata = known_metadata

        self.buffer_position = 0
        self.buffers = []

        self.lemmas = lemmas

        # With a language, words are split by spaCy's tokenizer for it rather than by token_regex (the loader only
        # gives a language spaCy has a tokenizer for): see index_tokens
        # With a spaCy model, which tags the words as they are in the text: words keep their form, before it's made the
        # indexed one, in their spacy_token attribute
        self.spacy_model = parse_options.get("spacy_model")
        self.tokenizer = None
        if parse_options.get("language"):
            self.tokenizer = spacy_tokenizer(parse_options["language"], parse_options.get("spacy_model"))
            # Characters not to index split words for the tokenizer, as they do for token_regex
            self.chars_not_to_index = re.compile(parse_options.get("chars_not_to_index", CHARS_NOT_TO_INDEX), re.I)
            self.word_breaks = word_breaks(parse_options.get("break_apost", True))

        if words_to_index:
            self.words_to_index = words_to_index
            self.defined_words_to_index = True
        else:
            self.defined_words_to_index = False

    def parse(self, input_file: TextIOWrapper):
        """Top level function for reading a file and printing out the output."""
        bytes_read_in: int = 0
        # Begin by creating a document level object, just call it "text" for now.
        self.v.push("doc", "text", bytes_read_in)
        self.v["doc"].attrib["philo_doc_id"] = str(self.v["doc"].id[0])
        for k, v in list(self.known_metadata.items()):
            # if the parser was created with known_metadata,
            # we can attach it to the newly created doc object here.
            # you can attach metadata to an object at any time between push() and pull().
            self.v["doc"][k] = v
        open_para: bool = False
        previous_line_empty = False
        for line in input_file:  # TODO: recognize divs on lines with all characters in CAPS?
            if line == "\n" or not check_if_char_word.search(line):
                bytes_read_in += len(line.encode("utf8"))
                previous_line_empty = True
                continue
            if previous_line_empty is True or self.paragraph_on_newline is True:
                if open_para is True:
                    self.v.pull("para", bytes_read_in)
                self.v.push("para", "p", bytes_read_in)  # we assume a new paragraph for each line break
                open_para = True
            previous_line_empty = False
            if self.tokenizer is not None:
                self.index_tokens(line, bytes_read_in, input_file)
                bytes_read_in += len(line.encode("utf8"))
                continue
            start_byte = bytes_read_in
            last_word = ""
            words: List[Union[str, None]] = self.token_regex.split(line)
            for count, word in enumerate(words):
                is_sent = False
                if word is None:
                    continue
                word_in_utf8 = word.encode("utf8")
                word_length = len(word_in_utf8)
                end_byte = start_byte + word_length
                if check_if_char_word.search(word):
                    original = word
                    # Switch everything to lower case
                    word = word.lower()
                    last_word = word
                    # Check to see if the word is longer than we want. More than 235 characters appear to cause problems in the indexer.
                    if len(word_in_utf8) > 200:
                        print(f"Long word in {input_file.name}: {word}", file=sys.stderr)
                        print("Removing word from for index since over 200 bytes long...", file=sys.stderr)
                        start_byte += len(word_in_utf8)
                        continue

                    word = "".join(c for c in word if unicodedata.category(c)[0] != "C")  # remove control characters
                    word = word.replace("_", "").strip()
                    word = word.replace(" ", "")
                    if word:
                        if self.defined_words_to_index is True and word not in self.words_to_index:
                            start_byte += len(word_in_utf8)
                            continue
                        token = original if self.spacy_model and original != word else None
                        self.push_word(word, start_byte, end_byte, token)
                        start_byte += len(word_in_utf8)
                        continue
                has_next = count + 1 < len(words)
                is_sent = self.is_sentence_end(word, last_word, has_next, words[count + 1] if has_next else None)
                self.index_between_words(word, start_byte, end_byte, is_sent)
                start_byte += len(word_in_utf8)
            bytes_read_in += len(line.encode("utf8"))
        self.v.pull("doc", self.filesize)

    def push_word(self, word, start_byte, end_byte, token=None):
        """Store a word, with its lemma, and its token (as a spaCy model tags it), if given"""
        self.v.push("word", word, start_byte)
        if self.lemmas is not None:
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

    def is_sentence_end(self, word, last_word, has_next, next_word):
        """Whether what's between two words ends a sentence, after last_word and before next_word, if it has_next"""
        is_sent = False
        # Break on ! and ?, unless a word in lowercase follows, which goes on with the sentence: an interjection
        # (Ah! mon Dieu!) or the words introducing what was said (« Viens-tu ? » dit-il)
        if "!" in word or "?" in word:
            is_sent = not (has_next and next_word is not None and next_word.islower())

        # Periods are messy. Let's try by length of previous word and
        # capital letters to avoid hitting abbreviations.
        elif "." in word:
            is_sent = True
            if len(last_word) < 3:
                if cap_char_or_num.search(last_word):
                    is_sent = False

            # Periods in numbers don't break sentences.
            if has_next and (next_word is None or next_word.islower() or next_word.isdigit()):
                is_sent = False
        elif word in self.sentence_breakers:
            is_sent = True
        return is_sent

    @staticmethod
    def index_form(word):
        """The form under which a word is indexed: in lowercase, without control characters, "_" and spaces"""
        word = word.lower()
        word = "".join(c for c in word if unicodedata.category(c)[0] != "C")  # remove control characters
        word = word.replace("_", "").strip()
        return word.replace(" ", "")

    def index_tokens(self, line, start_byte, input_file):
        """Index the words of a line starting at start_byte, split by the tokenizer, and what's between them split by
        token_regex, each with its exact bytes in the file. Words leave out the punctuation at the edges of their
        token (as l' or -il has), and split at its hyphens and apostrophes (with break_apost: peut-être and aujourd'hui
        have two words). The first keeps the token, as it is in the text, in its spacy_token attribute when it isn't
        the indexed word, the others have it empty (for the tagger, see LoadFilters.spacy_tagger)."""
        text_bytes = TextBytes([(start_byte, line)])
        items = []  # (text, start byte, end byte, whether it's a word, token)
        position = 0

        def add_items_between_words(end):
            nonlocal position
            for item in self.token_regex.split(line[position:end]) if end > position else ():
                if item:
                    items.append((item, text_bytes.start(position), text_bytes.end(position + len(item)), False, None))
                    position += len(item)

        tokenizer_text = tokenizer_input(line, self.chars_not_to_index)
        for token_start, token_end, parts, token_text in split_words(
            self.tokenizer, line, check_if_char_word.search, self.word_breaks, tokenizer_text
        ):
            add_items_between_words(token_start)
            for number, (word_start, word_end) in enumerate(parts):
                token = "" if number else token_text  # the words of a token after its first: their token is the first's
                word_bytes = (text_bytes.start(word_start), text_bytes.end(word_end))
                items.append((line[word_start:word_end], *word_bytes, True, token))
            position = token_end
        add_items_between_words(len(line))

        last_word = ""
        for count, (text, start, end, is_word, token) in enumerate(items):
            if is_word:
                last_word = text.lower()
                if len(text.encode("utf8")) > 200:
                    print(f"Long word in {input_file.name}: {last_word}", file=sys.stderr)
                    print("Removing word from for index since over 200 bytes long...", file=sys.stderr)
                    continue
                word = self.index_form(text)
                if word and not (self.defined_words_to_index is True and word not in self.words_to_index):
                    self.push_word(word, start, end, token if token != word else None)
                continue
            has_next = count + 1 < len(items)
            is_sent = self.is_sentence_end(text, last_word, has_next, items[count + 1][0] if has_next else None)
            self.index_between_words(text, start, end, is_sent)

    def index_between_words(self, text, start_byte, end_byte, is_sent):
        """Index what's between two words, from start_byte to end_byte: the end of a sentence, or else its punctuation,
        each with the exact bytes of its characters in the file, without the spaces around them"""
        stripped = text.strip()
        if not stripped:
            return
        start_byte += len(text[: len(text) - len(text.lstrip())].encode("utf8"))
        end_byte -= len(text[len(text.rstrip()) :].encode("utf8"))
        if is_sent:
            # a little hack--we don't know the punctuation mark that will end a sentence
            # until we encounter it--so instead, we let the push on "word" create a
            # implicit philo_virtual sentence, then change its name once we actually encounter
            # the punctuation token.
            name = text.replace("\t", " ").strip()
            if "sent" not in self.v:
                self.v.push("sent", name, start_byte)
            self.v["sent"].name = name
            self.v.pull("sent", end_byte)
        elif self.punct_regex.search(text):
            for character in stripped:
                character_end = start_byte + len(character.encode("utf8"))
                if character not in " \t" and unicodedata.category(character)[0] != "C":  # (control characters)
                    self.v.push("punct", character, start_byte)
                    self.v.pull("punct", character_end)
                start_byte = character_end


if __name__ == "__main__":
    for docid, fn in enumerate(sys.argv[1:], 1):
        print(docid, fn, file=sys.stderr)
        size = os.path.getsize(fn)
        fh = open(fn, encoding="utf8")
        parser = PlainTextParser(
            sys.stdout,
            docid,
            size,
            known_metadata={"filename": fn},
            paragraph_on_newline=False,
        )
        parser.parse(fh)
