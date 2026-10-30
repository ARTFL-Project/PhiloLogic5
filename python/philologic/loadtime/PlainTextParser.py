#!/var/lib/philologic5/philologic_env/bin/python3

import os
import sys
import unicodedata
from io import TextIOWrapper
from typing import List, Union

import regex as re
from philologic.loadtime import OHCOVector

TOKEN_REGEX = r"[\p{L}\p{M}\p{N}]+|[&\p{L};]+"
PUNCTUATION = r"""[;,:=+()"]"""
SECTION_LENGTH = 100_000  # in characters

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
        page_marker_regex=None,
        plain_text_section_length=SECTION_LENGTH,
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

        # Lines which match page_marker_regex are page breaks, numbered by the regex's first group
        if page_marker_regex:
            self.page_marker_regex = re.compile(page_marker_regex)
        else:
            self.page_marker_regex = None
        # Split the text into div1 sections of about section_length characters. 0 or None keeps one virtual div1.
        self.section_length = plain_text_section_length
        self.section_start = 0  # in characters, as section_length
        self.section_pages = []  # page numbers the open section spans

        self.xpaths = xpaths[:]
        self.metadata_xpaths = metadata_xpaths[:]

        self.suppress_xpaths = suppress_tags
        self.pseudo_empty_tags = pseudo_empty_tags
        self.known_metadata = known_metadata

        self.buffer_position = 0
        self.buffers = []

        self.lemmas = lemmas

        if words_to_index:
            self.words_to_index = words_to_index
            self.defined_words_to_index = True
        else:
            self.defined_words_to_index = False

    def parse(self, input_file: TextIOWrapper):
        """Top level function for reading a file and printing out the output."""
        bytes_read_in: int = 0
        chars_read_in: int = 0  # for the length of sections
        # Begin by creating a document level object, just call it "text" for now.
        self.v.push("doc", "text", bytes_read_in)
        self.v["doc"].attrib["philo_doc_id"] = str(self.v["doc"].id[0])
        for k, v in list(self.known_metadata.items()):
            # if the parser was created with known_metadata,
            # we can attach it to the newly created doc object here.
            # you can attach metadata to an object at any time between push() and pull().
            self.v["doc"][k] = v
        if self.section_length:
            self.open_section(bytes_read_in, chars_read_in)
        open_para: bool = False
        previous_line_empty = False
        for line in input_file:  # TODO: recognize divs on lines with all characters in CAPS?
            if self.page_marker_regex is not None:
                page_marker = self.page_marker_regex.fullmatch(line.strip())
                if page_marker is not None:  # a page break: not indexed, and the end of the paragraph
                    self.v.pull("para", bytes_read_in)
                    open_para = False
                    self.page_break(page_marker, bytes_read_in, chars_read_in)
                    bytes_read_in += len(line.encode("utf8"))
                    chars_read_in += len(line)
                    previous_line_empty = True
                    continue
            if line == "\n" or not check_if_char_word.search(line):
                bytes_read_in += len(line.encode("utf8"))
                chars_read_in += len(line)
                previous_line_empty = True
                continue
            if previous_line_empty is True or self.paragraph_on_newline is True:
                if open_para is True:
                    self.v.pull("para", bytes_read_in)
                # A section which got this long without a page break ends with the paragraph
                if self.section_length and chars_read_in - self.section_start >= 2 * self.section_length:
                    self.close_section(bytes_read_in)
                    self.open_section(bytes_read_in, chars_read_in)
                self.v.push("para", "p", bytes_read_in)  # we assume a new paragraph for each line break
                open_para = True
            previous_line_empty = False
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
                    # Switch everything to lower case
                    word = word.lower()
                    last_word = word
                    # The long s is an s, as the XML parser has it
                    word = word.replace("ſ", "s")
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
                        self.v.pull("word", end_byte)
                        start_byte += len(word_in_utf8)
                        continue
                if "!" in word or "?" in word:
                    is_sent = True

                # Periods are messy. Let's try by length of previous word and
                # capital letters to avoid hitting abbreviations.
                elif "." in word:
                    is_sent = True
                    if len(last_word) < 3:
                        if cap_char_or_num.search(last_word):
                            is_sent = False

                    # Periods in numbers don't break sentences.
                    try:
                        next_word = words[count + 1]
                        if next_word is None or next_word.islower() or next_word.isdigit():
                            is_sent = False
                    except IndexError:
                        pass
                elif word in self.sentence_breakers:
                    is_sent = True

                if is_sent:
                    # a little hack--we don't know the punctuation mark that will end a sentence
                    # until we encounter it--so instead, we let the push on "word" create a
                    # implicit philo_virtual sentence, then change its name once we actually encounter
                    # the punctuation token.
                    if "sent" not in self.v:
                        self.v.push("sent", word.replace("\t", " ").strip(), start_byte)
                    self.v["sent"].name = word.replace("\t", " ").strip()
                    self.v.pull("sent", start_byte + len(word.encode("utf8")))
                elif self.punct_regex.search(word):
                    punc_pos = start_byte
                    punct = word.strip()
                    punct = punct.replace("\t", " ")
                    punct = "".join(c for c in punct if unicodedata.category(c)[0] != "C")  # remove control characters
                    for single_punct in punct:
                        end_pos = len(single_punct.encode("utf8"))
                        if single_punct != " ":
                            self.v.push("punct", single_punct, punc_pos)
                            self.v.pull("punct", punc_pos + len(single_punct.encode("utf8")))
                        punc_pos = end_pos

                start_byte += len(word_in_utf8)
            bytes_read_in += len(line.encode("utf8"))
            chars_read_in += len(line)
        if self.section_length:
            self.close_section(self.filesize)
        self.v.pull("page", self.filesize)
        self.v.pull("doc", self.filesize)

    def page_break(self, page_marker, start_byte, start_char):
        """Start a new page at a page marker, and a new section if the current one is long enough."""
        try:
            n = str(int(page_marker.group(1)))
        except (IndexError, TypeError, ValueError):
            n = "na"
        new_section = self.section_length and start_char - self.section_start >= self.section_length
        if new_section:
            self.close_section(start_byte)
        self.v.push("page", "pb", start_byte)  # this also ends the previous page
        self.v["page"]["n"] = n
        if new_section:
            self.open_section(start_byte, start_char)  # after the page, which it starts with
        else:
            self.section_pages.append(n)

    def open_section(self, start_byte, start_char):
        """Start a div1 section at start_byte, which is the start_char-th character of the text."""
        self.v.push("div1", "div", start_byte)
        self.section_start = start_char
        if "page" in self.v:
            self.section_pages = [self.v["page"]["n"]]
        else:
            self.section_pages = []

    def close_section(self, end_byte):
        """End the current div1 section, with the pages it spans as its head."""
        if not self.section_pages:
            head = f"Section {self.v['div1'].id[1]}"
        elif self.section_pages[0] == self.section_pages[-1]:
            head = f"Page {self.section_pages[0]}"
        else:
            head = f"Pages {self.section_pages[0]}–{self.section_pages[-1]}"
        self.v["div1"]["head"] = head
        self.v["div1"].attrib["philo_div1_id"] = " ".join(str(i) for i in self.v["div1"].id[:2])
        self.v.pull("div1", end_byte)


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
