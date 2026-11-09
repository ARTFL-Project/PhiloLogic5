#!/var/lib/philologic5/philologic_env/bin/python3
"""The page markers of plain texts: the lines which the page_marker_regex of a load makes page breaks of. The plain-text
parser makes pages of them, and the formatters show them as the <pb> of XML texts. And the "&", "<" and ">" of plain
texts, which are text there, not markup: the formatters, which read every text as XML, get them escaped."""

import io
from bisect import bisect_left
from functools import lru_cache
from itertools import accumulate

import regex as re

ESCAPES = {b"&": b"&amp;", b"<": b"&lt;", b">": b"&gt;"}


def escape_plain_text(text):
    """The bytes of a plain text, with its "&", "<" and ">" escaped as an XML text has them, so that the formatters show
    them as they are; and the function giving the position in those of each byte offset of text, where the highlights
    and passages inserted at the byte offsets of the file go: at the same character, never inside an escape."""
    if b"&" not in text and b"<" not in text and b">" not in text:  # most of the contexts of hits
        return text, same_position
    # Found with bytes.find, many times faster than a regex over the 100 kB of a section
    found = sorted(
        (index, len(escape) - 1) for character, escape in ESCAPES.items() for index in _find_all(text, character)
    )
    positions = [index for index, _ in found]
    # The bytes the escapes of the first 0, 1, 2... of them add
    added = list(accumulate((extra for _, extra in found), initial=0))
    text = text.replace(b"&", b"&amp;").replace(b"<", b"&lt;").replace(b">", b"&gt;")

    def position(offset):
        return offset + added[bisect_left(positions, offset)]

    return text, position


def _find_all(text, character):
    index = text.find(character)
    while index != -1:
        yield index
        index = text.find(character, index + 1)


def same_position(offset):
    """The position of a byte offset of a text in the text as it is, not escaped: the same."""
    return offset


def unescape_plain_text(text):
    """A plain text escaped by escape_plain_text (decoded, a str) as it is in the file."""
    if "&" not in text:
        return text
    return text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


@lru_cache(maxsize=16)
def compile_page_marker_regex(page_marker_regex):
    """page_marker_regex compiled, once for all the texts and requests that use it."""
    return re.compile(page_marker_regex)


def page_marker_number(line, marker_regex):
    """The number (n) of the page the line starts if it is a page marker, else None. A page marker is a line which the
    compiled marker_regex matches whole once stripped of the whitespace around it. Its page is numbered by the
    regex's first group as a whole number, without leading zeros, or "na" when that is no number."""
    match = marker_regex.fullmatch(line.strip())
    if match is None:
        return None
    try:
        return str(int(match.group(1)))
    except (IndexError, TypeError, ValueError):
        return "na"


def page_markers_to_pb(text, page_marker_regex, escaped=False):
    """The text, with each of its lines which is a page marker of page_marker_regex (see page_marker_number) made the
    <pb n="..."/> of an XML text, numbered as the parser numbers its page, and the whitespace around it kept. The
    text as it is without a regex. An escaped text (see escape_plain_text) has its lines matched as they are in the
    file."""
    if not page_marker_regex:
        return text
    marker_regex = compile_page_marker_regex(page_marker_regex)
    lines = list(io.StringIO(text, newline=""))  # as the loader reads the lines of its files, with their line breaks
    changed = False
    for index, line in enumerate(lines):
        n = page_marker_number(unescape_plain_text(line) if escaped and "&" in line else line, marker_regex)
        if n is not None:
            body = line.rstrip("\r\n")
            start = len(body) - len(body.lstrip())
            lines[index] = f'{body[:start]}<pb n="{n}"/>{body[start + len(body.strip()):]}{line[len(body):]}'
            changed = True
    return "".join(lines) if changed else text


def cut_page_marker_start(text, page_marker_regex, escaped=False):
    """Where the page marker which the bytes of text end in the middle of starts, as a text cut out of a file may: its
    last line, with no line break after it, which stripped of its whitespace is the start of a page marker of
    page_marker_regex, but no whole one, and no line to page_markers_to_pb. None if text ends in no such line. An
    escaped text (see escape_plain_text) has its last line matched as it is in the file."""
    if not page_marker_regex:
        return None
    line_start = max(text.rfind(b"\n"), text.rfind(b"\r")) + 1
    line = text[line_start:].decode("utf8", "ignore").strip()
    if escaped:
        line = unescape_plain_text(line)
    if not line:
        return None
    match = compile_page_marker_regex(page_marker_regex).fullmatch(line, partial=True)
    if match is None or not match.partial:
        return None
    return line_start
