#!/var/lib/philologic5/philologic_env/bin/python3
"""The page markers of plain texts: the lines which the page_marker_regex of a load makes page breaks of. The plain-text
parser makes pages of them, and the formatters show them as the <pb> of XML texts."""

import io
from functools import lru_cache

import regex as re


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


def page_markers_to_pb(text, page_marker_regex):
    """The text, with each of its lines which is a page marker of page_marker_regex (see page_marker_number) made the
    <pb n="..."/> of an XML text, numbered as the parser numbers its page, and the whitespace around it kept. The
    text as it is without a regex."""
    if not page_marker_regex:
        return text
    marker_regex = compile_page_marker_regex(page_marker_regex)
    lines = list(io.StringIO(text, newline=""))  # as the loader reads the lines of its files, with their line breaks
    changed = False
    for index, line in enumerate(lines):
        n = page_marker_number(line, marker_regex)
        if n is not None:
            body = line.rstrip("\r\n")
            start = len(body) - len(body.lstrip())
            lines[index] = f'{body[:start]}<pb n="{n}"/>{body[start + len(body.strip()):]}{line[len(body):]}'
            changed = True
    return "".join(lines) if changed else text


def cut_page_marker_start(text, page_marker_regex):
    """Where the page marker which the bytes of text end in the middle of starts, as a text cut out of a file may: its
    last line, with no line break after it, which stripped of its whitespace is the start of a page marker of
    page_marker_regex, but no whole one, and no line to page_markers_to_pb. None if text ends in no such line."""
    if not page_marker_regex:
        return None
    line_start = max(text.rfind(b"\n"), text.rfind(b"\r")) + 1
    line = text[line_start:].decode("utf8", "ignore").strip()
    if not line:
        return None
    match = compile_page_marker_regex(page_marker_regex).fullmatch(line, partial=True)
    if match is None or not match.partial:
        return None
    return line_start
