import os
import subprocess

import regex as re
import re as re_stdlib
from philologic.runtime.DB import DB
from philologic.runtime.MetadataQuery import metadata_pattern_search
from unidecode import unidecode

from wsgi_helpers import BadRequest

_environ = os.environ
_environ["PATH"] += ":/usr/local/bin/"
_environ["LANG"] = "C"

patterns = [
    ("QUOTE", r'".+?"'),
    ("QUOTE", r'".+'),
    ("NOT", "NOT"),
    ("OR", r"\|"),
    ("RANGE", r"[^|\s]+?\-[^|\s]+"),
    ("RANGE", r"\d+\-\Z"),
    ("RANGE", r"\-\d+\Z"),
    ("NULL", r"NULL"),
    ("TERM", r'[^\-|"]+'),
]

accented_roman_chars = re.compile(r"[\u00c0-\u0174]")


def autocomplete_metadata(request, config):
    """Retrieve metadata list"""
    db = DB(config.db_path + "/data/")
    metadata = request.term
    field = request.field

    # Handle list case early (workaround for when jquery sends a list of words via back button)
    if isinstance(field, list):
        field = field[-1]
    if isinstance(metadata, list):
        metadata = metadata[-1]

    if field not in db.locals.metadata_fields:
        raise BadRequest("Invalid metadata field provided.")

    words = format_query(metadata, field, db)[:100]
    return words


def format_query(q, field, db):
    """Format query"""
    parsed = parse_query(q)
    parsed_split = []
    for label, token in parsed:
        l, t = label, token
        if l == "QUOTE":
            if t[-1] != '"':
                t += '"'
            subtokens = t[1:-1].split("|")
            parsed_split += [("QUOTE_S", sub_t) for sub_t in subtokens if sub_t]
        elif l == "RANGE":
            parsed_split += [("TERM", t)]
        else:
            parsed_split += [(l, t)]
    output_string = []
    label, token = parsed_split[-1]
    prefix = " ".join('"' + t[1] + '"' if t[0] == "QUOTE_S" else t[1] for t in parsed_split[:-1])
    if prefix:
        prefix = prefix + " CUTHERE "
    if label == "QUOTE_S" or label == "TERM":
        norm_tok = token.lower()
        if db.locals.ascii_conversion is True:
            norm_tok = unidecode(norm_tok)

        safe_token = re_stdlib.escape(token.lower())
        safe_norm_tok = re_stdlib.escape(norm_tok).encode("utf-8")

        matches = metadata_pattern_search(
            safe_norm_tok, db.locals.db_path + "/data/frequencies/normalized_%s_frequencies" % field
        )

        substr_token = safe_token.lower()
        exact_matches = exact_word_pattern_search(
            substr_token + ".*", db.locals.db_path + "/data/frequencies/", field, label, db.locals.ascii_conversion
        )
        for m in exact_matches:
            if m not in matches:
                matches.append(m)
        matches = highlighter(matches, token, db.locals.ascii_conversion)
        for m in matches:
            if label == "QUOTE_S":
                output_string.append(prefix + '"%s"' % m)
            else:
                if re.search(r"\|", m):
                    m = '"' + m + '"'
                output_string.append(prefix + m)

    return output_string


def parse_query(qstring):
    """Parse query"""
    buf = qstring[:]
    parsed = []
    while len(buf) > 0:
        for label, pattern in patterns:
            m = re.match(pattern, buf)
            if m:
                parsed.append((label, m.group()))
                buf = buf[m.end() :]
                break
        else:
            buf = buf[1:]
    return parsed


def exact_word_pattern_search(term, path, field, label, ascii_conversion):
    """Exact word pattern search"""
    if label == "TERM":
        norm_term = term.lower()
        path = path + "normalized_%s_frequencies" % field
        command = ["rg", "-awie", "[[:blank:]]?" + norm_term, path]
        grep = subprocess.Popen(command, stdout=subprocess.PIPE, env=_environ)
        cut = subprocess.Popen(["cut", "-f", "2"], stdin=grep.stdout, stdout=subprocess.PIPE)
        match, _ = cut.communicate()
        matches = [i.decode("utf8") for i in match.split(b"\n") if i]

    elif label == "QUOTE_S":
        path = path + "%s_frequencies" % field
        command = ["rg", "-awie", "^" + term, path]
        grep = subprocess.Popen(command, stdout=subprocess.PIPE, env=_environ)
        cut = subprocess.Popen(["cut", "-f", "1"], stdin=grep.stdout, stdout=subprocess.PIPE)
        match, _ = cut.communicate()
        matches = [i.decode("utf8") for i in match.split(b"\n") if i]

    return matches


def highlighter(words, token, ascii_conversion):
    """Highlight autocomplete"""
    new_list = []
    for word in words:
        if ascii_conversion is True:
            flattened_token = unidecode(token)
            flattened_suggestion = unidecode(word)

        search_chunk = re.search(token, word, re.IGNORECASE)
        if not search_chunk:
            search_chunk = re.search(flattened_token, flattened_suggestion, re.IGNORECASE)

        word_chunk = word[search_chunk.start() : search_chunk.end()]
        highlighted_chunk = '<span class="highlight">' + word_chunk + "</span>"
        highlighted_word = word.replace(word_chunk, highlighted_chunk)
        new_list.append(highlighted_word)
    return new_list
