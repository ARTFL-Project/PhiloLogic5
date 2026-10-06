import os
import re as re_stdlib

import regex as re
from philologic.runtime.DB import DB
from philologic.runtime.MetadataQuery import metadata_pattern_search
from philologic.runtime.QuerySyntax import parse_metadata_query, quote_metadata_value, quoted_text
from unidecode import unidecode

from wsgi_helpers import BadRequest

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
    """Suggestions for the term being typed at the end of q, a metadata value (QuerySyntax.parse_metadata_query): the
    values of field it may begin, each after the rest of q and CUTHERE, where the client puts the one chosen back,
    quoted. The term is a quoted value, or the words since the last operator ("victor hu", "Rousseau, Jean-J")."""
    tokens = parse_metadata_query(q, db.locals.metadata_sql_types.get(field, "text"))
    start = len(tokens)
    if tokens and tokens[-1][0] == "QUOTE":
        start -= 1
        token = quoted_text(tokens[-1][1])
    else:
        while start > 0 and tokens[start - 1][0] in ("TERM", "RANGE"):
            start -= 1
        token = " ".join(word for _, word in tokens[start:])
    if not token.strip():  # an empty or blank term, or one ending with an operator: nothing to complete
        return []
    prefix = " ".join(quote_metadata_value(quoted_text(t)) if kind == "QUOTE" else t for kind, t in tokens[:start])
    if prefix:
        prefix = prefix + " CUTHERE "

    norm_tok = token.lower()
    if db.locals.ascii_conversion is True:
        norm_tok = unidecode(norm_tok)
    words = re_stdlib.findall(r"\w+", norm_tok)
    if len(words) > 1:
        matches = values_with_words(words, field, db)
    else:
        matches = metadata_pattern_search(
            re_stdlib.escape(norm_tok), db.locals.db_path + "/data", field, db.locals.ascii_conversion
        )
        # by the normalized token, as the index's words are: "émi" finds "Émile"
        exact_matches = exact_word_pattern_search(
            re_stdlib.escape(norm_tok) + ".*", db.locals.db_path + "/data/frequencies/", field, "TERM",
            db.locals.ascii_conversion,
        )
        for m in exact_matches:
            if m not in matches:
                matches.append(m)
    return [prefix + m for m in highlighter(matches, token, db.locals.ascii_conversion)]


def values_with_words(words, field, db, max_results=100):
    """The values of field with all of words (normalized), the last one as a beginning: "victor hu" suggests
    "Hugo, Victor, 1802-1885.", where only the last word counted."""
    from philologic.runtime.term_expansion import metadata_word_lookup

    *complete, last = words
    values = set(metadata_word_lookup(db.path, field, complete[0]))
    for word in complete[1:]:
        values &= set(metadata_word_lookup(db.path, field, word))
    normalize = (lambda v: unidecode(v.lower())) if db.locals.ascii_conversion is True else str.lower
    return sorted(v for v in values if any(w.startswith(last) for w in re_stdlib.findall(r"\w+", normalize(v))))[
        :max_results
    ]


def exact_word_pattern_search(term, path, field, label, ascii_conversion):
    """Exact word prefix search using LMDB metadata word index."""
    from philologic.runtime.term_expansion import metadata_word_prefix_scan

    # Strip trailing .* regex suffix and unescape regex escapes
    prefix = re_stdlib.sub(r"\.\*$", "", term).lower()
    prefix = re_stdlib.sub(r"\\(.)", r"\1", prefix)
    # Extract word characters for LMDB prefix scan
    words = re_stdlib.findall(r"\w+", prefix)
    if not words:
        return []

    # path is the frequencies directory; db_path is its parent
    db_path = os.path.dirname(path.rstrip("/"))

    return metadata_word_prefix_scan(db_path, field, words[-1], max_results=100)


def highlighter(words, token, ascii_conversion):
    """Highlight autocomplete"""
    token = token.strip()
    new_list = []
    for word in words:
        search_chunk = re.search(re.escape(token), word, re.IGNORECASE)
        if not search_chunk and ascii_conversion is True:
            search_chunk = re.search(re.escape(unidecode(token)), unidecode(word), re.IGNORECASE)
        if not search_chunk:  # matched on the words of token, not on token as it is ("Johann" in "Johann, ...")
            new_list.append(word)
            continue

        word_chunk = word[search_chunk.start() : search_chunk.end()]
        highlighted_chunk = '<span class="highlight">' + word_chunk + "</span>"
        highlighted_word = word.replace(word_chunk, highlighted_chunk)
        new_list.append(highlighted_word)
    return new_list
