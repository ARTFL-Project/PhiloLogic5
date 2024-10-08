#!/var/lib/philologic5/philologic_env/bin/python3

import os
import subprocess
import sys
from wsgiref.handlers import CGIHandler

import orjson
from philologic.runtime.DB import DB
from philologic.runtime.Query import grep_exact, grep_word, grep_word_attributes, split_terms
from philologic.runtime.QuerySyntax import group_terms, parse_query

sys.path.append("..")
import custom_functions

try:
    from custom_functions import WebConfig
except ImportError:
    from philologic.runtime import WebConfig
try:
    from custom_functions import WSGIHandler
except ImportError:
    from philologic.runtime import WSGIHandler


def term_list(environ, start_response):
    """Get term list"""
    status = "200 OK"
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response(status, headers)
    config = WebConfig(os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    db = DB(config.db_path + "/data/")
    request = WSGIHandler(environ, config)
    term = request.term
    if isinstance(term, list):
        term = term[-1]
    all_words = format_query(term, db, config)[:100]
    yield orjson.dumps(all_words)


def format_query(q, db, config):
    """Format query"""
    parsed = parse_query(q)
    group = group_terms(parsed)
    all_groups = split_terms(group)

    # We extract every word tuple
    word_groups = []
    for g in all_groups:
        for inner_g in g:
            word_groups.append(inner_g)
    last_group = word_groups.pop()  # we take the last tuple for autocomplete
    token = last_group[1]
    kind = last_group[0]
    if word_groups:
        prefix = " ".join([i[1] for i in word_groups]) + " "
    else:
        prefix = ""

    frequency_file = config.db_path + "/data/frequencies/normalized_word_frequencies"

    if kind == "TERM" and db.locals.ascii_conversion is True:
        expanded_token = token + ".*"
        grep_proc = grep_word(expanded_token, frequency_file, subprocess.PIPE, db.locals["lowercase_index"])
    elif kind == "TERM" and db.locals.ascii_conversion is False:
        expanded_token = token + ".*"
        grep_proc = grep_word(expanded_token, frequency_file, subprocess.PIPE, db.locals["lowercase_index"])
    elif kind == "QUOTE":
        expanded_token = token[:-1] + ".*" + token[-1]
        grep_proc = grep_exact(expanded_token, frequency_file, subprocess.PIPE)
    elif kind == "LEMMA":
        grep_proc = grep_word_attributes(f"{token}.*", frequency_file, subprocess.PIPE, "lemmas")
        grep_proc.wait()
    elif kind == "LEMMA_ATTR":
        grep_proc = grep_word_attributes(f"{token}.*", frequency_file, subprocess.PIPE, "lemma_word_attributes")
        grep_proc.wait()
    elif kind == "ATTR":
        grep_proc = grep_word_attributes(f"{token}.*", frequency_file, subprocess.PIPE, "word_attributes")
        grep_proc.wait()
    elif kind == "NOT" or kind == "OR":
        return []

    matches = []
    len_token = len(token)
    for line in grep_proc.stdout:
        if kind in ("TERM", "QUOTE"):
            word = line.split(b"\t")[1].strip().decode("utf8")
        else:
            word = line.strip().decode("utf8")
        highlighted_word = f'<span class="highlight">{word[:len_token]}</span>{word[len_token:]}'
        matches.append(highlighted_word)

    output_string = []
    for m in matches:
        if kind == "QUOTE":
            output_string.append(prefix + '"%s"' % m)
        else:
            output_string.append(prefix + m)

    return output_string


if __name__ == "__main__":
    CGIHandler().run(term_list)
