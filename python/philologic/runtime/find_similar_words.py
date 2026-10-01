#!/var/lib/philologic5/philologic_env/bin/python3
"""Find similar words to query term."""


import hashlib
import os

from Levenshtein import ratio
from philologic.runtime.Query import get_word_groups, rewrite_terms_file, split_terms
from philologic.runtime.QuerySyntax import group_terms, parse_query
from unidecode import unidecode


def get_query_groups(db, request):
    """The groups of the query, as the search sees them (see split_terms)."""
    words = request["q"].replace('"', "")
    return split_terms(group_terms(parse_query(words, query_patterns=db.locals.query_patterns)))


def group_tokens(group):
    """The tokens of a query group that words must match (not its OR, nor the NOT filter and what follows)."""
    tokens = []
    for kind, token in group:
        if kind == "NOT":
            break
        if kind != "OR":
            tokens.append(token.replace('"', ""))
    return tokens


def get_all_words(db, request, query_groups):
    """The words of each query group, normalized as in the word frequencies: those it expands to, or for a group
    that expands to none, as a misspelled word does, its own tokens."""
    words = request["q"].replace('"', "")
    hits = db.query(words)
    hits.finish()
    terms_file = f"{hits.filename}.terms"
    if not os.path.exists(terms_file):  # removed by the hitlist cleanup, from a cached search
        rewrite_terms_file(db, words, hits.filename)
    word_groups = []
    for group, expanded in zip(query_groups, get_word_groups(terms_file)):
        normalized_group = []
        for word in expanded or group_tokens(group):
            word = word.lower()
            if db.locals.ascii_conversion is True:
                word = unidecode(word)
            normalized_group.append(word)
        word_groups.append(normalized_group)
    return word_groups


def find_similar_words(db, config, request):
    """Edit distance function."""
    # Check if lookup is cached
    hashed_query = hashlib.sha256()
    hashed_query.update(request["q"].encode("utf8"))
    hashed_query.update(str(request.approximate_ratio).encode("utf8"))
    approximate_filename = os.path.join(db.hitlist_dir, f"{hashed_query.hexdigest()}.approximate_terms")
    if os.path.isfile(approximate_filename):
        with open(approximate_filename, encoding="utf8") as fh:
            approximate_terms = fh.read().strip()
        if approximate_terms:  # an empty one is from an earlier version, which could lose every query group
            return approximate_terms
    query_groups = get_query_groups(db, request)
    word_groups = get_all_words(db, request, query_groups)
    file_path = os.path.join(config.db_path, "data/frequencies/normalized_word_frequencies")
    new_query_groups = [set([]) for i in word_groups]
    with open(file_path, encoding="utf8") as fh:
        for line in fh:
            line = line.strip()
            try:
                normalized_word, regular_word = line.split("\t")
                for pos, word_group in enumerate(word_groups):
                    for query_word in word_group:
                        if ratio(query_word, normalized_word) >= float(request.approximate_ratio):
                            new_query_groups[pos].add(f'"{regular_word}"')
            except ValueError:
                pass
    # A group with no similar word keeps its own tokens, which find nothing: dropping it would change the query, or
    # leave none, which would be a search for everything.
    new_query = " ".join(
        " | ".join(sorted(similar)) if similar else " | ".join(group_tokens(group))
        for group, similar in zip(query_groups, new_query_groups)
    ) or request["q"]
    with open(approximate_filename, "w", encoding="utf8") as cached_file:
        cached_file.write(new_query)
    return new_query
