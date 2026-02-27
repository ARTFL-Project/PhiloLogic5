from philologic.runtime.DB import DB
from philologic.runtime.Query import split_terms
from philologic.runtime.QuerySyntax import group_terms, parse_query
from philologic.runtime.term_expansion import expand_autocomplete


def autocomplete_term(request, config):
    """Get term list"""
    db = DB(config.db_path + "/data/")
    term = request.term
    if isinstance(term, list):
        term = term[-1]
    all_words = format_query(term, db, config)[:100]
    return all_words


def format_query(q, db, config):
    """Format query using LMDB cursor scans (no subprocess)."""
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
    db_path = config.db_path + "/data"

    matches = expand_autocomplete(
        kind,
        token,
        frequency_file=frequency_file,
        db_path=db_path,
        ascii_conversion=db.locals.ascii_conversion,
        lowercase=db.locals["lowercase_index"],
        max_results=100,
    )

    if not matches:
        return []

    # len of the typed portion (without surrounding quotes for QUOTE)
    raw_token = token[1:-1] if kind == "QUOTE" else token
    len_token = len(raw_token)

    output_string = []
    for word in matches:
        highlighted = f'<span class="highlight">{word[:len_token]}</span>{word[len_token:]}'
        if kind == "QUOTE":
            output_string.append(prefix + '"%s"' % highlighted)
        else:
            output_string.append(prefix + highlighted)

    return output_string
