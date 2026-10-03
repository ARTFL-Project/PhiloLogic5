from philologic.runtime.Query import split_terms
from philologic.runtime.QuerySyntax import group_terms, parse_query
from philologic.runtime.term_expansion import REGEX_EXPANSION_CAP, cut_terms

def get_term_groups(request, config):
    if not request["q"]:
        return {"original_query": "", "term_groups": []}
    parsed = parse_query(request.q, query_patterns=config.db_locals.query_patterns)
    group = group_terms(parsed)
    all_groups = split_terms(group)
    term_groups = []
    for g in all_groups:
        term_group = ""
        not_started = False
        for kind, term in g:
            if kind == "NOT":
                if not_started is False:
                    not_started = True
                    term_group += " NOT "
            elif kind == "OR":
                term_group += "|"
            elif kind in ("TERM", "QUOTE", "LEMMA", "ATTR", "LEMMA_ATTR"):
                term_group += f" {term} "
        term_group = term_group.strip()
        term_groups.append(term_group)
    # The terms the search expanded to only REGEX_EXPANSION_CAP word forms, which the results summary says
    cut = cut_terms(
        all_groups,
        config.db_path + "/data/frequencies/normalized_word_frequencies",
        config.db_locals["ascii_conversion"],
        config.db_locals["lowercase_index"],
    )
    return {
        "term_groups": term_groups,
        "original_query": request.original_q,
        "cut_terms": [{"term": term, "not": negated} for term, negated in cut],
        "expansion_cap": REGEX_EXPANSION_CAP,
    }
