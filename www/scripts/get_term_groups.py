from philologic.runtime.Query import split_terms
from philologic.runtime.QuerySyntax import group_terms, parse_query
from philologic.runtime.term_expansion import REGEX_EXPANSION_CAP, cut_terms


def group_text(group):
    """A query group as the results summary shows it: its terms, joined by |, and its NOT filter."""
    term_group = ""
    not_started = False
    for kind, term in group:
        if kind == "NOT":
            if not_started is False:
                not_started = True
                term_group += " NOT "
        elif kind == "OR":
            term_group += "|"
        elif kind in ("TERM", "QUOTE", "LEMMA", "ATTR", "LEMMA_ATTR"):
            term_group += f" {term} "
    return term_group.strip()


def approximate_groups(request, config, all_groups):
    """For an approximate search, the term each group of the query is the similar words of, and how many they are:
    find_similar_words makes one group of each group of the query as it was typed (quotes aside)."""
    if request.approximate != "yes" or not request.original_q:
        return []
    typed = split_terms(
        group_terms(parse_query(request.original_q.replace('"', ""), query_patterns=config.db_locals.query_patterns))
    )
    if len(typed) != len(all_groups):
        return []
    groups = []
    for typed_group, group in zip(typed, all_groups):
        kinds = [kind for kind, _ in group]
        words = kinds[: kinds.index("NOT")] if "NOT" in kinds else kinds
        groups.append({"term": group_text(typed_group), "variants": sum(1 for kind in words if kind != "OR")})
    return groups


def get_term_groups(request, config):
    if not request["q"]:
        return {"original_query": "", "term_groups": []}
    parsed = parse_query(request.q, query_patterns=config.db_locals.query_patterns)
    group = group_terms(parsed)
    all_groups = split_terms(group)
    term_groups = [group_text(g) for g in all_groups]
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
        # folded in the results summary: liberté (27 similar terms), not the 27 words
        "approximate_groups": approximate_groups(request, config, all_groups),
    }
