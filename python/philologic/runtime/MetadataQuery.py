#!/var/lib/philologic5/philologic_env/bin/python3

import re
import struct
import sys

import regex
from unidecode import unidecode

from . import HitList
from .exceptions import BadRequest
from .QuerySyntax import group_terms, parse_date_query, parse_metadata_query, quoted_text
from .sql_validation import validate_column

_OBJ_PREFIX_LEN = {"doc": 1, "div1": 2, "div2": 3, "div3": 4, "para": 5, "sent": 6}


def bulk_load_metadata(db, fields, extra_columns=None, inherit=False):
    """Bulk-load metadata fields from toms into dicts keyed by philo_id prefix.

    Groups fields by object level and runs one SQL query per level.
    Fields must be pre-validated via validate_column() if they come from user input.

    Returns {field_name: (prefix_len, {philo_id_prefix_tuple: value})}
    When extra_columns is provided, values become tuples: (value, extra1, extra2, ...)

    A div field is keyed by the 4 numbers of each div, so a hit finds the one of its innermost div. With inherit
    (and no extra_columns), a div with no value has that of the div holding it, as HitWrapper reads a div field: the
    words of an article's implicit div2 and div3 have its head and author.
    """
    metadata_types = db.locals.metadata_types
    caches = {}
    by_level = {}
    for field in fields:
        obj_type = metadata_types.get(field, "doc")
        by_level.setdefault(obj_type, []).append(field)

    cursor = db.dbh.cursor()
    for obj_type, obj_fields in by_level.items():
        if obj_type == "div":
            philo_types = ("div1", "div2", "div3")
            prefix_len = 4
        else:
            philo_types = (obj_type,)
            prefix_len = _OBJ_PREFIX_LEN.get(obj_type, 1)

        cols = ", ".join(obj_fields)
        if extra_columns:
            cols += ", " + ", ".join(extra_columns)
        placeholders = ", ".join("?" for _ in philo_types)
        cursor.execute(
            f"SELECT philo_id, {cols} FROM toms WHERE philo_type IN ({placeholders})",
            philo_types,
        )
        n_fields = len(obj_fields)
        for row in cursor:
            philo_id_str = row[0]
            parts = philo_id_str.split()
            prefix = tuple(int(x) for x in parts[:prefix_len])
            for i, field in enumerate(obj_fields):
                val = row[i + 1] or ""
                if extra_columns:
                    extras = tuple(row[n_fields + 1 + j] for j in range(len(extra_columns)))
                    val = (val,) + extras
                if field not in caches:
                    caches[field] = (prefix_len, {})
                caches[field][1][prefix] = val
        if inherit and obj_type == "div":
            for field in obj_fields:
                if field in caches:
                    caches[field] = (prefix_len, _inherited_div_values(caches[field][1]))

    return caches


def _inherited_div_values(values):
    """values, keyed by div (doc, div1, div2, div3), with a div's missing value taken from its div2, else its div1."""
    inherited = {}
    for (doc, div1, div2, div3), value in values.items():
        if not value and div3:
            value = values.get((doc, div1, div2, 0))
        if not value and div2:
            value = values.get((doc, div1, 0, 0))
        inherited[doc, div1, div2, div3] = value or ""
    return inherited


def query_levels(db, metadata):
    """The levels of a metadata query: the fields of metadata with values (lists of them), by level of
    metadata_hierarchy, doc first, each level with the philo_types of the objects it selects, those of its last field
    in metadata with a type (None if none has one). philo_id goes on the last level."""
    levels = []
    for level_fields in db.locals["metadata_hierarchy"]:
        philo_types, fields = None, {}
        for field, values in metadata.items():
            if values and field in level_fields:
                fields[field] = values
                if field in db.locals["metadata_types"]:
                    philo_types = _philo_types(db.locals["metadata_types"][field])
        if fields:
            levels.append((philo_types, fields))
    if "philo_id" in metadata:
        if levels:
            levels[-1][1]["philo_id"] = metadata["philo_id"]
        else:
            levels.append((None, {"philo_id": metadata["philo_id"]}))
    return levels


def _philo_types(metadata_type):
    """The philo_types of the objects of a metadata type: those of div1, div2 and div3 for div fields."""
    return ("div", "div1", "div2", "div3") if metadata_type == "div" else (metadata_type,)


def metadata_query(db, filename, levels, sort_order, raw_results=False, ascii_conversion=True, lock=None):
    """Write to filename the object ids the levels of a metadata query (query_levels) select, and return their
    HitList. Releases lock (see HitList.claim_hitlist) once filename is complete."""
    if db.locals["debug"]:
        print("METADATA_QUERY:", levels, "\nASCII CONVERSION", ascii_conversion, file=sys.stderr)
    # The file is always written in load order, which filtering word hits against it relies on, and it is cached
    # whatever sort was asked for: sort_order only applies to the HitList returned below.
    pack = struct.Struct("7I").pack
    try:
        with open(filename, "wb") as corpus_fh:
            for philo_id in object_ids(db, levels, ascii_conversion):
                corpus_fh.write(pack(*philo_id))
    except Exception:
        # Not an empty corpus: have the next request query it again, and this one fail rather than show no results
        HitList.fail_hitlist(filename, lock)
        raise
    HitList.finish_hitlist(filename, lock)
    return HitList.HitList(filename, 0, db, raw=raw_results, sort_order=sort_order, ascii_conversion=ascii_conversion)


def object_ids(db, levels, ascii_conversion=True):
    """The philo_ids, as tuples in load order, of the objects the last of levels selects within those the levels
    before it select: author and head, the divs with that head in that author's documents."""
    if not levels:  # metadata with no field to query, as philo_type alone: it fails, as it always did
        raise ValueError("No metadata field to select objects by")
    outer = None
    for n, (philo_types, fields) in enumerate(levels):
        rows = level_query(db, philo_types, fields, ascii_conversion)
        ids = (tuple(map(int, row[0].split(" "))) for row in rows)
        if outer is not None:
            ids = _within(ids, outer)
        if n == len(levels) - 1:
            yield from ids
        else:
            outer = list(ids)
            if not outer:
                return


def _within(ids, outer):
    """Those of ids (in load order) within one of the outer objects (in load order), whose philo_id they start with,
    up to its first 0. They end after the last outer object, or the outermost one containing it."""
    prefixes = {philo_id[: _depth(philo_id)] for philo_id in outer}
    lengths = sorted({len(prefix) for prefix in prefixes})
    last = outer[-1][: _depth(outer[-1])]
    last = next(last[:n] for n in lengths if last[:n] in prefixes)
    for philo_id in ids:
        if philo_id[: len(last)] > last:
            return
        if any(philo_id[:n] in prefixes for n in lengths):
            yield philo_id


def _depth(philo_id):
    """The length of a philo_id up to its first 0."""
    return philo_id.index(0) if 0 in philo_id else len(philo_id)


def level_query(db, philo_types, fields, ascii_conversion):
    """The rows (philo_id) of the objects of philo_types (any if None) with the values of fields, in load order."""
    clauses, params = [], []
    for column, values in fields.items():
        column = validate_column(column, db)
        for v in values:
            field_type = db.locals.metadata_sql_types.get(column, "text")
            if field_type == "date":
                v = v.replace('"', "")  # remove quotes
                parsed = parse_date_query(v)
            else:
                parsed = parse_metadata_query(v, field_type)
            grouped = group_terms(parsed)
            expanded = expand_grouped_query(grouped, db.path, column, ascii_conversion)
            sql_clause = make_grouped_sql_clause(expanded, column, db)
            if db.locals["debug"]:
                print("METADATA_TOKENS:", parsed, file=sys.stderr)
                print("METADATA_SYNTAX GROUPED:", grouped, file=sys.stderr)
                print("METADATA_SYNTAX EXPANDED:", expanded, file=sys.stderr)
                print("SQL_SYNTAX:", sql_clause, file=sys.stderr)
            clauses.append(sql_clause)
    if philo_types:
        clauses.append(f"philo_type IN ({', '.join('?' for _ in philo_types)})")
        params = list(philo_types)
    if clauses:
        query = "SELECT philo_id FROM toms WHERE " + " AND ".join("(%s)" % c for c in clauses)
    else:
        query = "SELECT philo_id FROM toms"
    query += " ORDER BY rowid"
    if db.locals["debug"]:
        print("INNER QUERY: ", query, params, file=sys.stderr, flush=True)
    return db.dbh.execute(query, params)


def expand_grouped_query(grouped, db_path, field, ascii_conversion):
    """Expand grouped SQL query"""
    expanded = []
    for group in grouped:
        expanded_group = []
        for kind, token in group:
            if kind == "TERM":
                norm_term = token.lower()
                if ascii_conversion is True:
                    norm_term = unidecode(norm_term)
                expanded_terms = metadata_pattern_search(norm_term, db_path, field, ascii_conversion)
                if expanded_terms:
                    expanded_tokens = [("QUOTE", '"' + e + '"') for e in expanded_terms]
                    fully_expanded_tokens = []
                    first = True
                    for e in expanded_tokens:
                        if first:
                            first = False
                        else:
                            fully_expanded_tokens.append(("OR", "|"))
                        fully_expanded_tokens.append(e)
                else:  # if we have no matches, just put an inexact match in as placeholder.  Will fail later.
                    fully_expanded_tokens = [("QUOTE", '"' + norm_term + '"')]
                expanded_group.extend(fully_expanded_tokens)
            else:
                if kind == "NOT":
                    if expanded_group:
                        expanded.append(expanded_group)
                    expanded_group = [(kind, token)]
                elif kind != "OR":
                    expanded_group.append((kind, token))
        if expanded_group:
            expanded.append(expanded_group)
    return expanded


def _range_bounds(kind, value, column, db):
    """The bounds of a range: "1700-1750", "-1750" or "1700-" (filled with the column's lowest or highest value),
    negative years too ("-500--400"), or a date range "x<=>y". BadRequest for anything else ("1789-07-14")."""
    if kind == "DATE_RANGE":
        bounds = value.split("<=>")
    else:
        match = re.fullmatch(r"(-?\d*)-(-?\d*)", value) or re.fullmatch(r"([^-]*)-([^-]*)", value)
        bounds = list(match.groups()) if match else []
    if len(bounds) != 2:
        raise BadRequest(f"{value}: a range is two values, from-to, as 1700-1750")
    for i, function in ((0, "min"), (1, "max")):
        if not bounds[i]:
            cursor = db.dbh.cursor()
            cursor.execute(f"select {function}({column}) from toms")
            bounds[i] = str(cursor.fetchone()[0])
    return bounds


def make_grouped_sql_clause(expanded, column, db):
    """The SQL clause for the groups of a metadata value, which must all match: each group the OR of its ranges, its
    values and NULL, or with NOT none of them. NOT x is everything x doesn't select, objects with no value included
    (SQL's NOT left them out: NOT hugo missed the 248 documents with no author), unless x is NULL or has it.

    Note: column is expected to be pre-validated by validate_column() in level_query()
    before being passed to this function, ensuring SQL injection protection.
    """
    esc = escape_sql_string
    clauses = []
    for group in expanded:
        negated = group[0][0] == "NOT"
        alternatives, values = [], []
        for kind, token in group[1:] if negated else group:
            if kind in ("RANGE", "DATE_RANGE"):
                lower, upper = _range_bounds(kind, token, column, db)
                alternatives.append(f"({column} >= {esc(lower)} AND {column} <= {esc(upper)})")
            elif kind == "QUOTE":
                values.append(esc(quoted_text(token)))
            elif kind == "DATE":
                values.append(esc(token))
            elif kind == "NULL":
                alternatives.append(f"{column} IS NULL")
        if values or not alternatives:
            alternatives.insert(0, f"{column} IN ({', '.join(values)})")
        clause = " OR ".join(alternatives)
        if not negated:
            clauses.append(f"({clause})")
        elif any(kind == "NULL" for kind, _ in group[1:]):
            clauses.append(f"NOT ({clause})")
        else:
            clauses.append(f"(NOT ({clause}) OR {column} IS NULL)")
    return "(%s)" % " AND ".join(clauses)


# The parts of a word, as jean-jacques or d'autriche has: runs of anything but hyphens, apostrophes and the punctuation
# the index splits values at, regex bracket expressions whole ("[a-z]")
_WORD_PARTS = regex.compile(r"(?:\[[^\]]*\]|[^\-'\u2019\u02bc,;:!/])+")


def metadata_pattern_search(term, db_path, field, ascii_conversion=True):
    """The metadata values that have term as a word, using the LMDB index. term is normalized (lowercase, and with
    ascii_conversion unidecoded), as the index's words are. A regex matches whole words. A word of several parts
    (jean-jacques, d'autriche) needs its parts side by side and in that order, whatever separates them."""
    if isinstance(term, bytes):
        term = term.decode("utf-8", errors="replace")

    from .term_expansion import metadata_word_lookup, metadata_word_regex_scan, _is_regex_pattern

    parts = []  # (regex, pattern) or (word, word): those of the index, which are \w+ runs
    for part in _WORD_PARTS.findall(term):
        if _is_regex_pattern(part):
            parts.append(("regex", part))
        else:
            parts.extend(("word", word) for word in re.findall(r"\w+", part))
    if not parts:
        return []
    if len(parts) == 1:
        kind, part = parts[0]
        if kind == "regex":  # a cursor scan of the index's words
            return metadata_word_regex_scan(db_path, field, part)
        return metadata_word_lookup(db_path, field, part)
    common = None
    for kind, part in parts:
        lookup = metadata_word_regex_scan if kind == "regex" else metadata_word_lookup
        found = set(lookup(db_path, field, part))
        common = found if common is None else common & found
    matchers = [regex.compile(part).fullmatch if kind == "regex" else part.__eq__ for kind, part in parts]

    def side_by_side(value):
        # The words of the value, normalized as term is: "qu’en" (unidecoded to "qu'en") is in "Qu’en dira-t-on"
        words = re.findall(r"\w+", unidecode(value.lower()) if ascii_conversion else value.lower())
        return any(
            all(match(words[i + k]) for k, match in enumerate(matchers)) for i in range(len(words) - len(matchers) + 1)
        )

    return [v for v in common if side_by_side(v)]


def escape_sql_string(s):
    """Escape SQL string"""
    s = s.replace("'", "''")
    return "'%s'" % s
