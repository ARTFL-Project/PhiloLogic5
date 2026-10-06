#!/var/lib/philologic5/philologic_env/bin/python3

import unicodedata

import regex as re

from philologic.runtime.exceptions import BadRequest

patterns = [
    ("LEMMA_ATTR", r'lemma:[^\-|\s"]+:[^\-|\s"]+'),
    ("LEMMA", r'lemma:[^\-|\s"]+'),
    ("ATTR", r'[^\-|\s"]+:[^\-|\s"]+'),
    ("QUOTE", r'".+?"'),
    ("QUOTE", r'".+'),
    ("NOT", "NOT"),
    ("OR", r"\|"),
    ("RANGE", r"[^|\s\[]+?\-[^|\s]+"),  # no "[" before the dash: "17[0-4]." is a regex
    ("RANGE", r"\d+\-\Z"),
    ("RANGE", r"\-\d+\Z"),
    ("NULL", r"NULL"),
    ("TERM", r'(?:\[[^\]]*\]|[^\-|\s"])+'),  # with the hyphens of its regex bracket expressions ("[a-z]")
]


def quoted_text(token):
    """The text of a QUOTE token, without its quotes: the closing one may be missing (patterns take '".+' too)."""
    return token[1:-1] if len(token) > 1 and token.endswith('"') else token[1:]


def parse_query(qstring, query_patterns=None):
    """Parse query"""
    if query_patterns is None:
        query_patterns = patterns
    buf = qstring[:]
    parsed = []
    while len(buf) > 0:
        for label, pattern in query_patterns:
            m = re.match(pattern, buf)
            if m:
                parsed.append((label, m.group()))
                buf = buf[m.end() :]
                break
        else:
            buf = buf[1:]
    return parsed


# Metadata values have a grammar of their own (docs/query_syntax.md): no word-search rules, and ranges only in numeric
# fields, so that "-" and "'" are parts of words in text fields ("jean-jacques", "d'autriche")
_METADATA_QUOTES = {'"': '"', "\u201c": "\u201d", "\u201d": "\u201d"}  # "…", “…”
_METADATA_STOPS = set('|\uff5c"\u201c\u201d')  # | ｜ and the quotes end a word


def quote_metadata_value(value):
    """value as a quoted metadata value, which matches it exactly: its quotes doubled."""
    return '"' + str(value).replace('"', '""') + '"'


def parse_metadata_query(value, field_type="text"):
    """The tokens of a metadata value for a field of field_type (its metadata_sql_types entry):
    * QUOTE: a quoted value, "…" or “…”, matched whole and exactly, its doubled quotes ("") quotes of the value;
      the token is the value between two quotes, as quoted_text reads it;
    * OR (| or OR), NOT and NULL;
    * RANGE, in int fields only: a word with a hyphen (1700-1750, -1750, 1750-);
    * TERM: any other word (up to a space, |, or a quote), full-width forms made ASCII."""
    tokens = []
    i, n = 0, len(value)
    while i < n:
        char = value[i]
        if char.isspace():
            i += 1
        elif char in _METADATA_QUOTES:
            closing, j, text = _METADATA_QUOTES[char], i + 1, []
            while j < n:
                if value[j] == closing:
                    if closing == '"' and j + 1 < n and value[j + 1] == '"':  # doubled: a quote of the value
                        text.append('"')
                        j += 2
                        continue
                    break
                text.append(value[j])
                j += 1
            tokens.append(("QUOTE", '"' + "".join(text) + '"'))
            i = j + 1
        elif char in "|\uff5c":
            tokens.append(("OR", "|"))
            i += 1
        else:
            j = i
            while j < n and not value[j].isspace() and value[j] not in _METADATA_STOPS:
                j += 1
            word = unicodedata.normalize("NFKC", value[i:j])
            i = j
            if word == "OR":
                tokens.append(("OR", "|"))
            elif word in ("NOT", "NULL"):
                tokens.append((word, word))
            elif field_type == "int" and "-" in word and "[" not in word:
                tokens.append(("RANGE", word))
            else:
                tokens.append(("TERM", word))
    return tokens


def value_groups(value, field_type="text"):
    """The groups of a metadata value for a field of field_type, all of which must match: (negated, tokens), each
    the OR of its tokens. A value starts a group unless an OR or a NOT is before it, a NOT starts a negated group, and
    in date fields, values join the group before them ("1789 1790" is either year)."""
    if field_type == "date":
        tokens = parse_date_query(value)
    else:
        tokens = parse_metadata_query(value, field_type)
    groups, joined = [], False
    for kind, token in tokens:
        if kind == "NOT":
            groups.append((True, []))
            joined = True
        elif kind == "OR":
            joined = True
        else:
            if not groups or not (joined or field_type == "date"):
                groups.append((False, []))
            groups[-1][1].append((kind, token))
            joined = False
    return groups


def parse_date_query(value):
    """The tokens of a value for a date field, its dates as the database stores them (1789-07-14, 0800-01-01):
    * OR (| or OR), NOT and NULL;
    * DATE: a day (1789-7-14);
    * DATE_RANGE: from<=>to, from the first day of a year, month or day to the last of another, either left out for
      no bound (<=>1790-06 is up to the end of June 1790); and a year or a month alone, as the range of its days.
    Quotes are left out. BadRequest for any other word."""
    value = "".join(char for char in unicodedata.normalize("NFKC", value) if char not in _METADATA_QUOTES)
    tokens = []
    for word in re.findall(r"\||[^\s|]+", re.sub(r"\s*<=>\s*", "<=>", value)):
        if word in ("|", "OR"):
            tokens.append(("OR", "|"))
        elif word in ("NOT", "NULL"):
            tokens.append((word, word))
        elif "<=>" in word:
            start, end = word.split("<=>", 1)
            tokens.append(("DATE_RANGE", f"{_date_bound(start, word, True)}<=>{_date_bound(end, word, False)}"))
        else:  # a day is a DATE: its first day is its last
            first, last = _date_bound(word, word, True), _date_bound(word, word, False)
            tokens.append(("DATE", first) if first == last else ("DATE_RANGE", f"{first}<=>{last}"))
    return tokens


_DATE = re.compile(r"(\d+)(?:-(\d+)(?:-(\d+))?)?")


def _date_bound(date, word, start):
    """A year, month or day as the database stores dates, at its first day if start, else at its last: 1789-7 is
    1789-07-01 or 1789-07-31. "" for "", no bound. BadRequest if date is no date (word is the token it is in)."""
    if not date:
        return ""
    match = _DATE.fullmatch(date)
    if not match:
        raise BadRequest(f"{word!r} is no date: dates are written 1789, 1789-07 or 1789-07-14, ranges 1789<=>1790")
    year, month, day = match.groups()
    month = int(month) if month else (1 if start else 12)
    day = int(day) if day else (1 if start else 31)
    return f"{int(year):04d}-{month:02d}-{day:02d}"


def group_terms(parsed):
    """Group terms for SQL query"""
    grouped = []
    current_clause = []
    last_term = None
    for kind, val in parsed:
        if last_term == "RANGE" and kind != "OR":
            # detach ranges, unless an alternative follows ("1700-1750 | 1800-1850")
            grouped.append(current_clause)
            current_clause = []

        if kind in ("LEMMA", "TERM", "QUOTE", "ATTR", "LEMMA_ATTR", "NULL"):
            if last_term != "OR" and last_term != "NOT":
                grouped.append(current_clause)
                current_clause = []
        elif kind == "OR":
            pass
        elif kind == "RANGE":
            # RANGE should immediately detach a new clause and then close it, unless it is an alternative
            if last_term not in ("NOT", "OR"):
                grouped.append(current_clause)
                current_clause = []
        elif kind == "NOT":
            # NOT should be put in the same clause as it's predecessors
            pass
        current_clause.append((kind, val))
        last_term = kind
    grouped.append(current_clause)
    # filter out possible empty groups
    grouped = [g for g in grouped if g != []]
    return grouped
