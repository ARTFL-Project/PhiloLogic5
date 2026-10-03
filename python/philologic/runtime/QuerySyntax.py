#!/var/lib/philologic5/philologic_env/bin/python3

import unicodedata

import regex as re

YEAR_MONTH = re.compile(r"^(\d+)-(\d+)\Z")
YEAR = re.compile(r"^(\d+)\Z")

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

date_patterns = [
    ("NOT", "NOT"),
    ("OR", r"\|"),
    ("DATE", r"(\d+-\d+-\d+)\Z"),
    ("YEAR", r"(\d+)\Z"),
    ("YEAR_MONTH", r"(\d+-\d+)\Z"),
    ("YEAR_MONTH_DAY", r"(\d+-\d+-\d+)\Z"),
    ("DATE_RANGE", r"([^<]+)<=>(.*)"),
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


def expand_date(date, start=True):
    """Expand incomplete dates"""
    if YEAR.search(date):
        if start is True:
            date = f"{date}-01-01"
        else:
            date = f"{date}-12-31"
    if YEAR_MONTH.search(date):
        if start is True:
            date = f"{date}-01"
        else:
            date = f"{date}-31"
    return date


def parse_date_query(qstring):
    """Parse date query"""
    buf = qstring[:]
    parsed = []
    while len(buf) > 0:
        for label, pattern in date_patterns:
            m = re.match(pattern, buf)
            if m:
                date = m.group().strip()
                if label == "YEAR":
                    label = "DATE_RANGE"
                    query = f"{date}-01-01<=>{date}-12-31"
                elif label == "YEAR_MONTH":
                    label = "DATE_RANGE"
                    query = f"{date}-01<=>{date}-31"
                elif label == "DATE_RANGE":
                    start_date, end_date = date.split("<=>")
                    start_date = expand_date(start_date)
                    end_date = expand_date(end_date, start=False)
                    query = f"{start_date}<=>{end_date}"
                else:
                    query = m.group()
                parsed.append((label, query))
                buf = buf[m.end() :]
                break
        else:
            buf = buf[1:]
    return parsed


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
