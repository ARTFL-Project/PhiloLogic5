#!/var/lib/philologic5/philologic_env/bin/python3
"""Parses queries stored in the environ object."""


import urllib.parse

from philologic.runtime.access_control import is_authenticated
from philologic.runtime.exceptions import BadRequest
from philologic.runtime.Query import query_parse, resolve_method

# Larger integers are no position or count here, and don't fit the 64 bits of the JSON encoder
MAX_INTEGER = 2**63 - 1


def whole_number(name, value, default):
    """The integer a query parameter gives, or its default if it is empty. BadRequest if it gives none: int() raised
    a ValueError while the middleware built the request, so every report gave a 500."""
    if value in (None, ""):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise BadRequest(f"{name} must be a whole number, not {value!r}") from None
    if abs(number) > MAX_INTEGER:
        raise BadRequest(f"{name} is too large: {value}")
    return number


def expand_approximate_query(request, config):
    """Expand query terms for approximate/fuzzy search. Requires DB access."""
    from philologic.runtime.DB import DB
    from philologic.runtime.find_similar_words import find_similar_words

    request.cgi["original_q"] = request.cgi["q"][:]
    db = DB(config.db_path + "/data/")
    request.cgi["q"][0] = find_similar_words(db, config, request)


def parse_metadata(cgi, q, metadata_fields, metadata_sql_types, config):
    """Parse metadata fields from query params. Returns (metadata_dict, no_metadata). Their values are read by the
    metadata grammar (QuerySyntax.parse_metadata_query), not rewritten by the word-search rules."""
    metadata = {}
    num_empty = 0
    for field in metadata_fields:
        if field in cgi and cgi[field]:
            if q != "":
                metadata[field] = cgi[field][0]
            elif cgi[field][0] != "":
                metadata[field] = cgi[field][0]
        if field not in cgi or not cgi[field][0]:
            num_empty += 1
    metadata["philo_type"] = cgi.get("philo_type", [""])[0]
    no_metadata = num_empty == len(metadata_fields)
    return metadata, no_metadata


class WSGIHandler(object):
    """Class which parses the environ object and massages query arguments for PhiloLogic5."""

    def __init__(self, environ, config):
        """Initialize class."""
        self.path_info = environ.get("PATH_INFO", "")
        self.query_string = environ["QUERY_STRING"]
        self.db_path = environ.get("PHILOLOGIC_DBURL", "")

        self.authenticated = is_authenticated(environ, config)
        self.cgi = urllib.parse.parse_qs(self.query_string, keep_blank_values=True)
        self.defaults = {"results_per_page": "25", "start": "0", "end": "0"}

        # Check the header for JSON content_type or look for a format=json
        # keyword
        if "CONTENT_TYPE" in environ:
            self.content_type = environ["CONTENT_TYPE"]
        else:
            self.content_type = "text/HTML"
        # If format is set, it overrides the content_type
        if "format" in self.cgi:
            if self.cgi["format"][0] == "json":
                self.content_type = "application/json"
            else:
                self.content_type = self.cgi["format"][0] or ""

        # Make byte a direct attribute of the class since it is a special case and
        # can contain more than one element
        if "byte" in self.cgi:
            self.byte = self.cgi["byte"]

        if "approximate" in self.cgi:
            ratio = self.cgi.get("approximate_ratio", [""])[0]
            if ratio != "":
                try:
                    self.approximate_ratio = float(ratio) / 100
                except ValueError:
                    raise BadRequest(f"approximate_ratio must be a number, not {ratio!r}") from None
            else:
                self.approximate_ratio = 1

        if "q" in self.cgi:
            self.cgi["q"][0] = query_parse(self.cgi["q"][0], config)
            if self.approximate == "yes":
                expand_approximate_query(self, config)
            if self.cgi["q"][0] != "":
                self.no_q = False
            else:
                self.no_q = True
            # self.cgi['q'][0] = self.cgi['q'][0].encode('utf8')
        else:
            self.no_q = True

        method, self.arg = resolve_method(
            self.q, self["method"], self["method_arg"], self.cooc_order, config.db_locals["query_patterns"]
        )
        self.cgi["arg"] = [self.arg]
        self.cgi["method"] = [method]

        self.metadata_fields = config.db_locals["metadata_fields"]

        self.start = whole_number("start", self["start"], 0)
        self.end = whole_number("end", self["end"], 0)
        self.results_per_page = whole_number("results_per_page", self["results_per_page"], 25)
        for key in ("start", "end", "results_per_page"):  # for reports reading request[key] too: "" was a 500
            if key in self.cgi:
                self.cgi[key][0] = str(getattr(self, key))
        if self.start_date:
            try:
                self.start_date = int(self["start_date"])
            except ValueError:
                self.start_date = "invalid"
        if self.end_date:
            try:
                self.end_date = int(self["end_date"])
            except ValueError:
                self.end_date = "invalid"

        self.metadata, self.no_metadata = parse_metadata(
            self.cgi, self["q"], self.metadata_fields,
            config.db_locals["metadata_sql_types"], config,
        )

        try:
            self.path_components = [c for c in self.path_info.split("/") if c]
        except:
            self.path_components = []

        # Fields to sort results by. The web client sends them as sort_by: sort_by[]=author&sort_by[]=title
        # (axios), sort_by=author&sort_by=title (from its URL) or sort_by=author,title (export links).
        # sort_order is the older name. "rowid" means load order.
        sort_order = []
        for key in ("sort_by[]", "sort_by", "sort_order"):
            if key in self.cgi:
                sort_order = [field for value in self.cgi[key] for field in value.split(",") if field]
                break
        sort_order = [field for field in sort_order if field != "rowid"]
        for field in sort_order:  # sorts look fields up by object type: others gave KeyErrors (500)
            if field not in config.db_locals["metadata_types"]:
                raise BadRequest(f"Results can't be sorted by {field!r}: it is no metadata field of this database")
        self.cgi["sort_order"] = [sort_order or ["rowid"]]

        if "start_byte" in self.cgi:
            try:
                self.start_byte = int(self["start_byte"])
            except (ValueError, TypeError) as e:
                self.start_byte = ""
            try:
                self.end_byte = int(self["end_byte"])
            except (ValueError, TypeError) as e:
                self.end_byte = ""

        if "full" in self.cgi and self["full"] == "true":
            self.full_bibliography = True
        else:
            self.full_bibliography = False

    def __getattr__(self, key):
        """Return query arg as attribute of class."""
        return self[key]

    def __getitem__(self, key):
        """Return query arg as key of class."""
        if key in self.cgi:
            return self.cgi[key][0]
        elif key in self.defaults:
            return self.defaults[key]
        else:
            return ""

    def __setitem__(self, key, item):
        if key not in self.cgi:
            self.cgi[key] = []
        if isinstance(item, list or set):
            self.cgi[key] = item
        else:
            try:
                self.cgi[key][0] = item
            except IndexError:
                self.cgi[key] = [""]

    def __delattr__(self, name):
        if name in self.cgi:
            del self.cgi[name]
        elif name in self.defaults:
            self.defaults[name] = ""
        else:
            pass

    def __iter__(self):
        """Iterate over query args."""
        for key in list(self.cgi.keys()):
            yield (key, self[key])

    def __repr__(self):
        return repr(self.cgi)

    def __str__(self):
        return " ".join(["{}: {}".format(i, j) for i, j in self])
