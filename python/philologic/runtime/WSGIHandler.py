#!/var/lib/philologic5/philologic_env/bin/python3
"""Parses queries stored in the environ object."""


import hashlib
import urllib.parse
from http.cookies import SimpleCookie

from philologic.runtime.Query import query_parse, resolve_method


def check_cookie_auth(environ, secret):
    """Validate auth cookie. Returns True if authenticated."""
    if "HTTP_COOKIE" not in environ:
        return False
    cookies = SimpleCookie("".join(environ["HTTP_COOKIE"].split()))
    if "hash" not in cookies or "timestamp" not in cookies:
        return False
    h = hashlib.md5()
    h.update(cookies["timestamp"].value.encode("utf8"))
    h.update(secret.encode("utf8"))
    return cookies["hash"].value == h.hexdigest()


def expand_approximate_query(request, config):
    """Expand query terms for approximate/fuzzy search. Requires DB access."""
    from philologic.runtime.DB import DB
    from philologic.runtime.find_similar_words import find_similar_words

    request.cgi["original_q"] = request.cgi["q"][:]
    db = DB(config.db_path + "/data/")
    request.cgi["q"][0] = find_similar_words(db, config, request)


def parse_metadata(cgi, q, metadata_fields, metadata_sql_types, config):
    """Parse metadata fields from query params. Returns (metadata_dict, no_metadata)."""
    metadata = {}
    num_empty = 0
    for field in metadata_fields:
        if field in cgi and cgi[field]:
            if metadata_sql_types[field] not in ("int", "date") and isinstance(cgi[field][0], str):
                if not cgi[field][0].startswith('"') and field != "filename":
                    cgi[field][0] = query_parse(cgi[field][0], config)
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

        self.authenticated = check_cookie_auth(environ, config.db_locals.secret)
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
            if "approximate_ratio" in self.cgi:
                self.approximate_ratio = float(self.cgi["approximate_ratio"][0]) / 100
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

        method, self.arg = resolve_method(self.q, self["method"], self["method_arg"], self.cooc_order)
        self.cgi["arg"] = [self.arg]
        self.cgi["method"] = [method]

        self.metadata_fields = config.db_locals["metadata_fields"]

        self.start = int(self["start"] or 0)
        self.end = int(self["end"] or 0)
        self.results_per_page = int(self["results_per_page"])
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

        if "sort_order" in self.cgi:
            sort_order = []
            for metadata in self.cgi["sort_order"]:
                sort_order.append(metadata)
            self.cgi["sort_order"][0] = sort_order
        else:
            self.cgi["sort_order"] = [["rowid"]]

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
            self.defaults[key] = ""
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
