#!/var/lib/philologic5/philologic_env/bin/python3
"""Parses queries stored in the environ object."""


import hashlib
import urllib.parse
from http.cookies import SimpleCookie

from philologic.runtime.DB import DB
from philologic.runtime.find_similar_words import find_similar_words
from philologic.runtime.Query import query_parse


class WSGIHandler(object):
    """Class which parses the environ object and massages query arguments for PhiloLogic5."""

    def __init__(self, environ, config):
        """Initialize class."""
        # Create db object to access config variables
        db = DB(config.db_path + "/data/")
        self.path_info = environ.get("PATH_INFO", "")
        self.query_string = environ["QUERY_STRING"]
        self.script_filename = environ["SCRIPT_FILENAME"]
        self.db_path = "/".join(environ["SCRIPT_NAME"].split("/")[:-2])

        self.authenticated = False
        if "HTTP_COOKIE" in environ:
            self.cookies = SimpleCookie(
                "".join(environ["HTTP_COOKIE"].split())
            )  # remove all whitespace in Cookie since it breaks parsing in Python 3.6
            if "hash" in self.cookies and "timestamp" in self.cookies:
                h = hashlib.md5()
                # h.update(environ["REMOTE_ADDR"].encode("utf8"))
                h.update(self.cookies["timestamp"].value.encode("utf8"))
                h.update(db.locals.secret.encode("utf8"))
                if self.cookies["hash"].value == h.hexdigest():
                    self.authenticated = True
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
                self.cgi["original_q"] = self.cgi["q"][:]
                self.cgi["q"][0] = find_similar_words(db, config, self)
            if self.cgi["q"][0] != "":
                self.no_q = False
            else:
                self.no_q = True
            # self.cgi['q'][0] = self.cgi['q'][0].encode('utf8')
        else:
            self.no_q = True

        words = [w for w in self.q.split() if w]
        method = self["method"] or "proxy"
        try:
            self.arg = int(self["method_arg"])
        except ValueError:
            self.arg = 0
        if len(words) == 1:
            method = "single_term"
        elif self.arg == 0 and method in ("proxy", "exact_cooc"):
            if self.cooc_order == "yes":
                method = "phrase_ordered"
            else:
                method = "phrase_unordered"
        if method == "proxy":
            if self.cooc_order == "yes" and self.arg > 0:  # Co-occurrence search within n words ordered
                method = "proxy_ordered"
            else:  # Co-occurrence search within n words unordered
                method = "proxy_unordered"
        elif method == "exact_cooc":
            if self.cooc_order == "yes":
                method = "exact_cooc_ordered"
            else:
                method = "exact_cooc_unordered"
        elif method == "sentence":
            self.arg = 6
            if self.cooc_order == "yes":
                method = "sentence_ordered"
            else:
                method = "sentence_unordered"
        self.arg = str(self.arg)
        self.cgi["arg"] = [self.arg]
        self.cgi["method"] = [method]

        self.metadata_fields = db.locals["metadata_fields"]
        self.metadata = {}
        num_empty = 0

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

        for field in self.metadata_fields:
            if field in self.cgi and self.cgi[field]:
                # Hack to remove hyphens in Frantext
                if db.locals["metadata_sql_types"][field] not in ("int", "date") and isinstance(
                    self.cgi[field][0], str
                ):
                    if not self.cgi[field][0].startswith('"') and field != "filename":
                        self.cgi[field][0] = query_parse(self.cgi[field][0], config)
                # these ifs are to fix the no results you get when you do a
                # metadata query
                if self["q"] != "":
                    self.metadata[field] = self.cgi[field][0]
                elif self.cgi[field][0] != "":
                    self.metadata[field] = self.cgi[field][0]
            # in case of an empty query
            if field not in self.cgi or not self.cgi[field][0]:
                num_empty += 1

        self.metadata["philo_type"] = self["philo_type"]

        if num_empty == len(self.metadata_fields):
            self.no_metadata = True
        else:
            self.no_metadata = False

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
