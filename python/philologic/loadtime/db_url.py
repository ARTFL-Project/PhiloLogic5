"""Find the URL a loaded database is served at, by trying the likely ones on this machine.

The web app takes its host and URL prefix from the request, so nothing in PhiloLogic's own config says what they are.
We try this machine's host names and localhost under /philologic5/, then gunicorn's own port if it listens on TCP,
and return the first URL that serves the database. A 200 isn't enough, since the web server may list the raw database
directory there: only the web app answers with an index.html whose <base href> ends with /<db>/.

This only feeds a line printed at the end of a load, so nothing here may raise: db_url() returns None."""

import re
import socket
import urllib.request

GUNICORN_CONF = "/var/lib/philologic5/web_app/gunicorn.conf.py"


def candidates(dbname, gunicorn_conf=GUNICORN_CONF):
    hosts = list(dict.fromkeys([socket.getfqdn(), socket.gethostname()]))
    urls = [f"{scheme}://{host}/philologic5/{dbname}/" for host in hosts for scheme in ("https", "http")]
    urls.append(f"http://localhost/philologic5/{dbname}/")
    try:
        with open(gunicorn_conf) as f:
            bind = re.search(r"""^bind\s*=\s*["']([^"']+)["']""", f.read(), re.M).group(1)
        if not bind.startswith("unix:"):
            host, port = bind.rsplit(":", 1)
            urls.append(f"http://{host if host not in ('', '0.0.0.0') else 'localhost'}:{port}/{dbname}/")
    except (OSError, AttributeError, ValueError):
        pass
    return urls


def probe(url, dbname, timeout=2):
    """The final URL if url serves the web app's page for dbname, else None."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            page = response.read(65536).decode("utf-8", "replace")
            if re.search(rf"""<base\s[^>]*href=["'][^"']*/{re.escape(dbname)}/["']""", page):
                return response.geturl()
    except Exception:
        pass
    return None


def db_url(dbname, gunicorn_conf=GUNICORN_CONF, timeout=2):
    try:
        return next(filter(None, (probe(url, dbname, timeout) for url in candidates(dbname, gunicorn_conf))), None)
    except Exception:
        return None
