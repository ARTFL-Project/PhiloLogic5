"""Tests for finding a database's URL by probing candidate URLs (philologic.loadtime.db_url)."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from philologic.loadtime import db_url as mod

DB = "mydb"


def conf(tmp_path, bind):
    path = tmp_path / "gunicorn.conf.py"
    path.write_text(f'workers = 2\nbind = "{bind}"\n')
    return str(path)


@pytest.fixture
def names(monkeypatch):
    monkeypatch.setattr(mod.socket, "getfqdn", lambda: "a.org")
    monkeypatch.setattr(mod.socket, "gethostname", lambda: "a")


def test_candidate_order(names, tmp_path):
    assert mod.candidates(DB, conf(tmp_path, "unix:/run/g.sock")) == [
        f"https://a.org/philologic5/{DB}/", f"http://a.org/philologic5/{DB}/",
        f"https://a/philologic5/{DB}/", f"http://a/philologic5/{DB}/",
        f"http://localhost/philologic5/{DB}/",
    ]


def test_tcp_bind_comes_last(names, tmp_path):
    assert mod.candidates(DB, conf(tmp_path, "127.0.0.1:8080"))[-1] == f"http://127.0.0.1:8080/{DB}/"
    assert mod.candidates(DB, conf(tmp_path, "0.0.0.0:8080"))[-1] == f"http://localhost:8080/{DB}/"


def serve(body, status=200):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/philologic5/{DB}/"


def test_page_with_base_tag_is_found():
    server, url = serve(f'<head><base href="/philologic5/{DB}/" /></head>')
    assert mod.probe(url, DB) == url
    assert mod.probe(url, "other") is None
    server.shutdown()


def test_listing_without_base_tag_is_not_found():
    server, url = serve("<title>Index of /philologic5/mydb</title>")
    assert mod.probe(url, DB) is None
    server.shutdown()


def test_404_and_refused_port():
    server, url = serve("not found", status=404)
    assert mod.probe(url, DB) is None
    server.shutdown()
    server.server_close()
    assert mod.probe(url, DB) is None
