"""Unit tests for the web app's index.html: served with the database's own path as its <base href>, which the client
(built for no host or URL prefix) takes its paths from, so no url_root setting is needed."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "www"))

from resources.spa import with_base  # noqa: E402

INDEX = b'<!DOCTYPE html><html><head><base href="/" /><meta charset="utf-8" /></head><body></body></html>'


@pytest.mark.unit
class TestWithBase:
    def test_replaces_the_base(self):
        assert with_base(INDEX, "/philologic5/mydb/").count(b"<base ") == 1
        assert b'<base href="/philologic5/mydb/" />' in with_base(INDEX, "/philologic5/mydb/")

    def test_adds_one_to_a_custom_page_without_one(self):
        page = with_base(b"<html><head><title>x</title></head></html>", "/mydb/")
        assert page == b'<html><head><base href="/mydb/" /><title>x</title></head></html>'

    def test_escaped(self):
        assert b'<base href="/a&quot;b/" />' in with_base(INDEX, '/a"b/')


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    pytest.importorskip("falcon")
    from tests.fixtures.web_app import web_app_client

    root = tmp_path_factory.mktemp("spa") / "root"
    (root / "mydb" / "data").mkdir(parents=True)
    (root / "mydb" / "data" / "db.locals.py").write_text("metadata_sql_types = {}\nmetadata_fields = []\n")
    (root / "mydb" / "data" / "web_config.cfg").write_text("access_control = False\n")
    (root / "mydb" / "app" / "dist").mkdir(parents=True)
    (root / "mydb" / "app" / "dist" / "index.html").write_bytes(INDEX)
    (root / "mydb" / "app" / "dist" / "index.html.br").write_bytes(b"not the page")
    with web_app_client(root) as client:
        yield client


@pytest.mark.unit
class TestServedIndex:
    @pytest.mark.parametrize(
        "path, base",
        [
            ("/philologic5/mydb/", "/philologic5/mydb/"),
            ("/philologic5/mydb/navigate/1/2", "/philologic5/mydb/"),  # a route of the client, any depth
            ("/other/prefix/mydb/concordance", "/other/prefix/mydb/"),  # whatever prefix the proxy passes
            ("/mydb/", "/mydb/"),  # none
        ],
    )
    def test_base_is_the_database_path(self, client, path, base):
        resp = client.simulate_get(path, headers={"Accept-Encoding": "gzip, br"})
        assert resp.status_code == 200
        assert f'<base href="{base}" />' in resp.text
        assert resp.headers.get("Content-Encoding") is None  # written as it is served: not the .br file
