"""Unit tests for the web app's static files (www/resources/static.py): only what a route's own directory holds is
served, not the rest of the database directory (data/logins.txt, db.locals.py, the texts...) or anything outside it."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

pytest.importorskip("falcon")

from tests.fixtures.web_app import web_app_client


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A test client of the web app, on a database root holding a database, "mydb", whose name begins that of
    another, "mydb_other", and a file next to the root."""
    base = tmp_path_factory.mktemp("static")
    root = base / "root"
    for db in ("mydb", "mydb_other"):
        (root / db / "app" / "dist" / "assets").mkdir(parents=True)
        (root / db / "app" / "dist" / "img").mkdir()
        (root / db / "data").mkdir()
        (root / db / "data" / "db.locals.py").write_text("metadata_sql_types = {}\nmetadata_fields = []\n")
        (root / db / "data" / "logins.txt").write_text("secret")
    (root / "mydb" / "app" / "dist" / "assets" / "index.js").write_text("console.log('app')")
    (root / "mydb" / "app" / "dist" / "assets" / "index.js.br").write_bytes(b"compressed")
    (root / "mydb" / "app" / "dist" / "img" / "logo.png").write_bytes(b"png")
    (root / "mydb" / "favicon.ico").write_bytes(b"ico")
    (root / "mydb_other" / "app" / "dist" / "assets" / "other.js").write_text("other")
    (base / "outside.txt").write_text("outside")
    with web_app_client(root) as client:
        yield client


@pytest.mark.unit
class TestStaticFiles:
    @pytest.mark.parametrize(
        "path, content",
        [
            ("/philologic5/mydb/assets/index.js", b"console.log('app')"),
            ("/philologic5/mydb/img/logo.png", b"png"),
            ("/philologic5/mydb/favicon.ico", b"ico"),
            ("/philologic5/mydb_other/assets/other.js", b"other"),
        ],
    )
    def test_serves_the_app_files(self, client, path, content):
        resp = client.simulate_get(path)
        assert resp.status_code == 200
        assert resp.content == content

    def test_serves_brotli_when_accepted(self, client):
        resp = client.simulate_get("/philologic5/mydb/assets/index.js", headers={"Accept-Encoding": "br"})
        assert resp.status_code == 200
        assert resp.headers["Content-Encoding"] == "br"
        assert resp.content == b"compressed"

    @pytest.mark.parametrize(
        "path",
        [
            # The paths as the app gets them, with %2e and %2f decoded by the WSGI server
            "/philologic5/mydb/assets/../../../data/logins.txt",
            "/philologic5/mydb/assets/../../../data/db.locals.py",
            "/philologic5/mydb/img/../../../data/logins.txt",
            "/philologic5/mydb/assets/../../../favicon.ico",
            # In a database whose name begins with this one's
            "/philologic5/mydb/assets/../../../../mydb_other/data/logins.txt",
            "/philologic5/mydb/assets/../../../../mydb_other/app/dist/assets/other.js",
            # Outside the database root
            "/philologic5/mydb/assets/../../../../../outside.txt",
        ],
    )
    def test_no_file_outside_the_routes_directory(self, client, path):
        resp = client.simulate_get(path)
        assert resp.status_code == 403
        assert b"secret" not in resp.content and b"outside" not in resp.content

    @pytest.mark.parametrize(
        "path",
        [
            "/philologic5/../assets/../outside.txt",
            "/philologic5/./assets/../outside.txt",
            "/philologic5/../favicon.ico",
        ],
    )
    def test_dot_segments_name_no_database(self, client, path):
        """"." and ".." are directories of the database root, but they name no database."""
        resp = client.simulate_get(path)
        assert resp.status_code == 404
        assert b"secret" not in resp.content and b"outside" not in resp.content
