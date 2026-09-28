"""hitlist_dir in the global config: every file cached at request time goes there, none in the databases.

Without it, hitlists stay in each database's data/hitlists/. With it, the web app serves a read-only copy of the
test corpus (directories that can't be written, with the corpus's files through symlinks), and every kind of request
that caches something must still succeed, with its cache files in hitlist_dir/<database>/. Neither the copy nor the
corpus may change: that also shows that no temporary file (claims, sorts in progress) was written in the database.
Run as root, which can write read-only directories, only that comparison shows writes.
"""

import ast
import json
import os
import re
import stat
import time
from pathlib import Path
from urllib.parse import urlencode

import pytest

from philologic.runtime.hitlist_dir import _configured_root, default_hitlist_dir, get_hitlist_dir

REPO_ROOT = Path(__file__).parent.parent.parent
HELPER = REPO_ROOT / "python" / "philologic" / "runtime" / "hitlist_dir.py"


@pytest.fixture
def global_config(tmp_path, monkeypatch):
    """Make a global config with the given extra lines the one read."""

    def write(extra):
        path = tmp_path / "philologic5.cfg"
        path.write_text(f'database_root = "{tmp_path}/"\nurl_root = "http://localhost/"\n{extra}', encoding="utf8")
        monkeypatch.setenv("PHILOLOGIC_CONFIG", str(path))
        _configured_root.cache_clear()

    yield write
    _configured_root.cache_clear()


# =============================================================================
# The helper
# =============================================================================


@pytest.mark.parametrize("extra", ["", "hitlist_dir = None\n"], ids=["absent", "None"])
def test_default_is_the_databases_own(global_config, tmp_path, extra):
    """Configs without hitlist_dir (as all older ones) keep hitlists in data/hitlists/, without creating anything."""
    global_config(extra)
    data = tmp_path / "mydb" / "data"
    assert get_hitlist_dir(str(data) + "/") == os.path.join(str(data) + "/", "hitlists") == default_hitlist_dir(str(data) + "/")
    assert get_hitlist_dir(str(data)) == str(data / "hitlists")
    assert not (tmp_path / "mydb").exists()


def test_configured_dir_per_database(global_config, tmp_path):
    """With hitlist_dir, each database has its own directory in it, named after the database, created when needed."""
    global_config(f'hitlist_dir = "{tmp_path}/cache/"\n')
    assert get_hitlist_dir(f"{tmp_path}/dbs/first/data/") == str(tmp_path / "cache" / "first")
    assert get_hitlist_dir(f"{tmp_path}/dbs/first/data") == str(tmp_path / "cache" / "first")
    assert get_hitlist_dir(f"{tmp_path}/dbs/second/data/") == str(tmp_path / "cache" / "second")
    assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == ["first", "second"]
    assert all(p.is_dir() for p in (tmp_path / "cache").iterdir())
    assert not (tmp_path / "dbs").exists()


def test_missing_global_config(monkeypatch, tmp_path):
    monkeypatch.setenv("PHILOLOGIC_CONFIG", str(tmp_path / "none.cfg"))
    _configured_root.cache_clear()
    try:
        assert get_hitlist_dir("/somewhere/mydb/data/") == "/somewhere/mydb/data/hitlists"
    finally:
        _configured_root.cache_clear()


# =============================================================================
# Only the helper names the hitlists directory
# =============================================================================


def hitlist_path_literals(source):
    """Lines of the string literals (docstrings aside) with a "hitlists" path component: where paths are made."""
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and re.search(r"(^|/)hitlists(/|$)", node.value)
    ]


def test_path_literal_check_finds_them():
    for made in (
        'p = db.path + "/hitlists/" + name',
        'p = os.path.join(db_path, "data", "hitlists", name)',
        'p = os.path.join(db_path, "data/hitlists")',
        'p = f"{path}/data/hitlists/{name}"',
    ):
        assert hitlist_path_literals(made) == [1], made
    assert hitlist_path_literals('"""Clear the hitlists directory"""\nlog("the hitlists directory is full")\n') == []


def test_only_the_helper_names_the_hitlists_directory():
    """Every hitlist path goes through philologic.runtime.hitlist_dir: no other code names data/hitlists."""
    sources = [*(REPO_ROOT / "python" / "philologic").rglob("*.py"), *(REPO_ROOT / "www").rglob("*.py")]
    sources = [s for s in sources if s != HELPER and not {"app", "__pycache__"} & set(s.relative_to(REPO_ROOT).parts)]
    assert len(sources) > 50
    found = [
        f"{source.relative_to(REPO_ROOT)}:{line}"
        for source in sources
        for line in hitlist_path_literals(source.read_text(encoding="utf8"))
    ]
    assert found == []


# =============================================================================
# The web app with hitlist_dir set, on a read-only database
# =============================================================================


def read_only_copy(corpus, copy):
    """The corpus's directory tree, with its files as symlinks to the corpus's, and directories no one can write."""
    for dirpath, _, filenames in os.walk(corpus):
        target = copy / os.path.relpath(dirpath, corpus)
        target.mkdir(exist_ok=True)
        for name in filenames:
            (target / name).symlink_to(os.path.join(dirpath, name))
    set_directory_modes(copy, 0o555)


def set_directory_modes(root, mode):
    for dirpath, _, _ in os.walk(root, topdown=False):
        os.chmod(dirpath, mode)


def tree_state(root):
    """Every entry under root with its size and modification time: whatever is written, added or removed shows."""
    state = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = os.path.join(dirpath, name)
            st = os.lstat(path)
            state[os.path.relpath(path, root)] = (stat.S_IFMT(st.st_mode), st.st_size, st.st_mtime_ns)
    return state


def fetch(server, path, params, timeout=300):
    from tests.integration.test_web_concurrency import UnixHTTPConnection  # not at the top: needs gunicorn

    conn = UnixHTTPConnection(server.socket, timeout)
    try:
        conn.request("GET", f"/philologic5/{server.db_name}/{path}?{urlencode(params, doseq=True)}")
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def get_json(server, path, params):
    status, body = fetch(server, path, params)
    assert status == 200, f"{path} {params}: {status} {body[:300]!r}"
    return json.loads(body.splitlines()[-1])  # streamed responses end with the result, after progress lines


class ReadOnlyService:
    """The web app serving a read-only copy of the corpus, with hitlist_dir set."""

    def __init__(self, corpus, tmp_path):
        self.corpus = corpus
        self.tmp_path = tmp_path
        self.root = tmp_path / "dbs"
        self.cache = tmp_path / "cache" / corpus.name  # the corpus's hitlist directory
        self.config = tmp_path / "philologic5.cfg"
        self.config.write_text(
            f'database_root = "{self.root}/"\nurl_root = "http://localhost/philologic5/"\n'
            f'hitlist_dir = "{self.cache.parent}"\n',
            encoding="utf8",
        )
        self.server = None

    def start(self):
        from tests.integration.test_web_concurrency import Server  # not at the top: needs gunicorn

        self.root.mkdir()
        read_only_copy(self.corpus, self.root / self.corpus.name)
        self.before = tree_state(self.root), tree_state(self.corpus)
        self.server = Server(self.root, self.corpus.name, self.tmp_path)

    def stop(self):
        if self.server is not None:
            self.server.stop()
            self.server = None

    def assert_databases_unchanged(self):
        """Stops the server (and its searches still running), then compares the copy and the corpus to before."""
        self.stop()
        assert tree_state(self.root) == self.before[0], "the database directory the web app served changed"
        assert tree_state(self.corpus) == self.before[1], "the corpus changed"


@pytest.fixture
def served_read_only(eltec_db_path, tmp_path, monkeypatch):
    service = ReadOnlyService(Path(eltec_db_path).parent, tmp_path)
    monkeypatch.setenv("PHILOLOGIC_CONFIG", str(service.config))  # the server's global config
    try:
        service.start()
        yield service
    finally:
        service.stop()
        if service.root.exists():
            set_directory_modes(service.root, 0o755)
        _configured_root.cache_clear()  # in case anything here read the global config set above


@pytest.mark.integration
def test_everything_cached_in_hitlist_dir(served_read_only):
    server, cache, corpus = served_read_only.server, served_read_only.cache, served_read_only.corpus
    q = {"q": "love", "method": "proxy", "method_arg": "", "results_per_page": "25"}

    everything = get_json(server, "reports/bibliography.py", {"results_per_page": "25"})
    author = re.findall(r"\w+", everything["results"][0]["metadata_fields"]["author"])[0]
    in_author = {"author": author}
    get_json(server, "reports/bibliography.py", {"results_per_page": "25", **in_author})
    get_json(server, "reports/bibliography.py", {"results_per_page": "25", "sort_by": "title", **in_author})
    assert get_json(server, "reports/concordance.py", q)["results_length"] > 0
    assert get_json(server, "reports/concordance.py", {**q, **in_author})["results_length"] > 0
    get_json(server, "reports/concordance.py", {**q, "sort_by": "title"})
    get_json(server, "reports/kwic.py", q)
    get_json(server, "reports/concordance.py", {**q, "q": "lov", "approximate": "yes", "approximate_ratio": "80"})
    get_json(server, "scripts/get_query_terms.py", q)
    get_json(server, "scripts/get_total_results.py", q)
    get_json(server, "scripts/get_hitlist_stats.py", q)
    get_json(server, "scripts/get_frequency.py", {**q, "frequency_field": "author"})
    get_json(server, "reports/aggregation.py", {**q, "group_by": "author"})
    get_json(server, "reports/time_series.py", {**q, "start_date": "1800", "end_date": "1920", "year_interval": "10"})
    for options in (("left", "right"), ("author", "title")):
        sort = {"first_kwic_sorting_option": options[0], "second_kwic_sorting_option": options[1]}
        assert len(get_json(server, "scripts/get_sorted_kwic.py", {**q, **sort})["results"]) == 25
        get_json(server, "scripts/get_sorted_kwic.py", {**q, **sort, "start": "26", "end": "50"})  # from the cache
    for word_property in ("pos", "lemma"):
        get_json(server, "scripts/get_word_property_count.py", {**q, "word_property": word_property})
        get_json(server, "scripts/get_word_property_count.py", {**q, "word_property": word_property, **in_author})

    colloc = {"q": "love", "colloc_filter_choice": "frequency", "colloc_within": "sent", "method_arg": ""}
    whole = get_json(server, "reports/collocation.py", colloc)["file_path"]
    by_author = get_json(
        server, "reports/collocation.py", {**colloc, "filter_frequency": "100", "map_field": "author"}
    )["file_path"]
    by_year = get_json(
        server,
        "reports/collocation.py",
        {**colloc, "collocation_method": "timeSeries", "time_series_interval": "10", "map_field": "year"},
    )["file_path"]
    from philologic.runtime.reports.collocation import load_map_field_cache

    one_author = load_map_field_cache(by_author)[3][0]
    distribution = get_json(
        server, "scripts/get_collocate_distribution.py", {"file_path": by_author, "field": one_author}
    )["file_path"]
    get_json(server, "scripts/comparative_collocations.py",
             {"primary_file_path": whole, "other_file_path": distribution, "whole_corpus": "false"})
    get_json(server, "scripts/get_similar_collocate_distributions.py", {"primary_file_path": whole, "file_path": by_author})
    get_json(server, "scripts/collocation_time_series.py", {"file_path": by_year, "year_interval": "10", "period_number": "0"})

    for file_path in (whole, by_author, by_year, distribution):
        assert os.path.dirname(file_path) == str(cache)
    assert sorted(os.listdir(cache.parent)) == [corpus.name]
    cached = os.listdir(cache)
    for kind in (".hitlist", ".hitlist.done", ".hitlist.terms", ".sorted.title", ".kwic.sorted", ".pickle",
                 ".npz", ".approximate_terms"):
        assert any(name.endswith(kind) for name in cached), f"no {kind} file in {cache}"
    assert "time_series_year_data.npz" in cached
    assert server.errors() == []
    served_read_only.assert_databases_unchanged()


@pytest.mark.integration
def test_cleanup_in_hitlist_dir(served_read_only):
    """The hitlist cleanup removes old files from the database's hitlist_dir directory, and only from there."""
    server, cache = served_read_only.server, served_read_only.cache
    other_database = cache.parent / "other_database"
    other_database.mkdir()
    old = time.time() - 3600
    for path in (cache / "old.hitlist", cache / "new.hitlist", other_database / "old.hitlist"):
        path.write_bytes(b"")
    os.utime(cache / "old.hitlist", (old, old))
    os.utime(other_database / "old.hitlist", (old, old))
    for _ in range(500):  # the cleanup runs after one request in eleven
        assert fetch(server, "scripts/get_web_config.py", {})[0] == 200
        if not (cache / "old.hitlist").exists():
            break
    assert not (cache / "old.hitlist").exists()
    assert (cache / "new.hitlist").exists()
    assert (other_database / "old.hitlist").exists()
    served_read_only.assert_databases_unchanged()
