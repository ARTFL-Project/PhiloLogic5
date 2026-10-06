"""Unit tests for access control (python/philologic/runtime/access_control.py and its use by the web app): who the client
is behind a proxy, the IP and domain checks, auth cookies, and the refusal of every request but the login screen's."""

import hashlib
import hmac
import json
import os
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime import access_control
from philologic.runtime.access_control import (
    auth_cookie,
    check_login_info,
    client_address,
    connected_address,
    database_key,
    is_allowed,
    is_authenticated,
)


HOSTNAME_DOMAIN = access_control._client_domain  # the real one, which fresh_state replaces


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch, tmp_path):
    """No remembered answers, compiled whitelists in the test's directory, and no DNS."""
    monkeypatch.setattr(access_control, "_allowed", {})
    monkeypatch.setattr(access_control, "_COMPILED_IP_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(access_control, "_client_domain", lambda address: "unknown.example")


def make_config(db_path, secret="", access_file=""):
    return SimpleNamespace(db_path=str(db_path), db_locals=SimpleNamespace(secret=secret), access_file=access_file)


def cookie_header(set_cookie):
    """The Cookie header a browser sends back for a Set-Cookie value."""
    return set_cookie.split(";")[0]


@pytest.mark.unit
class TestClientAddress:
    def test_direct_client(self):
        assert client_address({"REMOTE_ADDR": "203.0.113.7"}) == "203.0.113.7"

    def test_direct_client_cannot_claim_another_address(self):
        environ = {"REMOTE_ADDR": "203.0.113.7", "HTTP_X_FORWARDED_FOR": "127.0.0.1"}
        assert client_address(environ) == "203.0.113.7"

    @pytest.mark.parametrize("proxy", ["", "127.0.0.1", "::1"])
    def test_behind_a_proxy_the_first_address(self, proxy):
        """As in 5.2.6, the first address: a gateway before the proxy may have written its user's address there."""
        environ = {"REMOTE_ADDR": proxy, "HTTP_X_FORWARDED_FOR": "198.51.100.4, 10.0.0.1, 203.0.113.7"}
        assert client_address(environ) == "198.51.100.4"

    @pytest.mark.parametrize("proxy", ["", "127.0.0.1", "::1"])
    def test_connected_address_the_one_the_proxy_appended(self, proxy):
        """The proxy appends the address it got the request from: what comes before is the client's say."""
        environ = {"REMOTE_ADDR": proxy, "HTTP_X_FORWARDED_FOR": "127.0.0.1, 10.0.0.1, 203.0.113.7"}
        assert connected_address(environ) == "203.0.113.7"

    def test_connected_address_behind_several_proxies(self):
        environ = {"REMOTE_ADDR": "", "HTTP_X_FORWARDED_FOR": "127.0.0.1, 203.0.113.7, 127.0.0.1"}
        assert connected_address(environ) == "203.0.113.7"

    def test_proxy_without_forwarded_for(self):
        assert client_address({"REMOTE_ADDR": "127.0.0.1"}) == "127.0.0.1"


# The cases of the access file below, with local networks off: (description, address, domain, allowed)
ACCESS_FILE = """
domain_list = [".edu", "example.com"]
blocked_ips = ["10.0.0.99", "203.0.113.100", "192.168.5.5"]
allowed_ips = [
    "192.168.1.1",              # Exact IP match
    "172.16.0.0/16",            # CIDR notation
    "192.168.2.10-20",          # Range in last octet
    "192.168-170.0.0",          # Range in non-last octet
    "192.168.4-5.10-20",        # Multiple ranges
    "172.18",                   # Network prefix
    "134.226",                  # Network prefix with 2 octets
    "192.70.186",               # Network prefix with 3 octets
    "140.141.",                 # Network prefix with trailing dot
    "216.87.19-20",             # Should match 216.87.19.* and 216.87.20.*
]
"""
ADDRESS_CASES = [
    ("Exact IP match", "192.168.1.1", "example.org", True),
    ("Exact IP non-match", "192.168.1.2", "example.org", False),
    ("CIDR match start", "172.16.0.1", "example.org", True),
    ("CIDR match end", "172.16.255.255", "example.org", True),
    ("CIDR non-match", "172.17.0.1", "example.org", False),
    ("Last octet range match start", "192.168.2.10", "example.org", True),
    ("Last octet range match middle", "192.168.2.15", "example.org", True),
    ("Last octet range match end", "192.168.2.20", "example.org", True),
    ("Last octet range non-match below", "192.168.2.9", "example.org", False),
    ("Last octet range non-match above", "192.168.2.21", "example.org", False),
    ("Non-last octet range match start", "192.168.0.0", "example.org", True),
    ("Non-last octet range match end", "192.170.0.0", "example.org", True),
    ("Non-last octet range non-match", "192.171.0.0", "example.org", False),
    ("Multiple ranges match", "192.168.4.15", "example.org", True),
    ("Multiple ranges match edge", "192.168.5.20", "example.org", True),
    ("Multiple ranges non-match", "192.168.6.15", "example.org", False),
    ("Network prefix match", "172.18.0.1", "example.org", True),
    ("Network prefix match edge", "172.18.255.255", "example.org", True),
    ("Network prefix non-match", "172.19.0.1", "example.org", False),
    ("Local network 10.x.x.x", "10.1.2.3", "example.org", False),
    ("Local network 172.16.x.x", "172.16.5.10", "example.org", True),  # by the CIDR rule
    ("Local network 192.168.x.x", "192.168.0.1", "example.org", False),
    ("Local network 127.x.x.x", "127.0.0.1", "example.org", False),
    ("Domain exact match", "203.0.113.1", "example.com", True),
    ("Domain suffix match", "203.0.113.1", "university.edu", True),
    ("Domain non-match", "203.0.113.1", "example.net", False),
    ("Blocked IP", "10.0.0.99", "example.org", False),
    ("Blocked IP overrides allowed network", "192.168.5.5", "example.org", False),
    ("Network prefix with 2 octets", "134.226.12.34", "example.org", True),
    ("Network prefix with 3 octets", "192.70.186.50", "example.org", True),
    ("Network prefix with trailing dot", "140.141.10.20", "example.org", True),
    ("Network prefix with trailing dot non-match", "140.142.10.20", "example.org", False),
    ("Multiple adjacent octets range", "216.87.19.50", "example.org", True),
    ("Multiple adjacent octets range", "216.87.20.50", "example.org", True),
]


@pytest.mark.unit
class TestAddressChecks:
    @pytest.mark.parametrize("description, address, domain, allowed", ADDRESS_CASES, ids=[c[0] for c in ADDRESS_CASES])
    def test_access_file(self, monkeypatch, tmp_path, description, address, domain, allowed):
        monkeypatch.setattr(access_control, "local_networks", [])
        monkeypatch.setattr(access_control, "_client_domain", lambda _: domain)
        access_file = tmp_path / "access.py"
        access_file.write_text(ACCESS_FILE)
        config = make_config(tmp_path / "db", access_file=str(access_file))
        assert is_allowed({"REMOTE_ADDR": address}, config) is allowed

    def test_local_networks_are_allowed(self, tmp_path):
        access_file = tmp_path / "access.py"
        access_file.write_text("allowed_ips = []\n")
        config = make_config(tmp_path / "db", access_file=str(access_file))
        assert is_allowed({"REMOTE_ADDR": "10.1.2.3"}, config)

    def test_no_access_file_lets_nobody_in(self, tmp_path):
        assert not is_allowed({"REMOTE_ADDR": "10.1.2.3"}, make_config(tmp_path / "db"))
        assert not is_allowed({"REMOTE_ADDR": "10.1.2.3"}, make_config(tmp_path / "db", access_file="missing.py"))

    def test_answer_follows_the_access_file(self, tmp_path):
        """The answer is remembered for each version of the access file only."""
        access_file = tmp_path / "access.py"
        access_file.write_text('allowed_ips = ["198.51.100.7"]\n')
        config = make_config(tmp_path / "db", access_file=str(access_file))
        assert is_allowed({"REMOTE_ADDR": "198.51.100.7"}, config)
        access_file.write_text("allowed_ips = []\n")
        stat = access_file.stat()
        os.utime(access_file, (stat.st_atime, stat.st_mtime + 10))
        assert not is_allowed({"REMOTE_ADDR": "198.51.100.7"}, config)


@pytest.mark.unit
class TestDomains:
    """Domains match as before 5.2.6: by substring of the client's domain, from a reverse DNS lookup the forward lookup
    doesn't confirm. Stricter rules would have refused subscribers in the 2025 logs of artflsrv04: these cases."""

    @pytest.mark.parametrize(
        "fqdn, domain_list",
        [
            ("0587661688.vpn.umich.net", ["mich.net"]),  # UMich's VPN, by MichNet's entry
            ("bul-pr-proxy02.bibl.ulaval.ca", ["laval.ca"]),
            ("lib-ezproxy-01.oit.duke.edu", ["duke.edu"]),  # a name with no forward record
            ("cornell.idm.oclc.org", ["idm.oclc.org"]),  # OCLC-hosted EZproxy
            ("ezproxy.stjohnscollege.edu", ["stjohnscollege"]),
        ],
    )
    def test_allowed(self, monkeypatch, tmp_path, fqdn, domain_list):
        monkeypatch.setattr(access_control, "_client_domain", HOSTNAME_DOMAIN)
        monkeypatch.setattr(access_control.socket, "getfqdn", lambda address: fqdn)
        access_file = tmp_path / "access.py"
        access_file.write_text(f"domain_list = {domain_list!r}\n")
        assert is_allowed({"REMOTE_ADDR": "203.0.113.7"}, make_config(tmp_path / "db", access_file=str(access_file)))

    @pytest.mark.parametrize(
        "fqdn, domain",
        [("cs.uchicago.edu", "uchicago.edu"), ("uchicago.edu", "uchicago.edu"), ("host.cs.ox.ac.uk", "host.cs.ox.ac.uk"),
         ("example.org", "example.org"), ("203.0.113.7", "203.0.113.7")],
    )
    def test_client_domain(self, monkeypatch, fqdn, domain):
        monkeypatch.setattr(access_control.socket, "getfqdn", lambda address: fqdn)
        assert HOSTNAME_DOMAIN("203.0.113.7") == domain

    def test_no_lookup_without_domains(self, monkeypatch, tmp_path):
        def lookup(address):
            raise AssertionError("no domain_list, no DNS")

        monkeypatch.setattr(access_control, "_client_domain", lookup)
        access_file = tmp_path / "access.py"
        access_file.write_text('allowed_ips = ["198.51.100.0/24"]\n')
        assert not is_allowed({"REMOTE_ADDR": "203.0.113.7"}, make_config(tmp_path / "db", access_file=str(access_file)))


@pytest.mark.unit
class TestLogins:
    @pytest.fixture
    def config(self, tmp_path):
        (tmp_path / "db" / "data").mkdir(parents=True)
        (tmp_path / "db" / "data" / "logins.txt").write_text(
            "\nno-password\nuser\tpass\nother\t%41é b\n", encoding="utf8"
        )
        return make_config(tmp_path / "db")

    @pytest.mark.parametrize(
        "username, password, valid",
        [
            ("user", "pass", True),
            ("other", "%41é b", True),  # as typed: not URL-decoded a second time
            ("other", "Aé b", False),
            ("user", "wrong", False),
            ("pass", "user", False),
            ("no-password", "", False),
            ("nobody", "pass", False),
        ],
    )
    def test_logins(self, config, username, password, valid):
        assert check_login_info(config, username, password) is valid

    def test_no_logins_file(self, tmp_path):
        assert not check_login_info(make_config(tmp_path / "db"), "user", "pass")


@pytest.fixture
def installation_secret(monkeypatch, tmp_path):
    """An installation secret of the test's, next to a global config of its own."""
    config_path = tmp_path / "philologic5.cfg"
    config_path.write_text("")
    (tmp_path / "philologic5.secret").write_text("installation-secret\n")
    monkeypatch.setenv("PHILOLOGIC_CONFIG", str(config_path))
    access_control._installation_secret.cache_clear()
    yield b"installation-secret"
    access_control._installation_secret.cache_clear()


@pytest.mark.unit
class TestAuthCookies:
    def test_cookie_lets_its_client_in(self, installation_secret, tmp_path):
        config = make_config(tmp_path / "mydb")
        assert is_authenticated({"HTTP_COOKIE": cookie_header(auth_cookie(config))}, config)

    def test_cookie_attributes(self, installation_secret, tmp_path):
        attributes = [a.strip() for a in auth_cookie(make_config(tmp_path / "mydb")).split(";")[1:]]
        assert "HttpOnly" in attributes and "Path=/" in attributes and "SameSite=Lax" in attributes
        assert f"Max-Age={access_control.AUTH_COOKIE_MAX_AGE}" in attributes

    def test_no_cookie(self, installation_secret, tmp_path):
        assert not is_authenticated({}, make_config(tmp_path / "mydb"))

    def test_cookie_of_another_database(self, installation_secret, tmp_path):
        """Not even under this database's name: each database has its own key."""
        cookie = cookie_header(auth_cookie(make_config(tmp_path / "otherdb")))
        config = make_config(tmp_path / "mydb")
        assert not is_authenticated({"HTTP_COOKIE": cookie}, config)
        renamed = "philologic5_mydb=" + cookie.split("=", 1)[1]
        assert not is_authenticated({"HTTP_COOKIE": renamed}, config)

    def test_tampered_cookie(self, installation_secret, tmp_path):
        config = make_config(tmp_path / "mydb")
        name, value = cookie_header(auth_cookie(config)).split("=", 1)
        timestamp, signature = value.split(".")
        other_digit = "1" if signature.endswith("0") else "0"
        for forged in (f"{int(timestamp) + 1}.{signature}", f"{timestamp}.{signature[:-1]}{other_digit}", f"{timestamp}.", timestamp):
            assert not is_authenticated({"HTTP_COOKIE": f"{name}={forged}"}, config)

    def test_cookie_without_secret_cannot_be_forged(self, installation_secret, tmp_path):
        """The cookies of a database without a secret of its own were md5(timestamp) before: anyone could make one."""
        config = make_config(tmp_path / "mydb")
        timestamp = str(int(time.time()))
        old = f"hash={hashlib.md5(timestamp.encode()).hexdigest()}; timestamp={timestamp}"
        unsigned = f"philologic5_mydb={timestamp}.{hmac.new(b'', f'mydb\0{timestamp}'.encode(), 'sha256').hexdigest()}"
        assert not is_authenticated({"HTTP_COOKIE": old}, config)
        assert not is_authenticated({"HTTP_COOKIE": unsigned}, config)

    def test_cookie_expires(self, installation_secret, monkeypatch, tmp_path):
        config = make_config(tmp_path / "mydb")
        cookie = cookie_header(auth_cookie(config))
        now = time.time()
        monkeypatch.setattr(access_control.time, "time", lambda: now + access_control.AUTH_COOKIE_MAX_AGE + 1)
        assert not is_authenticated({"HTTP_COOKIE": cookie}, config)
        monkeypatch.setattr(access_control.time, "time", lambda: now - 3600)  # made in the future
        assert not is_authenticated({"HTTP_COOKIE": cookie}, config)

    def test_cookie_among_others(self, installation_secret, tmp_path):
        """Other sites of the host may set any cookie, even some http.cookies can't parse."""
        config = make_config(tmp_path / "mydb")
        cookies = f'other="a b; broken=x=y; philologic5_mydb=garbage; {cookie_header(auth_cookie(config))}; last=1'
        assert is_authenticated({"HTTP_COOKIE": cookies}, config)

    def test_database_secret_comes_first(self, installation_secret, tmp_path):
        assert database_key(make_config(tmp_path / "mydb", secret="own")) == b"own"

    def test_derived_keys(self, installation_secret, tmp_path):
        key = database_key(make_config(tmp_path / "mydb"))
        assert key == hmac.new(installation_secret, b"mydb", hashlib.sha256).digest()
        assert key != database_key(make_config(tmp_path / "otherdb"))

    def test_without_installation_secret(self, monkeypatch, tmp_path):
        """Cookies are still signed, with the secret drawn when the web app was loaded."""
        monkeypatch.setenv("PHILOLOGIC_CONFIG", str(tmp_path / "philologic5.cfg"))
        access_control._installation_secret.cache_clear()
        try:
            config = make_config(tmp_path / "mydb")
            expected = hmac.new(access_control._FALLBACK_SECRET, b"mydb", hashlib.sha256).digest()
            assert database_key(config) == expected
            assert is_authenticated({"HTTP_COOKIE": cookie_header(auth_cookie(config))}, config)
        finally:
            access_control._installation_secret.cache_clear()


# ── The web app ───────────────────────────────────────────────────────────────

ALLOWED, DENIED = "198.51.100.7", "203.0.113.7"


@pytest.fixture(scope="module")
def web_root(tmp_path_factory):
    """A database root with an access-controlled database, "closed", which lets ALLOWED in, and an "open" one."""
    base = tmp_path_factory.mktemp("access")
    (base / "access.py").write_text(f'allowed_ips = ["{ALLOWED}"]\n')
    root = base / "root"
    for name, web_config in (
        ("closed", f'access_control = True\naccess_file = "{base / "access.py"}"\n'),
        ("open", "access_control = False\n"),
    ):
        (root / name / "data").mkdir(parents=True)
        (root / name / "data" / "db.locals.py").write_text("metadata_sql_types = {}\nmetadata_fields = []\n")
        (root / name / "data" / "web_config.cfg").write_text(web_config)
        (root / name / "data" / "logins.txt").write_text("user\tpass\n")
        with closing(sqlite3.connect(root / name / "data" / "toms.db")) as dbh:
            dbh.execute("CREATE TABLE toms (philo_id text)")
        (root / name / "app" / "dist" / "assets").mkdir(parents=True)
        (root / name / "app" / "dist" / "index.html").write_text("<html>app</html>")
        (root / name / "app" / "dist" / "assets" / "index.js").write_text("app")
    return root


@pytest.fixture(scope="module")
def client(web_root):
    pytest.importorskip("falcon")
    from tests.fixtures.web_app import web_app_client

    with web_app_client(web_root) as client:
        yield client


def get(client, path, remote_addr, **kwargs):
    """A request from remote_addr ("" for a unix socket: a proxy)."""
    extras = kwargs.pop("extras", {})
    extras["REMOTE_ADDR"] = remote_addr
    return client.simulate_get(f"/philologic5/{path}", extras=extras, **kwargs)


RESTRICTED = ["scripts/get_custom_landing_page.py", "reports/concordance.py?q=a", "scripts/get_total_results.py?q=a",
              "scripts/export_results.py?q=a&report=concordance", "scripts/get_sorted_kwic.py?q=a"]


@pytest.mark.unit
class TestWebApp:
    @pytest.mark.parametrize("path", RESTRICTED)
    def test_restricted_to_clients_let_in(self, client, path):
        resp = get(client, f"closed/{path}", DENIED)
        assert resp.status_code == 403

    def test_allowed_address(self, client):
        assert get(client, "closed/scripts/get_custom_landing_page.py", ALLOWED).status_code == 200

    def test_open_database(self, client):
        assert get(client, "open/scripts/get_custom_landing_page.py", DENIED).status_code == 200

    def test_claimed_address(self, client):
        headers = {"X-Forwarded-For": ALLOWED}
        assert get(client, "closed/scripts/get_custom_landing_page.py", DENIED, headers=headers).status_code == 403

    def test_behind_proxy(self, client, capsys):
        """The first address decides; when the one that connected would decide otherwise, the audit says so."""
        path = "closed/scripts/get_custom_landing_page.py"
        assert get(client, path, "", headers={"X-Forwarded-For": f"{ALLOWED}, {DENIED}"}).status_code == 200
        assert get(client, path, "", headers={"X-Forwarded-For": f"{DENIED}, {ALLOWED}"}).status_code == 403
        audit = [line for line in capsys.readouterr().err.splitlines() if line.startswith("ACCESS AUDIT")]
        assert audit == [
            f"ACCESS AUDIT: closed: allowed by the first address of X-Forwarded-For, {ALLOWED}, refused by the one "
            f"that connected, {DENIED}",
            f"ACCESS AUDIT: closed: refused by the first address of X-Forwarded-For, {DENIED}, allowed by the one "
            f"that connected, {ALLOWED}",
        ]

    @pytest.mark.parametrize("path", ["", "concordance?q=a", "scripts/get_web_config.py", "assets/index.js"])
    def test_login_screen_needs(self, client, path):
        assert get(client, f"closed/{path}", DENIED).status_code == 200

    def test_access_request(self, client):
        resp = get(client, "closed/scripts/access_request.py", DENIED)
        assert resp.status_code == 200 and json.loads(resp.text)["access"] is False
        assert "Set-Cookie" not in resp.headers

    def test_cookie_from_access_request(self, client):
        """A client let in by its address gets a cookie that lets it in from another address."""
        resp = get(client, "closed/scripts/access_request.py", ALLOWED)
        assert json.loads(resp.text)["access"] is True
        cookie = cookie_header(resp.headers["Set-Cookie"])
        path = "closed/scripts/get_custom_landing_page.py"
        assert get(client, path, DENIED, headers={"Cookie": cookie}).status_code == 200
        assert json.loads(get(client, "closed/scripts/access_request.py", DENIED, headers={"Cookie": cookie}).text)["access"]

    def test_cookie_from_page(self, client):
        resp = get(client, "closed/", ALLOWED)
        cookie = cookie_header(resp.headers["Set-Cookie"])
        assert get(client, "closed/scripts/get_custom_landing_page.py", DENIED, headers={"Cookie": cookie}).status_code == 200
        assert "Set-Cookie" not in get(client, "closed/", DENIED).headers

    def test_cookie_of_another_database(self, client, web_root):
        from philologic.runtime import WebConfig

        cookie = cookie_header(auth_cookie(WebConfig(str(web_root / "open"))))
        assert get(client, "closed/scripts/get_custom_landing_page.py", DENIED, headers={"Cookie": cookie}).status_code == 403


    def test_login_in_the_body(self, client):
        resp = client.simulate_post(
            "/philologic5/closed/scripts/access_request.py",
            json={"username": "user", "password": "pass"},
            extras={"REMOTE_ADDR": DENIED},
        )
        assert resp.status_code == 200 and json.loads(resp.text)["access"] is True
        cookie = cookie_header(resp.headers["Set-Cookie"])
        assert get(client, "closed/scripts/get_custom_landing_page.py", DENIED, headers={"Cookie": cookie}).status_code == 200

    @pytest.mark.parametrize("body", [{"username": "user", "password": "wrong"}, {"username": "user"}, {}])
    def test_failed_login_in_the_body(self, client, body):
        resp = client.simulate_post("/philologic5/closed/scripts/access_request.py", json=body, extras={"REMOTE_ADDR": DENIED})
        assert resp.status_code == 200 and json.loads(resp.text)["access"] is False
        assert "Set-Cookie" not in resp.headers

    def test_login_body_not_an_object(self, client):
        resp = client.simulate_post("/philologic5/closed/scripts/access_request.py", json=["user", "pass"],
                                    extras={"REMOTE_ADDR": DENIED})
        assert resp.status_code == 400

    def test_login_in_the_query_string(self, client):
        """As clients built before 5.2.6 send it."""
        resp = get(client, "closed/scripts/access_request.py?username=user&password=pass", DENIED)
        assert json.loads(resp.text)["access"] is True and "Set-Cookie" in resp.headers

    def test_web_config_hides_the_access_file(self, client):
        web_config = json.loads(get(client, "closed/scripts/get_web_config.py", DENIED).text)
        assert web_config["access_control"] is True and "access_file" not in web_config

    def test_no_cross_origin_reads_of_restricted_databases(self, client):
        headers = {"Origin": "https://elsewhere.example"}
        allowed = get(client, "closed/scripts/get_custom_landing_page.py", ALLOWED, headers=headers)
        assert allowed.status_code == 200 and "Access-Control-Allow-Origin" not in allowed.headers
        open_db = get(client, "open/scripts/get_custom_landing_page.py", DENIED, headers=headers)
        assert open_db.headers["Access-Control-Allow-Origin"] == "https://elsewhere.example"

    @pytest.mark.parametrize("path", ["reports/concordance.py", "scripts/get_total_results.py", "scripts/export_results.py"])
    def test_refused_search(self, client, path):
        """A search the runtime refuses is a 400 with its reason, from every kind of resource."""
        resp = get(client, f"open/{path}?q=%22la+libert%C3%A9%22+%7C+roi&report=concordance&output_format=json", DENIED)
        assert resp.status_code == 400
        assert "quoted phrase" in json.loads(resp.text)["description"]
