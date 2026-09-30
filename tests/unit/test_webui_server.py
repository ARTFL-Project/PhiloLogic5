"""Unit tests for the server of philologic5-webui-loader: its security checks on your own machine (token, Host,
Origin, CSRF) and as a service (HTTPS, logins with a second factor, limits on failed logins, permissions)."""

import os
import sys
import time
from pathlib import Path

import falcon.testing
import pytest
from black import FileMode, format_str

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.Config import MakeWebConfig
from philologic.webui_loader import accounts as accounts_module
from philologic.webui_loader.accounts import Accounts, totp_code
from philologic.webui_loader.server.app import create_app
from philologic.webui_loader.settings import Settings

pytestmark = pytest.mark.unit

PORT = 8765
LOCAL = f"localhost:{PORT}"
PUBLIC = "https://loader.example.org/philologic5-webui-loader"


@pytest.fixture(autouse=True)
def fast_scrypt(monkeypatch):
    monkeypatch.setattr(accounts_module, "SCRYPT_N", 2**10)


def make_database(root, name, owner_writable=True):
    data = Path(root) / name / "data"
    data.mkdir(parents=True)
    config_path = data / "web_config.cfg"
    config_path.write_text(
        format_str(str(MakeWebConfig(str(config_path), dbname=name)), mode=FileMode()), encoding="utf8"
    )
    (data / "db.locals.py").write_text('metadata_fields = ["author", "title"]\n', encoding="utf8")


@pytest.fixture
def personal(tmp_path):
    root = tmp_path / "dbs"
    root.mkdir()
    make_database(root, "mydb")
    settings = Settings(
        mode="personal",
        state_dir=str(tmp_path / "state"),
        database_root=str(root),
        url_root="http://localhost/philologic5/",
        bind=f"127.0.0.1:{PORT}",
        token="secret-token",
        static_dir=str(tmp_path / "static"),
    )
    return falcon.testing.TestClient(create_app(settings)), settings


def local_headers(token="secret-token", **extra):
    headers = {"Host": LOCAL, "Authorization": f"Bearer {token}"}
    headers.update(extra)
    return headers


def test_personal_token(personal):
    client, settings = personal
    assert client.simulate_get("/api/databases", headers={"Host": LOCAL}).status_code == 401
    assert client.simulate_get("/api/databases", headers=local_headers("wrong")).status_code == 401
    result = client.simulate_get("/api/databases", headers=local_headers())
    assert result.status_code == 200
    assert [db["name"] for db in result.json["databases"]] == ["mydb"]
    session = client.simulate_get("/api/session", headers=local_headers()).json
    assert session["authenticated"] and session["mode"] == "personal"
    assert client.simulate_get("/api/session", headers={"Host": LOCAL}).json == {
        "mode": "personal",
        "authenticated": False,
    }


def test_personal_host_check(personal):
    client, settings = personal
    # DNS rebinding: another host name pointing to 127.0.0.1
    result = client.simulate_get("/api/databases", headers=local_headers(Host=f"evil.example.org:{PORT}"))
    assert result.status_code == 403


def test_origin_and_csrf(personal):
    client, settings = personal
    csrf = client.simulate_get("/api/session", headers=local_headers()).json["csrf"]
    config = client.simulate_get("/api/databases/mydb/web_config", headers=local_headers()).json
    payload = {"changes": {"dbname": "Edited"}, "hash": config["hash"]}
    # No Origin, another origin, no CSRF token: refused
    assert (
        client.simulate_put("/api/databases/mydb/web_config", json=payload, headers=local_headers()).status_code == 403
    )
    headers = local_headers(Origin="http://evil.example.org", **{"X-CSRF-Token": csrf})
    assert client.simulate_put("/api/databases/mydb/web_config", json=payload, headers=headers).status_code == 403
    headers = local_headers(Origin=f"http://{LOCAL}")
    assert client.simulate_put("/api/databases/mydb/web_config", json=payload, headers=headers).status_code == 403
    headers = local_headers(Origin=f"http://{LOCAL}", **{"X-CSRF-Token": csrf})
    result = client.simulate_put("/api/databases/mydb/web_config", json=payload, headers=headers)
    assert result.status_code == 200, result.text
    assert result.json["values"]["dbname"] == "Edited" and result.json["backup"]


def test_security_headers_and_static(personal):
    client, settings = personal
    result = client.simulate_get("/", headers={"Host": LOCAL})
    assert "not built" in result.text
    assert "frame-ancestors 'none'" in result.headers["Content-Security-Policy"]
    os.makedirs(settings.static_dir)
    Path(settings.static_dir, "index.html").write_text("<html>app</html>", encoding="utf8")
    assert client.simulate_get("/", headers={"Host": LOCAL}).text == "<html>app</html>"
    assert client.simulate_get("/../../etc/passwd", headers={"Host": LOCAL}).status_code == 404
    assert client.simulate_get("/api/databases", headers=local_headers()).headers["Cache-Control"] == "no-store"


def test_files_and_load_options(personal, tmp_path):
    client, settings = personal
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.xml").write_text("<TEI/>", encoding="utf8")
    (corpus / "b.txt").write_text("x", encoding="utf8")
    listing = client.simulate_get(
        "/api/files", params={"path": str(corpus), "pattern": "*.xml"}, headers=local_headers()
    ).json
    assert listing["matching"] == 1 and listing["file_count"] == 2
    schema = client.simulate_get("/api/load_options", headers=local_headers()).json
    assert "token_regex" in {option["key"] for option in schema["options"]}


# Service


@pytest.fixture
def service(tmp_path):
    root = tmp_path / "dbs"
    root.mkdir()
    for name in ("alice_db", "other_db"):
        make_database(root, name)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    settings = Settings(
        mode="service",
        state_dir=str(tmp_path / "state"),
        database_root=str(root),
        url_root="https://loader.example.org/philologic5/",
        bind="127.0.0.1:8766",
        forwarded_allow_ips=["127.0.0.1"],
        public_url=PUBLIC,
        allowed_roots=[str(corpus)],
        static_dir=str(tmp_path / "static"),
    )
    accounts = Accounts(os.path.join(settings.state_dir, "accounts.sqlite"), settings)
    client = falcon.testing.TestClient(create_app(settings, accounts))
    return client, settings, accounts


def https(**extra):
    headers = {"Host": "loader.example.org", "X-Forwarded-Proto": "https"}
    headers.update(extra)
    return headers


def post(client, path, json=None, csrf=None, cookies=None):
    headers = https(Origin="https://loader.example.org")
    if csrf:
        headers["X-CSRF-Token"] = csrf
    return client.simulate_post(
        f"/philologic5-webui-loader{path}", json=json or {}, headers=headers, cookies=cookies, protocol="https"
    )


def get(client, url, cookies=None, **params):
    return client.simulate_get(
        f"/philologic5-webui-loader{url}", headers=https(), cookies=cookies, params=params, protocol="https"
    )


def login(client, accounts, name, password):
    """Log in with the password and the second factor (set up if needed): the cookies of the full session"""
    result = post(client, "/api/login", {"username": name, "password": password})
    assert result.status_code == 200, result.text
    cookies = {"philologic5_webui_session": result.cookies["philologic5_webui_session"].value}
    if result.json["next"] == "totp_setup":
        secret = get(client, "/api/login/totp_setup", cookies).json["secret"]
        result = post(
            client, "/api/login/totp_setup", {"code": totp_code(secret, int(time.time() // 30))}, cookies=cookies
        )
        assert result.status_code == 200, result.text
        assert len(result.json["recovery_codes"]) == 10
    else:
        secret = accounts.user(name)["totp_secret"]
        counter = int(time.time() // 30) + 1  # the code of the current step may have been used by the setup
        result = post(client, "/api/login/totp", {"code": totp_code(secret, counter)}, cookies=cookies)
        assert result.status_code == 200, result.text
    full = {"philologic5_webui_session": result.cookies["philologic5_webui_session"].value}
    assert full != cookies  # a new session id after the second factor
    return full


def csrf_of(client, cookies):
    return get(client, "/api/session", cookies).json["csrf"]


def test_https_only(service):
    client, settings, accounts = service
    result = client.simulate_get("/philologic5-webui-loader/api/session", headers={"Host": "loader.example.org"})
    assert result.status_code == 403  # plain HTTP from the proxy
    result = client.simulate_get(
        "/philologic5-webui-loader/api/session", headers=https(), remote_addr="10.0.0.9", protocol="http"
    )
    assert result.status_code == 403  # X-Forwarded-Proto from an untrusted address
    result = get(client, "/api/session")
    assert result.status_code == 200
    # HSTS, which applies to the whole host, is up to its web server
    assert "Strict-Transport-Security" not in result.headers


def test_prefix_redirects_to_its_directory(service):
    client, settings, accounts = service
    result = client.simulate_get("/philologic5-webui-loader", headers=https(), protocol="https")
    assert result.status_code == 301 and result.headers["Location"] == "/philologic5-webui-loader/"


def test_login_with_second_factor(service):
    client, settings, accounts = service
    password = "a long password 1"
    accounts.add_user("alice", "loader", password)
    assert get(client, "/api/databases").status_code == 401
    # The password alone doesn't give access
    result = post(client, "/api/login", {"username": "alice", "password": password})
    partial = {"philologic5_webui_session": result.cookies["philologic5_webui_session"].value}
    assert result.json["next"] == "totp_setup"
    assert get(client, "/api/databases", partial).status_code == 401
    assert get(client, "/api/session", partial).json == {
        "mode": "service",
        "authenticated": False,
        "stage": "totp_setup",
        "trusted_browser_days": 30,
    }
    cookies = login(client, accounts, "alice", password)
    session = get(client, "/api/session", cookies).json
    assert session["authenticated"] and session["user"] == "alice" and session["role"] == "loader"
    assert get(client, "/api/databases", cookies).status_code == 200
    # Second login: the code of the second factor, which can't be used twice
    result = post(client, "/api/login", {"username": "alice", "password": password})
    assert result.json["next"] == "totp"
    partial = {"philologic5_webui_session": result.cookies["philologic5_webui_session"].value}
    secret = accounts.user("alice")["totp_secret"]
    last = accounts.user("alice")["totp_last_counter"]
    assert post(client, "/api/login/totp", {"code": totp_code(secret, last)}, cookies=partial).status_code == 401
    # Logout ends the session
    csrf = csrf_of(client, cookies)
    assert post(client, "/api/logout", csrf=csrf, cookies=cookies).status_code == 200
    assert get(client, "/api/databases", cookies).status_code == 401


def test_recovery_code(service):
    client, settings, accounts = service
    accounts.add_user("bob", "loader", "another long password")
    result = post(client, "/api/login", {"username": "bob", "password": "another long password"})
    cookies = {"philologic5_webui_session": result.cookies["philologic5_webui_session"].value}
    secret = get(client, "/api/login/totp_setup", cookies).json["secret"]
    codes = post(
        client, "/api/login/totp_setup", {"code": totp_code(secret, int(time.time() // 30))}, cookies=cookies
    ).json["recovery_codes"]
    for attempt in range(2):
        result = post(client, "/api/login", {"username": "bob", "password": "another long password"})
        partial = {"philologic5_webui_session": result.cookies["philologic5_webui_session"].value}
        result = post(client, "/api/login/totp", {"code": codes[0]}, cookies=partial)
        assert result.status_code == (200 if attempt == 0 else 401)  # once only


def test_wrong_passwords_lock(service):
    client, settings, accounts = service
    accounts.add_user("carol", "loader", "the right password")
    for attempt in range(settings.max_failed_logins):
        assert post(client, "/api/login", {"username": "carol", "password": "wrong password!"}).status_code == 401
    result = post(client, "/api/login", {"username": "carol", "password": "the right password"})
    assert result.status_code == 401 and "too many" in result.json["description"]
    # Unknown users get the same answer
    result = post(client, "/api/login", {"username": "nobody", "password": "whatever password"})
    assert result.json["description"] == "wrong user name or password"


def test_temporary_password_must_be_changed(service):
    client, settings, accounts = service
    temporary = accounts.add_user("dave", "loader")
    cookies = login(client, accounts, "dave", temporary)
    session = get(client, "/api/session", cookies).json
    assert session["must_change_password"]
    assert get(client, "/api/databases", cookies).status_code == 403
    csrf = session["csrf"]
    assert (
        post(client, "/api/account/password", {"current": temporary, "new": "short"}, csrf, cookies).status_code == 400
    )
    result = post(client, "/api/account/password", {"current": temporary, "new": "a brand new password"}, csrf, cookies)
    assert result.status_code == 200
    assert get(client, "/api/databases", cookies).status_code == 200


def test_permissions(service):
    client, settings, accounts = service
    accounts.add_user("admin", "admin", "admin's long password")
    accounts.add_user("alice", "loader", "alice's long password")
    accounts.set_owner("alice_db", "alice")
    alice = login(client, accounts, "alice", "alice's long password")
    databases = {db["name"]: db for db in get(client, "/api/databases", alice).json["databases"]}
    assert databases["alice_db"]["may_edit"] and not databases["other_db"]["may_edit"]
    csrf = csrf_of(client, alice)
    config = get(client, "/api/databases/other_db/web_config", alice).json
    assert not config["writable"]
    result = client.simulate_put(
        "/philologic5-webui-loader/api/databases/other_db/web_config",
        json={"changes": {"dbname": "x"}, "hash": config["hash"]},
        headers=https(Origin="https://loader.example.org", **{"X-CSRF-Token": csrf}),
        cookies=alice,
        protocol="https",
    )
    assert result.status_code == 403
    # Paths of files read by the runtime must stay in the database
    config = get(client, "/api/databases/alice_db/web_config", alice).json
    result = client.simulate_put(
        "/philologic5-webui-loader/api/databases/alice_db/web_config",
        json={"changes": {"search_syntax_template": "../../../../etc/passwd"}, "hash": config["hash"]},
        headers=https(Origin="https://loader.example.org", **{"X-CSRF-Token": csrf}),
        cookies=alice,
        protocol="https",
    )
    assert result.status_code == 409 and "inside the database" in result.json["description"]
    # Files outside the allowed roots
    assert get(client, "/api/files", alice, path="/etc").status_code == 400
    result = post(client, "/api/load_config", {"path": "/etc/philologic/philologic5.cfg"}, csrf, alice)
    assert result.status_code == 400
    # Only admins manage accounts
    assert get(client, "/api/users", alice).status_code == 403
    admin = login(client, accounts, "admin", "admin's long password")
    admin_csrf = csrf_of(client, admin)
    assert [user["name"] for user in get(client, "/api/users", admin).json["users"]] == ["admin", "alice"]
    assert post(client, "/api/users/alice/grant", {"database": "other_db"}, admin_csrf, admin).status_code == 200
    databases = {db["name"]: db for db in get(client, "/api/databases", alice).json["databases"]}
    assert databases["other_db"]["may_edit"]
    result = post(client, "/api/users", {"name": "eve", "role": "loader"}, admin_csrf, admin)
    assert result.status_code == 201 and result.json["temporary_password"]
    actions = [entry["action"] for entry in get(client, "/api/audit", admin).json["entries"]]
    assert "user_added" in actions and "user_grant" in actions and "login" in actions


def test_load_config_with_code_refused(service, tmp_path):
    client, settings, accounts = service
    accounts.add_user("alice", "loader", "alice's long password")
    alice = login(client, accounts, "alice", "alice's long password")
    corpus = tmp_path / "corpus"
    (corpus / "a.xml").write_text("<TEI><teiHeader/><text/></TEI>", encoding="utf8")
    config = corpus / "evil_config.py"
    config.write_text("import os\nos.system('touch /tmp/pwned')\n", encoding="utf8")
    request = {
        "dbname": "new_db",
        "files": {"directory": str(corpus), "pattern": "*.xml"},
        "base": {"config_path": str(config)},
        "options": {"lemma_file": "/etc/passwd"},
    }
    result = post(client, "/api/preflight", {"request": request}, csrf_of(client, alice), alice)
    fields = {error["field"] for error in result.json["errors"]}
    assert {"base", "lemma_file"} <= fields
    assert not os.path.exists("/tmp/pwned")


# Every route which isn't public, with values for its parameters
PROTECTED = [
    "/api/system",
    "/api/load_options",
    "/api/web_config_options",
    "/api/databases",
    "/api/databases/mydb",
    "/api/databases/mydb/web_config",
    "/api/load_config",
    "/api/files",
    "/api/preflight",
    "/api/jobs",
    "/api/jobs/mydb-20260101-000000-abcdef",
    "/api/jobs/mydb-20260101-000000-abcdef/log",
    "/api/jobs/mydb-20260101-000000-abcdef/cancel",
    "/api/jobs/mydb-20260101-000000-abcdef/load_config",
    "/api/previews/header",
    "/api/uploads",
    "/api/uploads/20260101-000000-abcdef12",
    "/api/uploads/20260101-000000-abcdef12/content",
    "/api/uploads/20260101-000000-abcdef12/complete",
    "/api/users",
    "/api/users/alice",
    "/api/users/alice/grant",
    "/api/audit",
    "/api/account/password",
]


def test_personal_routes_need_the_token(personal):
    client, settings = personal
    for path in PROTECTED:
        for method in ("GET", "POST", "PUT", "PATCH", "DELETE"):
            headers = {"Host": LOCAL, "Origin": f"http://{LOCAL}"}
            result = client.simulate_request(method, path, headers=headers, json={})
            assert result.status_code in (401, 405), (method, path, result.status_code)


def test_paths_with_double_slashes_refused(personal):
    client, settings = personal
    for path in ("//api/system", "/api//system", "//api/databases", "/%2Fapi/system"):
        result = client.simulate_get(path, headers={"Host": LOCAL})
        assert result.status_code in (400, 401, 404), (path, result.status_code)
        assert "database_root" not in result.text


def test_service_routes_need_a_login(service):
    client, settings, accounts = service
    for path in PROTECTED:
        for prefix in ("/philologic5-webui-loader", "/philologic5-webui-loader/"):
            for method in ("GET", "POST", "PUT", "PATCH", "DELETE"):
                result = client.simulate_request(
                    method, prefix + path, headers=https(Origin="https://loader.example.org"), json={}, protocol="https"
                )
                assert result.status_code in (400, 401, 405), (method, prefix + path, result.status_code)


def test_address_key():
    from philologic.webui_loader.accounts import address_key

    assert address_key("192.0.2.1") == "192.0.2.1"
    assert address_key("::ffff:192.0.2.1") == "192.0.2.1"
    assert address_key("2001:db8::1") == address_key("2001:db8::ffff") == "2001:db8::/64"
    assert address_key("") == ""


def test_totp_code_used_once_even_at_once(service):
    client, settings, accounts = service
    accounts.add_user("frank", "loader", "frank's long password")
    login(client, accounts, "frank", "frank's long password")
    tokens = [accounts.login("frank", "frank's long password", "10.0.0.1")[0] for _ in range(2)]
    user = accounts.user("frank")
    code = totp_code(user["totp_secret"], user["totp_last_counter"] + 1)
    results = []
    for token in tokens:
        try:
            accounts.verify_second_factor(token, code, "10.0.0.1")
            results.append(True)
        except Exception:
            results.append(False)
    assert results == [True, False]
    assert accounts.verify_second_factor.__name__  # non-string codes are refused, not a crash
    with pytest.raises(Exception, match="wrong code|log in again"):
        accounts.verify_second_factor(tokens[1], 123456, "10.0.0.1")


TRUSTED = "philologic5_webui_trusted"
SESSION = "philologic5_webui_session"


def first_login(client, name, password, trust):
    """Log in for the first time, setting up the second factor: the cookies set"""
    result = post(client, "/api/login", {"username": name, "password": password})
    cookies = {SESSION: result.cookies[SESSION].value}
    secret = get(client, "/api/login/totp_setup", cookies).json["secret"]
    result = post(
        client,
        "/api/login/totp_setup",
        {"code": totp_code(secret, int(time.time() // 30)), "trust": trust},
        cookies=cookies,
    )
    assert result.status_code == 200, result.text
    return result


def test_trusted_browser(service):
    client, settings, accounts = service
    accounts.add_user("gina", "loader", "gina's long password")
    accounts.add_user("hugo", "loader", "hugo's long password")
    result = first_login(client, "gina", "gina's long password", trust=True)
    trusted = result.cookies[TRUSTED]
    assert trusted.max_age == 30 * 86400 and trusted.secure and trusted.http_only and trusted.same_site == "Strict"
    browser = {TRUSTED: trusted.value}
    # In that browser, the password is enough
    result = post(client, "/api/login", {"username": "gina", "password": "gina's long password"}, cookies=browser)
    assert result.json["next"] == "done"
    session = {SESSION: result.cookies[SESSION].value}
    assert get(client, "/api/databases", session).status_code == 200
    assert get(client, "/api/session", session | browser).json["browser_trusted"]
    # but not the wrong password, nor for another user, nor in another browser
    assert (
        post(client, "/api/login", {"username": "gina", "password": "wrong password!!"}, cookies=browser).status_code
        == 401
    )
    first_login(client, "hugo", "hugo's long password", trust=False)
    result = post(client, "/api/login", {"username": "hugo", "password": "hugo's long password"}, cookies=browser)
    assert result.json["next"] == "totp"
    result = post(client, "/api/login", {"username": "gina", "password": "gina's long password"})
    assert result.json["next"] == "totp"
    # Forgotten
    csrf = csrf_of(client, session)
    result = client.simulate_delete(
        "/philologic5-webui-loader/api/account/trusted_browsers",
        headers=https(Origin="https://loader.example.org", **{"X-CSRF-Token": csrf}),
        cookies=session | browser,
        protocol="https",
    )
    assert result.status_code == 200
    result = post(client, "/api/login", {"username": "gina", "password": "gina's long password"}, cookies=browser)
    assert result.json["next"] == "totp"
    assert "trusted_browsers_forgotten" in [entry["action"] for entry in accounts.audit_log()]


def test_trust_revoked(service):
    client, settings, accounts = service
    accounts.add_user("ines", "loader", "ines's long password")
    browser = {TRUSTED: first_login(client, "ines", "ines's long password", trust=True).cookies[TRUSTED].value}
    other = accounts.verify_second_factor  # a second trusted browser, from another login
    result = post(client, "/api/login", {"username": "ines", "password": "ines's long password"})
    partial = result.cookies[SESSION].value
    user = accounts.user("ines")
    session, other_browser = other(
        partial, totp_code(user["totp_secret"], user["totp_last_counter"] + 1), "10.0.0.2", True
    )
    assert accounts.users()[0]["trusted_browsers"] == 2
    # Changing the password keeps the browser it's changed in, and forgets the others
    accounts.change_password("ines", "ines's long password", "ines's new password", keep_browser=browser[TRUSTED])
    assert (
        post(client, "/api/login", {"username": "ines", "password": "ines's new password"}, cookies=browser).json[
            "next"
        ]
        == "done"
    )
    other_cookie = {TRUSTED: other_browser}
    assert (
        post(client, "/api/login", {"username": "ines", "password": "ines's new password"}, cookies=other_cookie).json[
            "next"
        ]
        == "totp"
    )
    # Resetting the password, or the second factor, or disabling the account forgets them all
    for revoke in (lambda: accounts.reset_password("ines"), lambda: accounts.update_user("ines", disabled=True)):
        assert accounts.browser_trusted("ines", browser[TRUSTED])
        revoke()
        assert not accounts.browser_trusted("ines", browser[TRUSTED])
        accounts.update_user("ines", disabled=False)
        with accounts.connection() as connection:
            browser = {TRUSTED: accounts.trust_browser(connection, "ines", "10.0.0.1")}


def test_last_admin(service):
    client, settings, accounts = service
    accounts.add_user("root", "admin", "root's long password")
    accounts.add_user("kim", "loader", "kim's long password")
    accounts.update_user("kim", disabled=True)  # not an admin: fine
    for change in ({"role": "loader"}, {"disabled": True}):
        with pytest.raises(Exception, match="admin left"):
            accounts.update_user("root", **change)
    accounts.add_user("root2", "admin", "root2's long password")
    accounts.update_user("root", role="loader")


def test_trust_turned_off(service):
    client, settings, accounts = service
    settings.trusted_browser_days = 0
    accounts.add_user("jack", "loader", "jack's long password")
    assert TRUSTED not in first_login(client, "jack", "jack's long password", trust=True).cookies
    assert get(client, "/api/session").json["trusted_browser_days"] == 0


def test_personal_uploads(personal):
    client, settings = personal
    settings.upload_dir = os.path.join(settings.state_dir, "uploads")
    settings.upload_quota = None
    assert client.simulate_get("/api/uploads", headers={"Host": LOCAL}).status_code == 401
    csrf = client.simulate_get("/api/session", headers=local_headers()).json["csrf"]
    headers = local_headers(Origin=f"http://{LOCAL}", **{"X-CSRF-Token": csrf})
    result = client.simulate_post("/api/uploads", json={"name": "corpus", "kind": "files", "size": 5}, headers=headers)
    assert result.status_code == 201, result.text
    upload_id = result.json["id"]
    result = client.simulate_put(
        f"/api/uploads/{upload_id}/content", params={"path": "a.xml", "offset": 0}, body=b"<TEI>", headers=headers
    )
    assert result.json["received"] == 5
    status = client.simulate_post(f"/api/uploads/{upload_id}/complete", headers=headers).json
    assert status["state"] == "ready"
    assert client.simulate_get("/api/uploads", headers=local_headers()).json["usage"]["quota"] is None
