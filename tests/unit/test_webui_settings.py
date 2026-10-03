"""Unit tests for the settings of philologic5-webui-loader, in PhiloLogic's global config: webui_loader turns it off on a
machine, the service is served on the host of the databases, and refuses to run without HTTPS or with settings which
aren't literal values."""

import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.webui_loader.settings import (
    LoaderDisabled,
    SettingsError,
    personal_settings,
    service_settings,
    web_app_url,
)

pytestmark = pytest.mark.unit

GLOBAL = 'database_root = "/var/www/html/philologic5"\n'


@pytest.fixture
def global_config(tmp_path):
    def write(text):
        path = tmp_path / "philologic5.cfg"
        path.write_text(GLOBAL + text, encoding="utf8")
        return str(path)

    return write


def service(global_config, text):
    return service_settings(global_config('webui_loader_allowed_roots = ["/data"]\n' + text))


def test_service_on_the_host_of_the_databases(global_config):
    settings = service(global_config, "")
    assert settings.url_prefix == "/philologic5-webui-loader"
    # The databases, on the same host, under the prefix of PhiloLogic's install
    assert settings.databases_url == "/philologic5/" and settings.database_url("frantext") == "/philologic5/frantext/"
    assert settings.bind == "127.0.0.1:8766" and settings.forwarded_allow_ips == ["127.0.0.1"]
    assert settings.allowed_roots == ["/data"]


def test_no_url_setting(global_config):
    """No URL to set: the service takes its own from the requests, and a url_root of earlier versions is ignored"""
    settings = service(global_config, 'url_root = "http://philologic.example.edu/philologic5/"\n')
    assert settings.databases_url == "/philologic5/"
    # On your own machine, whose UI is on a port of localhost, no link to databases a web server serves
    personal = personal_settings(global_config=global_config(""), web_app_config="/nonexistent/gunicorn.conf.py")
    assert personal.url_prefix == "" and personal.databases_url is None and personal.database_url("frantext") is None


@pytest.mark.parametrize(
    "bind, url",
    [
        ('"127.0.0.1:8080"', "http://127.0.0.1:8080/"),  # as on a Mac: gunicorn serves the databases itself
        ('"0.0.0.0:8000"', "http://localhost:8000/"),
        ('["unix:/run/philologic/gunicorn.sock", "localhost:9000"]', "http://localhost:9000/"),
        ('"unix:/var/run/philologic/gunicorn.sock"', None),  # behind a web server, under its prefix
    ],
)
def test_databases_where_gunicorn_serves_them(global_config, tmp_path, bind, url):
    config = tmp_path / "gunicorn.conf.py"
    config.write_text(f"import multiprocessing\nbind = {bind}\nworkers = multiprocessing.cpu_count()\n", encoding="utf8")
    assert web_app_url(str(config)) == url
    personal = personal_settings(global_config=global_config(""), web_app_config=str(config))
    assert personal.database_url("frantext") == (url and url + "frantext/")


@pytest.mark.parametrize(
    "text, message",
    [
        ("webui_loader_forwarded_allow_ips = []\n", "web server"),
        ('webui_loader_forwarded_allow_ips = ["*"]\n', r"\*"),
        ('webui_loader_bind = "unix:/run/x.sock"\n', "TCP"),
        ("webui_loader_upload_quota = 2 * 1024\n", "literal"),
        ("webui_loader_no_such_setting = 1\n", "unknown"),
        # The service no longer has a URL or a certificate of its own
        ('webui_loader_public_url = "https://philologic.example.edu:8766"\n', "unknown"),
        ('webui_loader_certfile = "/etc/ssl/cert.pem"\nwebui_loader_keyfile = "/etc/ssl/key.pem"\n', "unknown"),
        ("webui_loader_max_cores = 0\n", "max_cores"),
    ],
)
def test_refused(global_config, text, message):
    with pytest.raises(SettingsError, match=message):
        service(global_config, text)


def test_turned_off(global_config):
    path = global_config("webui_loader = False\n")
    with pytest.raises(LoaderDisabled):
        personal_settings(global_config=path)
    with pytest.raises(LoaderDisabled):
        service_settings(path)
    # Accounts can still be managed
    assert service_settings(path, check=False).url_prefix == "/philologic5-webui-loader"


def test_personal_settings(global_config):
    settings = personal_settings(
        global_config=global_config('webui_loader_allowed_roots = ["/data"]\nwebui_loader_max_cores = 8\n')
    )
    assert settings.max_cores == 8 and settings.allowed_roots == ["/data"] and not settings.service
    # On by default
    assert personal_settings(global_config=global_config("")).max_cores is None


def test_other_code_of_the_global_config_is_ignored(global_config):
    settings = personal_settings(
        global_config=global_config("import os\nhitlist_dir = os.path.join('/var', 'cache')\n")
    )
    assert settings.database_root == "/var/www/html/philologic5"
