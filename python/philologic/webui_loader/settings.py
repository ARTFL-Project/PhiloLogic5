"""Settings of philologic5-webui-loader, from PhiloLogic's global config (/etc/philologic/philologic5.cfg): webui_loader
turns it on or off on the machine (False on a machine where nothing is loaded, such as a production machine), and
webui_loader_* keys set how it runs (see extras/webui_loader/settings.cfg). On your own machine it runs as you, with
its state (jobs, token) in ~/.local/state/philologic5/webui_loader. The global config is read, not run: its keys used
here must be literal values."""

import os
import sys
import urllib.parse
from dataclasses import dataclass, field

from philologic.webui_loader.pyconfig import ConfigFile

GLOBAL_CONFIG = os.getenv("PHILOLOGIC_CONFIG", "/etc/philologic/philologic5.cfg")
DEFAULT_STATIC_DIR = "/var/lib/philologic5/webui_loader/dist"
DEFAULT_PORT = 8765
DEFAULT_SERVICE_PORT = 8766
# Where the web server of the databases serves the service, on their host (url_root)
SERVICE_PATH = "/philologic5-webui-loader"
PREFIX = "webui_loader_"

# Keys of the global config, without their prefix: those of the service, and those also used on your own machine
PERSONAL_KEYS = ("allowed_roots", "max_cores")
SERVICE_KEYS = (
    "allowed_roots",
    "max_cores",
    "bind",
    "forwarded_allow_ips",
    "state_dir",
    "static_dir",
    "threads",
    "upload_dir",
    "upload_quota",
    "max_upload_files",
    "session_idle_timeout",
    "session_max_age",
    "max_failed_logins",
    "max_failed_logins_per_ip",
    "lockout_time",
    "trusted_browser_days",
)


class SettingsError(Exception):
    """Settings with which the UI can't run"""


class LoaderDisabled(SettingsError):
    """The UI is turned off on this machine"""


def personal_state_dir():
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local", "state")
    return os.path.join(base, "philologic5", "webui_loader")


@dataclass
class Settings:
    mode: str  # "personal" or "service"
    state_dir: str
    database_root: str
    url_root: str
    global_config: str = GLOBAL_CONFIG
    static_dir: str = DEFAULT_STATIC_DIR
    python: str = sys.executable  # runs philoload5 (python -m philologic.loadtime)
    # Directories whose files can be loaded as a service; on your own machine, where the file browser starts
    allowed_roots: list = field(default_factory=list)
    max_cores: int = None  # most cores a load can use (all of them if None)
    # Server
    bind: str = f"127.0.0.1:{DEFAULT_PORT}"
    threads: int = 8
    forwarded_allow_ips: list = field(default_factory=list)  # reverse proxies trusted for X-Forwarded-*
    public_url: str = None  # service: URL at which users reach the UI, SERVICE_PATH on the host of url_root
    token: str = None  # personal: the secret in the URL opened in the browser
    # Service
    upload_dir: str = None
    upload_quota: int = 20 * 1024**3  # bytes per user
    max_upload_files: int = 200000
    session_idle_timeout: int = 8 * 3600
    session_max_age: int = 24 * 3600
    max_failed_logins: int = 5  # per account, then locked for lockout_time
    max_failed_logins_per_ip: int = 20
    lockout_time: int = 15 * 60
    trusted_browser_days: int = 30  # a browser can be trusted for the second factor for that long (0: never)

    @property
    def service(self):
        return self.mode == "service"

    @property
    def loads_dir(self):
        return os.path.join(self.state_dir, "loads")

    @property
    def url_prefix(self):
        """Path under which the UI is served, without a trailing slash ("" at the root)"""
        if not self.public_url:
            return ""
        path = self.public_url.split("://", 1)[-1].partition("/")[2]
        return ("/" + path).rstrip("/") if path else ""

    @property
    def public_origin(self):
        if not self.public_url:
            return None
        scheme, rest = self.public_url.split("://", 1)
        return f"{scheme}://{rest.partition('/')[0]}"


def read_global_config(path=GLOBAL_CONFIG):
    """database_root, url_root, whether the UI is on (webui_loader), and its webui_loader_* settings (without the
    prefix), read without running the file: those keys must be literal values"""
    try:
        with open(path, encoding="utf8") as config_file:
            config = ConfigFile(config_file.read())
    except FileNotFoundError as error:
        raise SettingsError(f"{path} doesn't exist") from error
    except SyntaxError as error:
        raise SettingsError(f"{path} isn't valid Python: {error}") from error
    used = [
        name
        for name in config.entries
        if name in ("database_root", "url_root", "webui_loader") or name.startswith(PREFIX)
    ]
    code = [name for name in used if config.entries[name].is_code]
    if code:
        raise SettingsError(f"{path} is read, not run: write literal values for {', '.join(code)}")
    values = config.values
    loader = {name[len(PREFIX) :]: values[name] for name in used if name.startswith(PREFIX)}
    unknown = set(loader) - set(SERVICE_KEYS)
    if unknown:
        raise SettingsError(f"unknown settings in {path}: {', '.join(PREFIX + key for key in sorted(unknown))}")
    database_root, url_root = values.get("database_root"), values.get("url_root")
    if not database_root or database_root == "None":
        raise SettingsError(f"database_root is not set in {path}")
    if not url_root or url_root == "None":
        raise SettingsError(f"url_root is not set in {path}")
    enabled = values.get("webui_loader", True)
    if not isinstance(enabled, bool):
        raise SettingsError(f"webui_loader is True or False in {path}")
    return {"database_root": database_root.rstrip("/"), "url_root": url_root, "enabled": enabled, "loader": loader}


def global_settings(global_config=GLOBAL_CONFIG):
    """database_root and url_root of the global config"""
    values = read_global_config(global_config)
    return values["database_root"], values["url_root"]


def check_enabled(values, path):
    if not values["enabled"]:
        raise LoaderDisabled(f"philologic5-webui-loader is turned off on this machine (webui_loader = False in {path})")


def check_max_cores(settings, path):
    if settings.max_cores is not None and (not isinstance(settings.max_cores, int) or settings.max_cores < 1):
        raise SettingsError(f"{PREFIX}max_cores is a whole number, at least 1, or None in {path}")


def personal_settings(port=DEFAULT_PORT, global_config=GLOBAL_CONFIG, static_dir=None, state_dir=None):
    values = read_global_config(global_config)
    check_enabled(values, global_config)
    settings = Settings(
        mode="personal",
        state_dir=state_dir or personal_state_dir(),
        database_root=values["database_root"],
        url_root=values["url_root"],
        global_config=global_config,
        static_dir=static_dir or DEFAULT_STATIC_DIR,
        bind=f"127.0.0.1:{port}",
    )
    for key in PERSONAL_KEYS:
        if key in values["loader"]:
            setattr(settings, key, values["loader"][key])
    # Uploads from the computer of the browser, which may be another one (through an SSH tunnel): yours, no quota
    settings.upload_dir = os.path.join(settings.state_dir, "uploads")
    settings.upload_quota = None
    check_max_cores(settings, global_config)
    return settings


def service_settings(global_config=GLOBAL_CONFIG, check=True):
    """Settings of the service, which the web server of the databases serves at SERVICE_PATH of their host, over
    HTTPS. Raises SettingsError if they aren't safe to run with (unless check is False, to manage accounts)."""
    values = read_global_config(global_config)
    if check:
        check_enabled(values, global_config)
    url_root = urllib.parse.urlsplit(values["url_root"])
    settings = Settings(
        mode="service",
        # Not in /var/lib/philologic5, which install.sh deletes
        state_dir="/var/lib/philologic5-webui-loader",
        database_root=values["database_root"],
        url_root=values["url_root"],
        global_config=global_config,
        bind=f"127.0.0.1:{DEFAULT_SERVICE_PORT}",
        forwarded_allow_ips=["127.0.0.1"],  # the web server, on this machine
        public_url=f"{url_root.scheme.lower()}://{url_root.netloc}{SERVICE_PATH}",
    )
    for key, value in values["loader"].items():
        setattr(settings, key, value)
    if check:
        check_service_settings(settings, global_config)
    return settings


def check_service_settings(settings, path):
    key = lambda name: PREFIX + name  # noqa: E731
    if not settings.public_url.startswith("https://"):
        raise SettingsError(
            f"the UI needs HTTPS, and is served on the host of the databases: url_root must be https:// in {path}"
        )
    if settings.bind.startswith("unix:"):
        raise SettingsError(f"{key('bind')} in {path} must be a TCP address, such as 127.0.0.1:8766")
    if not settings.allowed_roots and not settings.upload_dir:
        raise SettingsError(
            f"{key('allowed_roots')} or {key('upload_dir')} must be set in {path}: the directories of the files to load"
        )
    if not settings.forwarded_allow_ips:
        raise SettingsError(f"{key('forwarded_allow_ips')} lists the addresses of the web server in {path}")
    if "*" in settings.forwarded_allow_ips:
        raise SettingsError(f"{key('forwarded_allow_ips')} can't be * in {path}: list the addresses of the web server")
    check_max_cores(settings, path)
    days = settings.trusted_browser_days
    if not isinstance(days, int) or isinstance(days, bool) or days < 0:
        raise SettingsError(f"{key('trusted_browser_days')} is a whole number of days in {path} (0: never)")
    settings.allowed_roots = [os.path.realpath(root) for root in settings.allowed_roots]
