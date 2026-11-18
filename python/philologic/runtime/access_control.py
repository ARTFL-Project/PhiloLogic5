#!/var/lib/philologic5/philologic_env/bin/python3
"""Access control, for databases whose web_config.cfg sets access_control.

A client gets in if the database's access_file allows its IP address or domain, or if it logs in with a user and
password of the database's data/logins.txt. Either way it gets a cookie signed with the database's key (see
database_key), so that its next requests need no check. The web app checks every request, but those the login screen
needs (see www/middleware.py).
"""

import hashlib
import hmac
import itertools
import os
import secrets
import socket
import sys
import time
import tomllib
from functools import cache, lru_cache

import netaddr
import regex as re
from netaddr import IPSet

from philologic.utils import load_module

# These should always be allowed for local access
local_networks = ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8"]

# How long an auth cookie lets its client in
AUTH_COOKIE_MAX_AGE = 7 * 24 * 3600

# How long a worker remembers whether the access file lets an address in, to spare each request of a client without
# a cookie (a script using the API) a reverse DNS lookup
_ALLOWED_TTL = 600
_allowed = {}

# Signs auth cookies if the installation has no secret (see _installation_secret). It is drawn when the web app is
# loaded, so Gunicorn's workers share it if the app is preloaded (preload_app, as in www/gunicorn.conf.py); it changes
# when the web app restarts, which ends every login.
_FALLBACK_SECRET = secrets.token_bytes(32)


def _global_config_path():
    return os.environ.get("PHILOLOGIC_CONFIG", "/etc/philologic/philologic5.cfg")


def _octet_box(rule):
    """The box of an octet-range rule such as "128.120.0-82" or "192.168-170.0.0": four (low, high) pairs, one for each
    octet, those not given being 0-255. A trailing dot is ignored. None if the rule has more than four parts, or a part
    that is not a number or a range of numbers between 0 and 255."""
    parts = rule.split(".")
    while parts and parts[-1] == "":
        parts.pop()
    if not parts or len(parts) > 4:
        return None
    box = []
    for part in parts:
        try:
            low, high = (int(number) for number in part.split("-", 1)) if "-" in part else (int(part), int(part))
        except ValueError:
            return None
        if not 0 <= low <= high <= 255:
            return None
        box.append((low, high))
    return box + [(0, 255)] * (4 - len(box))


def _add_box(network_set, box):
    """Add to network_set the addresses each of whose octets is in its range of box, as one IPRange for each
    combination of the octets before the last one that is not 0-255 (those after it make the range contiguous)."""
    last = max((i for i, (low, high) in enumerate(box) if (low, high) != (0, 255)), default=-1)
    if last < 0:
        network_set.add(netaddr.IPNetwork("0.0.0.0/0"))
        return
    head = [range(low, high + 1) for low, high in box[:last]]
    low, high = box[last]
    tail = 3 - last
    for prefix in itertools.product(*head):
        start = ".".join(map(str, (*prefix, low, *[0] * tail)))
        end = ".".join(map(str, (*prefix, high, *[255] * tail)))
        network_set.add(netaddr.IPRange(start, end))


@lru_cache(maxsize=32)
def _read_access_file(path, mtime):
    """What the TOML access file at path, as modified at mtime, allows: (prefixes, networks, domains, blocked_ips).
    It has an optional top-level blocked_ips list, and any number of tables, each with optional domains and ip_rules
    lists, whose rules together let a client in; other keys are ignored. A rule of ip_rules with a "/" is a CIDR
    network, one with a "-" an octet range; one that can't be read is skipped with a warning, any other a prefix of the
    client's address string. The local networks are always in networks. Raises if the file can't be read or isn't
    valid TOML."""
    with open(path, "rb") as access_file:
        config = tomllib.load(access_file)
    network_set = IPSet(netaddr.IPNetwork(network) for network in local_networks)
    prefixes, domains = [], []
    for table in config.values():
        if not isinstance(table, dict):
            continue
        domains.extend(table.get("domains", []))
        for rule in table.get("ip_rules", []):
            rule = rule.strip()
            if not rule:
                continue
            if "/" in rule:
                try:
                    network_set.add(netaddr.IPNetwork(rule))
                except netaddr.AddrFormatError:
                    print(f"Warning: skipping the ip rule '{rule}' of {path}: not a CIDR network", file=sys.stderr)
            elif "-" in rule:
                box = _octet_box(rule)
                if box is None:
                    print(f"Warning: skipping the ip rule '{rule}' of {path}: not an octet range", file=sys.stderr)
                else:
                    _add_box(network_set, box)
            else:
                prefixes.append(rule)
    return tuple(prefixes), network_set, tuple(domains), frozenset(config.get("blocked_ips", []))


@cache
def trusted_proxies():
    """The addresses of the reverse proxies whose X-Forwarded-For is to be believed: those of the global config's
    trusted_proxies, by default loopback. A connection to a unix socket, which has no REMOTE_ADDR, is always a local
    proxy's."""
    path = _global_config_path()
    proxies = getattr(load_module("philologic5", path), "trusted_proxies", None) if os.path.isfile(path) else None
    return frozenset(proxies if proxies is not None else ("127.0.0.1", "::1"))


def _forwarded_for(environ):
    return [a.strip() for a in environ.get("HTTP_X_FORWARDED_FOR", "").split(",") if a.strip()]


def client_address(environ):
    """The client's IP address, which the access file is checked against. Behind trusted proxies, the first of
    X-Forwarded-For, as in 5.2.6: when a proxy or gateway before them wrote its user's address there, that address,
    which no one verifies, rather than the one that connected (connected_address), in case access files name it."""
    address = environ.get("REMOTE_ADDR", "")
    if address and address not in trusted_proxies():
        return address
    forwarded = _forwarded_for(environ)
    return forwarded[0] if forwarded else address


def connected_address(environ):
    """The address that connected to the trusted proxies: the last of X-Forwarded-For that is not theirs, which they
    appended themselves (whatever comes before is the client's say)."""
    address = environ.get("REMOTE_ADDR", "")
    proxies = trusted_proxies()
    if address and address not in proxies:
        return address
    forwarded = _forwarded_for(environ)
    while forwarded:
        address = forwarded.pop()
        if address not in proxies:
            break
    return address


def _client_domain(incoming_address):
    """The client's domain, which the access file's domains are matched against and the login screen shows, from a
    reverse DNS lookup of its address."""
    fq_domain_name = socket.getfqdn(incoming_address).split(",")[-1]
    edit_domain = re.split(r"\.", fq_domain_name)
    if re.match("edu", edit_domain[-1]):
        return ".".join([edit_domain[-2], edit_domain[-1]])
    if len(edit_domain) == 2:
        return ".".join([edit_domain[-2], edit_domain[-1]])
    return fq_domain_name


def get_client_info(environ):
    """The client's IP address and domain."""
    incoming_address = client_address(environ)
    return incoming_address, _client_domain(incoming_address)


def is_allowed(environ, config):
    """Whether the database's access file lets the client in, by its IP address or domain. Workers remember the answer
    for each address and version of the file for _ALLOWED_TTL seconds."""
    incoming_address = client_address(environ)
    access_file = config.access_file
    if access_file and not os.path.isabs(access_file):
        access_file = os.path.join(config.db_path, "data", access_file)
    mtime = os.path.getmtime(access_file) if access_file and os.path.isfile(access_file) else None
    key = (access_file, mtime, incoming_address)
    now = time.monotonic()
    if key in _allowed and now - _allowed[key][1] < _ALLOWED_TTL:
        return _allowed[key][0]
    allowed = _check_address(incoming_address, access_file, mtime)
    connected = connected_address(environ)
    if connected != incoming_address and _check_address(connected, access_file, mtime, log=False) != allowed:
        # Whether checking the address that connected, as 5.2.7 did, would refuse anyone: the first address decides
        print(
            f"ACCESS AUDIT: {os.path.basename(os.path.normpath(config.db_path))}: "
            f"{'allowed' if allowed else 'refused'} by the first address of X-Forwarded-For, {incoming_address}, "
            f"{'refused' if allowed else 'allowed'} by the one that connected, {connected}",
            file=sys.stderr,
        )
    if len(_allowed) > 10000:
        _allowed.clear()
    _allowed[key] = (allowed, now)
    return allowed


def _check_address(incoming_address, access_file, mtime, log=True):
    """Whether access_file, modified at mtime (None if it doesn't exist), lets incoming_address in; refusals are
    logged, if log."""
    say = print if log else (lambda *args, **kwargs: None)
    if not access_file:
        say(f"UNAUTHORIZED ACCESS TO:{incoming_address}: no access file is defined", file=sys.stderr)
        return False
    if mtime is None:
        say(f"ACCESS FILE DOES NOT EXIST. UNAUTHORIZED ACCESS TO: {incoming_address}", file=sys.stderr)
        return False

    try:
        prefixes, network_set, domain_list, blocked_ips = _read_access_file(access_file, mtime)
    except Exception as e:
        say("ACCESS ERROR", repr(e), "(access files are TOML, see docs/access_control.md)", file=sys.stderr)
        say(f"UNAUTHORIZED ACCESS TO:{incoming_address}: can't load access config", file=sys.stderr)
        return False

    if incoming_address in blocked_ips:
        say(f"BLOCKED IP ACCESS ATTEMPT: {incoming_address}", file=sys.stderr)
        return False

    if incoming_address.startswith(prefixes):
        return True
    try:
        if netaddr.IPAddress(incoming_address) in network_set:
            return True
    except (ValueError, netaddr.AddrFormatError):
        pass  # not an IP address

    # Check domain access, last: it takes a reverse DNS lookup. By substring, and with no forward lookup to confirm
    # the name.
    match_domain = _client_domain(incoming_address) if domain_list else incoming_address
    if match_domain in domain_list or any(domain in match_domain for domain in domain_list):
        return True

    # If no match found, access denied
    say(
        f"UNAUTHORIZED ACCESS TO:{incoming_address} from domain {match_domain}: IP not in whitelist",
        file=sys.stderr,
    )
    return False


def check_access(environ, config):
    """An auth cookie (a Set-Cookie header value) for the client, if the database's access file lets it in by its IP
    address or domain, else ""."""
    return auth_cookie(config) if is_allowed(environ, config) else ""


def login_access(environ, request, config, headers, username=None, password=None):
    """Whether the client may use the database: by its cookie, by the username and password it sends (by default those
    of the query string, where clients built before 5.2.6 put them), or else by its IP address or domain; and headers,
    with an auth cookie added to them if it gets in now."""
    if request.authenticated:
        return True, headers
    if username is None:
        username, password = request.username, request.password
    if username and password:
        access = check_login_info(config, username, password)
    else:
        access = is_allowed(environ, config)
    if access:
        headers.append(("Set-Cookie", auth_cookie(config)))
    return access, headers


def check_login_info(config, username, password):
    """Whether username and password are a login of the database's data/logins.txt: one per line, tab-separated."""
    login_file_path = os.path.join(config.db_path, "data/logins.txt")
    if not os.path.exists(login_file_path):
        return False
    with open(login_file_path, "rb") as password_file:
        for line in password_file:
            fields = line.decode("utf8", "ignore").strip().split("\t")
            if len(fields) < 2:  # empty line, or no password
                continue
            user, passwd = fields[0].encode("utf8"), fields[1].encode("utf8")
            if hmac.compare_digest(user, username.encode("utf8")) & hmac.compare_digest(passwd, password.encode("utf8")):
                return True
    return False


# ── Auth cookies ──────────────────────────────────────────────────────────────


@cache
def _installation_secret():
    """The installation's secret: the content of philologic5.secret, next to the global config, which install.sh
    creates. Without it, _FALLBACK_SECRET."""
    path = os.path.splitext(_global_config_path())[0] + ".secret"
    try:
        with open(path, encoding="utf8") as secret_file:
            secret = secret_file.read().strip()
    except OSError:
        secret = ""
    if secret:
        return secret.encode("utf8")
    print(
        f"No secret in {path}: auth cookies are signed with one drawn at startup, and logins end when the web app"
        " restarts",
        file=sys.stderr,
    )
    return _FALLBACK_SECRET


def _database_name(config):
    return os.path.basename(os.path.normpath(config.db_path))


def database_key(config):
    """The key that signs the auth cookies of config's database: the secret of its db.locals.py, if it has one, else
    a key of its own derived from the installation's secret."""
    if config.db_locals.secret:
        return config.db_locals.secret.encode("utf8")
    return hmac.new(_installation_secret(), _database_name(config).encode("utf8"), hashlib.sha256).digest()


def _cookie_name(config):
    """Each database has its cookie: they all have Path=/, as the URL path of a database depends on the proxy."""
    return "philologic5_" + re.sub(r"[^A-Za-z0-9_.-]", "_", _database_name(config))


def _signature(config, timestamp):
    message = f"{_database_name(config)}\0{timestamp}".encode("utf8")
    return hmac.new(database_key(config), message, hashlib.sha256).hexdigest()


def auth_cookie(config):
    """A Set-Cookie header value letting its client into config's database for AUTH_COOKIE_MAX_AGE seconds."""
    timestamp = int(time.time())
    return (
        f"{_cookie_name(config)}={timestamp}.{_signature(config, timestamp)}; Path=/; Max-Age={AUTH_COOKIE_MAX_AGE};"
        " HttpOnly; SameSite=Lax"
    )


def is_authenticated(environ, config):
    """Whether the request has an auth cookie for config's database that its key signed less than AUTH_COOKIE_MAX_AGE
    seconds ago. Cookies are parsed by hand: http.cookies gives up on the whole header for one cookie it can't parse,
    and other sites of the same host may set any."""
    name = _cookie_name(config)
    for cookie in environ.get("HTTP_COOKIE", "").split(";"):
        key, _, value = cookie.strip().partition("=")
        if key != name:
            continue
        timestamp, _, signature = value.partition(".")
        if not (timestamp.isascii() and timestamp.isdigit()):
            continue
        if not -60 <= time.time() - int(timestamp) <= AUTH_COOKIE_MAX_AGE:  # a minute's leeway for clock skew
            continue
        expected = _signature(config, int(timestamp))
        if hmac.compare_digest(signature.encode("utf8", "replace"), expected.encode("utf8")):
            return True
    return False
