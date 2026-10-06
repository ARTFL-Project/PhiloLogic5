#!/var/lib/philologic5/philologic_env/bin/python3
"""Access control, for databases whose web_config.cfg sets access_control.

A client gets in if the database's access_file allows its IP address or domain, or if it logs in with a user and
password of the database's data/logins.txt. Either way it gets a cookie signed with the database's key (see
database_key), so that its next requests need no check. The web app checks every request, but those the login screen
needs (see www/middleware.py).
"""

import hashlib
import hmac
import os
import pickle
import secrets
import socket
import sys
import time
from functools import cache

import netaddr
import regex as re
from netaddr import IPSet

from philologic.utils import load_module

# These should always be allowed for local access
local_networks = ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8"]
ip_ranges = [re.compile(rf"^{i.split('/')[0]}.*") for i in local_networks]  # For backward compatibility

# Cached IP whitelist info
_COMPILED_IP_CACHE_DIR = "/var/lib/philologic5/ip_cache"

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


def load_or_compile_ip_whitelist(access_file):
    """Load compiled IP whitelist or recompile if outdated"""
    safe_name = access_file.replace("/", "_").lstrip("_") + ".compiled_ips"
    compiled_file = os.path.join(_COMPILED_IP_CACHE_DIR, safe_name)
    access_mtime = os.path.getmtime(access_file)

    # Check if compiled file exists and is newer than access file
    if os.path.exists(compiled_file) and os.path.getmtime(compiled_file) >= access_mtime:
        try:
            with open(compiled_file, "rb") as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError):
            # If loading fails, recompile
            pass

    # Compile IP whitelist
    print("Compiling IP whitelist from", access_file, file=sys.stderr)
    access_config = load_module("access_config", access_file)

    network_set = IPSet()

    # Add local networks to the set
    for net in local_networks:
        network_set.add(netaddr.IPNetwork(net))

    # Regular IPs and networks
    exact_ips = set()
    regex_patterns = set()

    try:
        for ip in access_config.allowed_ips:
            try:
                # Handle CIDR notation
                if "/" in ip:
                    network_set.add(netaddr.IPNetwork(ip))
                    continue

                # Handle range in any octet (192.168-170.0.0)
                if "-" in ip:
                    split_numbers = ip.split(".")
                    range_count = sum(1 for octet in split_numbers if "-" in octet)

                    # Multiple ranges in different octets (like 192.168.4-5.10-20)
                    if range_count > 1:
                        expand_ip_range(ip, exact_ips, network_set, regex_patterns)
                        continue

                    # Range in last octet (192.168.0.1-100)
                    if len(split_numbers) == 4 and "-" in split_numbers[3]:
                        base_ip = ".".join(split_numbers[:3])
                        last_part = split_numbers[3]

                        if re.search(r"\d+-\d+", last_part):
                            start, end = map(int, last_part.split("-"))
                            network_set.add(netaddr.IPRange(f"{base_ip}.{start}", f"{base_ip}.{end}"))
                        continue

                    # Handle ranges in other octets
                    if any("-" in octet for octet in split_numbers):
                        expand_ip_range(ip, exact_ips, network_set, regex_patterns)
                        continue

                # Single IP or network prefix
                if ip.count(".") < 3:
                    # Check for trailing dot (like "140.141.")
                    if ip.endswith("."):
                        # Remove trailing dot for pattern creation
                        ip_pattern = ip.rstrip(".")
                        regex_patterns.add(re.compile(f"^{ip_pattern}\\.[0-9]+\\.[0-9]+$"))
                    else:
                        # Network prefix (e.g., "192.168")
                        regex_patterns.add(re.compile(f"^{ip}.*$"))
                else:
                    # Exact IP address
                    exact_ips.add(ip)

                    # Match prefixes of exact IPs as well (e.g., "192.168.1.12" should match "192.168.1.123")
                    regex_patterns.add(re.compile(f"^{re.escape(ip)}.*$"))

            except (ValueError, netaddr.AddrFormatError) as e:
                # Fallback to regex for any formats netaddr can't handle
                print(f"Warning: Could not parse IP '{ip}': {str(e)}", file=sys.stderr)
                regex_patterns.add(re.compile(f"^{ip}.*$"))

    except Exception as e:
        print(f"Error compiling IP whitelist: {repr(e)}", file=sys.stderr)

    # Package all the compiled IPs
    ip_whitelist = {
        "exact_ips": exact_ips,
        "network_set": network_set,  # IPSet instead of list
        "regex_patterns": regex_patterns,
        "timestamp": time.time(),
    }

    # Save compiled whitelist
    try:
        with open(compiled_file, "wb") as f:
            pickle.dump(ip_whitelist, f)
    except Exception as e:
        print(f"Error saving compiled IP whitelist: {repr(e)}", file=sys.stderr)

    return ip_whitelist


def expand_ip_range(ip_with_range, exact_ips_set, network_set, regex_patterns_set):
    """Expand IP ranges like 192.168-170.0.0 into multiple networks"""
    parts = ip_with_range.split(".")

    # For full IP addresses with ranges (all 4 octets specified)
    if len(parts) == 4:
        # Try to use IPRange for efficiency when possible
        has_range = any("-" in p for p in parts)
        if has_range:
            # Count ranges to determine approach
            range_octets = [i for i, p in enumerate(parts) if "-" in p]

            # If only one octet has a range, we can use an efficient IPRange
            if len(range_octets) == 1 and range_octets[0] == 3:  # Last octet only
                base_ip = ".".join(parts[:3])
                start, end = map(int, parts[3].split("-"))
                network_set.add(netaddr.IPRange(f"{base_ip}.{start}", f"{base_ip}.{end}"))
                return

    # Parse each octet, building lists of possible values
    octets = []
    for part in parts:
        if "-" in part and part.count("-") == 1:
            try:
                start, end = map(int, part.split("-"))
                octets.append(list(range(start, end + 1)))
            except ValueError:
                # Handle malformed ranges
                octets.append([0])
        else:
            try:
                octets.append([int(part)])
            except ValueError:
                octets.append([0])

    # Pad with zeros if we have fewer than 4 octets
    while len(octets) < 4:
        octets.append([0])

    # Generate all combinations
    for a in octets[0]:
        for b in octets[1]:
            for c in octets[2]:
                for d in octets[3]:
                    # Create the IP address - use IPSet when possible
                    if len(parts) == 4:  # Complete IP, add to IPSet
                        network_set.add(netaddr.IPAddress(f"{a}.{b}.{c}.{d}"))
                    else:
                        # Add to exact IPs for partial patterns
                        ip = f"{a}.{b}.{c}.{d}"
                        exact_ips_set.add(ip)

    # If we have fewer than 4 octets, create a regex pattern for EACH combination
    if len(parts) < 4:
        # Create regex patterns for all possible combinations up to the parts we have
        for a in octets[0]:
            if len(parts) > 1:
                for b in octets[1]:
                    if len(parts) > 2:
                        for c in octets[2]:
                            # Create pattern based on values so far
                            base = f"{a}.{b}.{c}"
                            regex_patterns_set.add(re.compile(f"^{base}.*$"))
                    else:
                        # Two octets
                        base = f"{a}.{b}"
                        regex_patterns_set.add(re.compile(f"^{base}.*$"))
            else:
                # Single octet
                base = f"{a}"
                regex_patterns_set.add(re.compile(f"^{base}.*$"))




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
    """The client's domain, which the access file's domain_list is matched against and the login screen shows, from a
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
    allowed = _check_address(incoming_address, access_file, mtime is not None)
    connected = connected_address(environ)
    if connected != incoming_address and _check_address(connected, access_file, mtime is not None, log=False) != allowed:
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


def _check_address(incoming_address, access_file, access_file_exists, log=True):
    """Whether access_file lets incoming_address in; refusals are logged, if log."""
    say = print if log else (lambda *args, **kwargs: None)
    if not access_file:
        say(f"UNAUTHORIZED ACCESS TO:{incoming_address}: no access file is defined", file=sys.stderr)
        return False
    if not access_file_exists:
        say(f"ACCESS FILE DOES NOT EXIST. UNAUTHORIZED ACCESS TO: {incoming_address}", file=sys.stderr)
        return False

    # Load access config and IP whitelist
    try:
        access_config = load_module("access_config", access_file)
        ip_whitelist = load_or_compile_ip_whitelist(access_file)
    except Exception as e:
        say("ACCESS ERROR", repr(e), file=sys.stderr)
        say(f"UNAUTHORIZED ACCESS TO:{incoming_address}: can't load access config", file=sys.stderr)
        return False

    # Check blocked IPs
    blocked_ips = set(getattr(access_config, "blocked_ips", []))
    if incoming_address in blocked_ips:
        say(f"BLOCKED IP ACCESS ATTEMPT: {incoming_address}", file=sys.stderr)
        return False

    # Check IP whitelist
    try:
        # 1. Check exact IPs first (fastest)
        if incoming_address in ip_whitelist["exact_ips"]:
            return True

        # 2. Check IP networks using IPSet (much faster)
        try:
            client_ip = netaddr.IPAddress(incoming_address)

            # This is a single O(log n) operation instead of O(n)
            if client_ip in ip_whitelist["network_set"]:
                return True

        except (ValueError, netaddr.AddrFormatError):
            # Skip network checks if IP format is invalid
            pass

        # 3. Check regex patterns (slowest)
        for pattern in ip_whitelist["regex_patterns"]:
            if pattern.search(incoming_address):
                return True
    except Exception as e:
        say(f"Error checking IP whitelist: {repr(e)}", file=sys.stderr)

    # Check domain access, last: it takes a reverse DNS lookup. By substring, and with no forward lookup to confirm
    # the name: stricter rules would have refused subscribers' VPNs and proxies in the 2025 logs of artflsrv04.
    domain_list = set(getattr(access_config, "domain_list", []))
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
