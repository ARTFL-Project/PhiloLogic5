#!/var/lib/philologic5/philologic_env/bin/python3

import hashlib
import os
import pickle
import socket
import sys
import time
from urllib.parse import unquote

import netaddr
import regex as re
from netaddr import IPSet
from philologic.runtime.DB import DB
from philologic.utils import load_module

# These should always be allowed for local access
local_networks = ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8"]
ip_ranges = [re.compile(rf"^{i.split('/')[0]}.*") for i in local_networks]  # For backward compatibility

# Cached IP whitelist info
COMPILED_IP_SUFFIX = ".compiled_ips"


def load_or_compile_ip_whitelist(access_file):
    """Load compiled IP whitelist or recompile if outdated"""
    compiled_file = access_file + COMPILED_IP_SUFFIX
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


def check_access(environ, config):
    """Check for access with cached IP whitelist"""
    db = DB(config.db_path + "/data/")
    incoming_address, match_domain = get_client_info(environ)

    # Verify access file exists
    if not config.access_file:
        print(
            f"UNAUTHORIZED ACCESS TO:{incoming_address} from domain {match_domain} no access file is defined",
            file=sys.stderr,
        )
        return ()

    # Use absolute or relative path as appropriate
    access_file = (
        config.access_file
        if os.path.isabs(config.access_file)
        else os.path.join(config.db_path, "data", config.access_file)
    )
    if not os.path.isfile(access_file):
        print(
            f"ACCESS FILE DOES NOT EXIST. UNAUTHORIZED ACCESS TO: {incoming_address} from domain {match_domain}: access file does not exist",
            file=sys.stderr,
        )
        return ()

    # Load access config and IP whitelist
    try:
        access_config = load_module("access_config", access_file)
        ip_whitelist = load_or_compile_ip_whitelist(access_file)
    except Exception as e:
        print("ACCESS ERROR", repr(e), file=sys.stderr)
        print(
            f"UNAUTHORIZED ACCESS TO:{incoming_address} from domain {match_domain}: can't load access config",
            file=sys.stderr,
        )
        return ()

    # Check blocked IPs
    blocked_ips = set(getattr(access_config, "blocked_ips", []))
    if incoming_address in blocked_ips:
        print(f"BLOCKED IP ACCESS ATTEMPT: {incoming_address}", file=sys.stderr)
        return ()

    # Check domain access
    domain_list = set(getattr(access_config, "domain_list", []))
    if match_domain in domain_list:
        return make_token(db)
    for domain in domain_list:
        if domain in match_domain:
            return make_token(db)

    # Check IP whitelist
    try:
        # 1. Check exact IPs first (fastest)
        if incoming_address in ip_whitelist["exact_ips"]:
            return make_token(db)

        # 2. Check IP networks using IPSet (much faster)
        try:
            client_ip = netaddr.IPAddress(incoming_address)

            # This is a single O(log n) operation instead of O(n)
            if client_ip in ip_whitelist["network_set"]:
                return make_token(db)

        except (ValueError, netaddr.AddrFormatError):
            # Skip network checks if IP format is invalid
            pass

        # 3. Check regex patterns (slowest)
        for pattern in ip_whitelist["regex_patterns"]:
            if pattern.search(incoming_address):
                return make_token(db)
    except Exception as e:
        print(f"Error checking IP whitelist: {repr(e)}", file=sys.stderr)

    # If no match found, access denied
    print(
        f"UNAUTHORIZED ACCESS TO:{incoming_address} from domain {match_domain}: IP not in whitelist",
        file=sys.stderr,
    )
    return ()


def get_client_info(environ):
    incoming_address = environ["REMOTE_ADDR"]
    fq_domain_name = socket.getfqdn(incoming_address).split(",")[-1]
    edit_domain = re.split(r"\.", fq_domain_name)

    if re.match("edu", edit_domain[-1]):
        match_domain = ".".join([edit_domain[-2], edit_domain[-1]])
    else:
        if len(edit_domain) == 2:
            match_domain = ".".join([edit_domain[-2], edit_domain[-1]])
        else:
            match_domain = fq_domain_name
    return incoming_address, match_domain


def login_access(environ, request, config, headers):
    db = DB(config.db_path + "/data/")
    if request.authenticated:
        access = True
    else:
        if request.username and request.password:
            access = check_login_info(config, request)
            if access:
                token = make_token(db)
                if token:
                    h, ts = token
                    headers.append(("Set-Cookie", f"hash={h}"))
                    headers.append(("Set-Cookie", f"timestamp={ts}"))
        else:
            # WORKAROUND because cookie not being sent on access_request.py request
            token = check_access(environ, config)
            if token:
                h, ts = token
                headers.append(("Set-Cookie", f"hash={h}"))
                headers.append(("Set-Cookie", f"timestamp={ts}"))
                access = True
            else:
                access = False
    return access, headers


def check_login_info(config, request):
    login_file_path = os.path.join(config.db_path, "data/logins.txt")
    unquoted_password = unquote(request.password)
    if os.path.exists(login_file_path):
        with open(login_file_path, "rb") as password_file:
            for line in password_file:
                try:
                    line = line.decode("utf8", "ignore")
                except UnicodeDecodeError:
                    continue
                line = line.strip()
                if not line:  # empty line
                    continue
                fields = line.split("\t")
                user = fields[0]
                passwd = fields[1]
                if user == request.username and passwd == unquoted_password:
                    return True
            return False
    else:
        return False


def make_token(db):
    h = hashlib.md5()
    now = str(time.time())
    h.update(now.encode("utf8"))
    h.update(db.locals.secret.encode("utf8"))
    return (h.hexdigest(), now)


def run_tests():
    """Run test cases to verify IP matching logic"""
    import tempfile

    # Explicitly tell Python we're using the global variables
    global DB
    global get_client_info
    global ip_ranges
    global local_networks

    # Store original functions and variables
    original_db = DB
    original_get_client_info_func = get_client_info
    original_ip_ranges = ip_ranges.copy()
    original_local_networks = local_networks.copy()

    # Disable local networks during testing
    ip_ranges = []
    local_networks = []

    # Define test cases
    test_cases = [
        # Format: (description, ip, domain, expected_result)
        # 1. Exact IP matching
        ("Exact IP match", "192.168.1.1", "example.org", True),
        ("Exact IP non-match", "192.168.1.2", "example.org", False),
        # 2. CIDR notation
        ("CIDR match start", "172.16.0.1", "example.org", True),
        ("CIDR match end", "172.16.255.255", "example.org", True),
        ("CIDR non-match", "172.17.0.1", "example.org", False),
        # 3. Range in last octet
        ("Last octet range match start", "192.168.2.10", "example.org", True),
        ("Last octet range match middle", "192.168.2.15", "example.org", True),
        ("Last octet range match end", "192.168.2.20", "example.org", True),
        ("Last octet range non-match below", "192.168.2.9", "example.org", False),
        ("Last octet range non-match above", "192.168.2.21", "example.org", False),
        # 4. Range in non-last octet
        ("Non-last octet range match start", "192.168.0.0", "example.org", True),
        ("Non-last octet range match end", "192.170.0.0", "example.org", True),
        ("Non-last octet range non-match", "192.171.0.0", "example.org", False),
        # 5. Multiple ranges
        ("Multiple ranges match", "192.168.4.15", "example.org", True),
        ("Multiple ranges match edge", "192.168.5.20", "example.org", True),
        ("Multiple ranges non-match", "192.168.6.15", "example.org", False),
        # 6. Network prefix
        ("Network prefix match", "172.18.0.1", "example.org", True),
        ("Network prefix match edge", "172.18.255.255", "example.org", True),
        ("Network prefix non-match", "172.19.0.1", "example.org", False),
        # 7. Local networks (disabled during testing)
        ("Local network 10.x.x.x", "10.1.2.3", "example.org", False),
        ("Local network 172.16.x.x", "172.16.5.10", "example.org", True),  # Still True due to CIDR match
        ("Local network 192.168.x.x", "192.168.0.1", "example.org", False),
        ("Local network 127.x.x.x", "127.0.0.1", "example.org", False),
        # 8. Domain matching
        ("Domain exact match", "203.0.113.1", "example.com", True),
        ("Domain suffix match", "203.0.113.1", "university.edu", True),
        ("Domain non-match", "203.0.113.1", "example.net", False),
        # 9. Blocked IPs
        ("Blocked IP", "10.0.0.99", "example.org", False),
        ("Blocked IP overrides allowed network", "192.168.5.5", "example.org", False),
        # 10. Additional network prefix cases
        ("Network prefix with 2 octets", "134.226.12.34", "example.org", True),
        ("Network prefix with 3 octets", "192.70.186.50", "example.org", True),
        ("Network prefix with trailing dot", "140.141.10.20", "example.org", True),
        ("Network prefix with trailing dot non-match", "140.142.10.20", "example.org", False),
        ("Multiple adjacent octets range", "216.87.19.50", "example.org", True),
        ("Multiple adjacent octets range", "216.87.20.50", "example.org", True),
    ]

    # Create test config with all necessary patterns
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            """
# Test access config with real-world examples
domain_list = [".edu", "example.com"]
blocked_ips = ["10.0.0.99", "203.0.113.100", "192.168.5.5"]
allowed_ips = [
    # Basic patterns
    "192.168.1.1",              # Exact IP match
    "172.16.0.0/16",            # CIDR notation
    "192.168.2.10-20",          # Range in last octet
    "192.168-170.0.0",          # Range in non-last octet
    "192.168.4-5.10-20",        # Multiple ranges
    "172.18",                   # Network prefix

    # Network prefixes and ranges
    "134.226",                  # Network prefix with 2 octets
    "192.70.186",               # Network prefix with 3 octets
    "140.141.",                 # Network prefix with trailing dot

    # Multiple adjacent octets range - missing from your list
    "216.87.19-20",             # Should match 216.87.19.* and 216.87.20.*
]
"""
        )
        test_config_path = f.name

    print("\n========== ACCESS CONTROL TEST CASES ==========")

    # Create a single mock DB and config
    class MockConfig:
        def __init__(self, path):
            self.db_path = path
            self.access_file = test_config_path

    class MockDB:
        class Locals:
            def __init__(self):
                self.secret = "test-secret"

        def __init__(self, path=None):
            self.locals = MockDB.Locals()
            self.path = path

    # Make DB return our mock
    DB = MockDB

    # Set up a single temporary directory for all tests
    with tempfile.TemporaryDirectory() as temp_dir:
        os.makedirs(os.path.join(temp_dir, "data"), exist_ok=True)
        mock_config = MockConfig(temp_dir)

        # Pre-compile the whitelist once for all tests
        get_client_info = lambda e: (e["REMOTE_ADDR"], "example.org")
        _ = check_access({"REMOTE_ADDR": "127.0.0.1"}, mock_config)

        # Run all tests with the same config
        for description, ip, domain, expected in test_cases:
            # Create mock environment
            env = {"REMOTE_ADDR": ip}

            # Update domain for this test
            get_client_info = lambda e: (e["REMOTE_ADDR"], domain)

            # Run test with the shared config
            result = bool(check_access(env, mock_config))

            # Check result
            status = "✓ PASS" if result == expected else "✗ FAIL"
            print(f"{status} - {description}: IP={ip}, Domain={domain}, Expected={expected}, Got={result}")

    # Restore original state
    get_client_info = original_get_client_info_func
    DB = original_db
    ip_ranges = original_ip_ranges
    local_networks = original_local_networks

    # Clean up
    os.unlink(test_config_path)
    print("========== TEST COMPLETE ==========\n")


if __name__ == "__main__":
    run_tests()
