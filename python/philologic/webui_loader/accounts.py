"""Accounts of the philologic5-webui-loader service on a distant machine: passwords (scrypt), a second factor (TOTP,
with recovery codes), sessions, limits on failed logins, who may do what, and an audit log. All in an SQLite database
in the state directory of the service, readable by the service account only."""

import base64
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import sqlite3
import struct
import threading
import time
import urllib.parse
from contextlib import contextmanager

# OWASP parameters for scrypt: 128 MiB of memory per hash
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**17, 8, 1
SCRYPT_MAXMEM = 256 * 1024**2
MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 1024
TOTP_PERIOD = 30
TOTP_DIGITS = 6
RECOVERY_CODES = 10
PARTIAL_SESSION_TIME = 5 * 60  # to give the second factor after the password
ROLES = ("admin", "loader")
ISSUER = "PhiloLogic loader"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    name TEXT PRIMARY KEY,
    password TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'loader')),
    totp_secret TEXT,
    totp_last_counter INTEGER NOT NULL DEFAULT 0,
    must_change_password INTEGER NOT NULL DEFAULT 0,
    disabled INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS recovery_codes (
    user TEXT NOT NULL REFERENCES users(name) ON DELETE CASCADE,
    code_hash TEXT NOT NULL,
    used REAL
);
CREATE TABLE IF NOT EXISTS sessions (
    id_hash TEXT PRIMARY KEY,
    user TEXT NOT NULL REFERENCES users(name) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    csrf TEXT NOT NULL,
    pending_totp_secret TEXT,
    created REAL NOT NULL,
    last_seen REAL NOT NULL,
    ip TEXT
);
CREATE TABLE IF NOT EXISTS login_failures (
    time REAL NOT NULL,
    user TEXT,
    ip TEXT
);
CREATE INDEX IF NOT EXISTS login_failures_time ON login_failures(time);
CREATE TABLE IF NOT EXISTS grants (
    user TEXT NOT NULL REFERENCES users(name) ON DELETE CASCADE,
    dbname TEXT NOT NULL,
    PRIMARY KEY (user, dbname)
);
CREATE TABLE IF NOT EXISTS database_owners (
    dbname TEXT PRIMARY KEY,
    user TEXT NOT NULL,
    loaded REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS trusted_browsers (
    token_hash TEXT PRIMARY KEY,
    user TEXT NOT NULL REFERENCES users(name) ON DELETE CASCADE,
    created REAL NOT NULL,
    expires REAL NOT NULL,
    last_used REAL,
    ip TEXT
);
CREATE TABLE IF NOT EXISTS audit (
    time REAL NOT NULL,
    user TEXT,
    ip TEXT,
    action TEXT NOT NULL,
    detail TEXT
);
"""


class AccountError(Exception):
    """A refused login or account change: the message can be shown to the user"""


def b64(data):
    return base64.b64encode(data).decode("ascii")


def hash_password(password):
    salt = os.urandom(16)
    key = hashlib.scrypt(
        password.encode("utf8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, maxmem=SCRYPT_MAXMEM, dklen=32
    )
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${b64(salt)}${b64(key)}"


def verify_password(password, stored):
    try:
        algorithm, n, r, p, salt, key = stored.split("$")
        if algorithm != "scrypt":
            return False
        computed = hashlib.scrypt(
            password.encode("utf8"),
            salt=base64.b64decode(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            maxmem=SCRYPT_MAXMEM,
            dklen=len(base64.b64decode(key)),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(computed, base64.b64decode(key))


_dummy_hash = None


def dummy_verify(password):
    """Takes as long as checking a password, for unknown users: response times don't tell which users exist"""
    global _dummy_hash
    if _dummy_hash is None:
        _dummy_hash = hash_password(secrets.token_urlsafe(16))
    verify_password(password, _dummy_hash)


def password_problem(password, name):
    if not isinstance(password, str) or len(password) < MIN_PASSWORD_LENGTH:
        return f"a password needs at least {MIN_PASSWORD_LENGTH} characters"
    if len(password) > MAX_PASSWORD_LENGTH:
        return "this password is too long"
    if password.lower() == name.lower():
        return "a password can't be the user name"
    return None


def new_totp_secret():
    return base64.b32encode(os.urandom(20)).decode("ascii").rstrip("=")


def totp_code(secret, counter):
    """RFC 6238 code (HMAC-SHA1, 6 digits) of a time step"""
    key = base64.b32decode(secret + "=" * (-len(secret) % 8))
    digest = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code = (struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF) % 10**TOTP_DIGITS
    return f"{code:0{TOTP_DIGITS}d}"


def totp_counter(secret, code, last_counter, now=None):
    """The time step of a valid code (one step of clock drift either way), newer than the last one used (a code
    can't be used twice), or None"""
    code = (code if isinstance(code, str) else "").replace(" ", "")
    if len(code) != TOTP_DIGITS or not code.isdigit():
        return None
    current = int((now or time.time()) // TOTP_PERIOD)
    for counter in (current - 1, current, current + 1):
        if counter > last_counter and hmac.compare_digest(totp_code(secret, counter), code):
            return counter
    return None


def totp_uri(secret, name):
    label = urllib.parse.quote(f"{ISSUER}:{name}")
    return f"otpauth://totp/{label}?secret={secret}&issuer={urllib.parse.quote(ISSUER)}&digits={TOTP_DIGITS}&period={TOTP_PERIOD}"


def hash_token(token):
    return hashlib.sha256(token.encode("utf8")).hexdigest()


def normalize_recovery_code(code):
    return (code if isinstance(code, str) else "").replace("-", "").replace(" ", "").upper()


def address_key(ip):
    """What failed logins are counted by: the address, or its /64 network for IPv6 (which one machine usually has)"""
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return ip or ""
    if address.version == 6:
        if address.ipv4_mapped:
            return str(address.ipv4_mapped)
        return str(ipaddress.ip_network(f"{address}/64", strict=False))
    return str(address)


# Password checks at once: each takes 128 MiB of memory (scrypt), so they are limited for logins by unknown clients
PASSWORD_CHECKS = threading.BoundedSemaphore(2)


class Accounts:
    """The accounts database of the service"""

    def __init__(self, path, settings):
        self.path = path
        self.settings = settings
        self.local = threading.local()
        directory = os.path.dirname(path)
        os.makedirs(directory, mode=0o700, exist_ok=True)
        if not os.path.exists(path):
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT, 0o600))
        os.chmod(path, 0o600)
        self.thread_connection().executescript(SCHEMA)  # which commits by itself

    def thread_connection(self):
        connection = getattr(self.local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            self.local.connection = connection
        return connection

    @contextmanager
    def connection(self):
        """The connection of this thread, in a transaction"""
        connection = self.thread_connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")

    # Audit

    def audit(self, action, user=None, ip=None, **detail):
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO audit VALUES (?, ?, ?, ?, ?)",
                (time.time(), user, ip, action, json.dumps(detail) if detail else None),
            )

    def audit_log(self, limit=500):
        with self.connection() as connection:
            rows = connection.execute("SELECT * FROM audit ORDER BY time DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) | {"detail": json.loads(row["detail"]) if row["detail"] else None} for row in rows]

    # Users

    def user(self, name):
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM users WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None

    def users(self):
        with self.connection() as connection:
            rows = connection.execute("SELECT * FROM users ORDER BY name").fetchall()
            grants = connection.execute("SELECT user, dbname FROM grants ORDER BY dbname").fetchall()
            trusted = dict(
                connection.execute(
                    "SELECT user, COUNT(*) FROM trusted_browsers WHERE expires > ? GROUP BY user", (time.time(),)
                ).fetchall()
            )
        granted = {}
        for grant in grants:
            granted.setdefault(grant["user"], []).append(grant["dbname"])
        return [
            {
                "name": row["name"],
                "role": row["role"],
                "totp_enabled": bool(row["totp_secret"]),
                "must_change_password": bool(row["must_change_password"]),
                "disabled": bool(row["disabled"]),
                "created": row["created"],
                "grants": granted.get(row["name"], []),
                "trusted_browsers": trusted.get(row["name"], 0),
            }
            for row in rows
        ]

    def add_user(self, name, role="loader", password=None):
        """Add a user, with a temporary password (returned) to change at the first login unless one is given"""
        if not name or not name.replace("_", "").replace("-", "").replace(".", "").isalnum() or len(name) > 64:
            raise AccountError("a user name is made of letters, digits, '.', '_' and '-'")
        if role not in ROLES:
            raise AccountError(f"a role is one of {', '.join(ROLES)}")
        temporary = password is None
        if temporary:
            password = secrets.token_urlsafe(12)
        else:
            problem = password_problem(password, name)
            if problem:
                raise AccountError(problem)
        with self.connection() as connection:
            if connection.execute("SELECT 1 FROM users WHERE name = ?", (name,)).fetchone():
                raise AccountError(f"there is already a user {name}")
            connection.execute(
                "INSERT INTO users (name, password, role, must_change_password, created) VALUES (?, ?, ?, ?, ?)",
                (name, hash_password(password), role, int(temporary), time.time()),
            )
        return password

    def update_user(self, name, role=None, disabled=None):
        with self.connection() as connection:
            user = connection.execute("SELECT role, disabled FROM users WHERE name = ?", (name,)).fetchone()
            if user is None:
                raise AccountError(f"no user {name}")
            if role is not None:
                if role not in ROLES:
                    raise AccountError(f"a role is one of {', '.join(ROLES)}")
                connection.execute("UPDATE users SET role = ? WHERE name = ?", (role, name))
            if disabled is not None:
                connection.execute("UPDATE users SET disabled = ? WHERE name = ?", (int(disabled), name))
                if disabled:
                    connection.execute("DELETE FROM sessions WHERE user = ?", (name,))
                    connection.execute("DELETE FROM trusted_browsers WHERE user = ?", (name,))
            # An admin can't be demoted or disabled if they are the last one
            if user["role"] == "admin" and not user["disabled"] and (role not in (None, "admin") or disabled):
                admins = connection.execute(
                    "SELECT COUNT(*) FROM users WHERE role = 'admin' AND disabled = 0"
                ).fetchone()[0]
                if admins == 0:
                    raise AccountError("there must be an admin left")

    def remove_user(self, name):
        with self.connection() as connection:
            row = connection.execute("SELECT role FROM users WHERE name = ?", (name,)).fetchone()
            if row is None:
                raise AccountError(f"no user {name}")
            connection.execute("DELETE FROM users WHERE name = ?", (name,))
            connection.execute("DELETE FROM database_owners WHERE user = ?", (name,))
            if row["role"] == "admin":
                admins = connection.execute(
                    "SELECT COUNT(*) FROM users WHERE role = 'admin' AND disabled = 0"
                ).fetchone()[0]
                if admins == 0:
                    raise AccountError("there must be an admin left")

    def reset_password(self, name):
        """A new temporary password, to change at the next login; the user's sessions end"""
        password = secrets.token_urlsafe(12)
        with self.connection() as connection:
            if not connection.execute("SELECT 1 FROM users WHERE name = ?", (name,)).fetchone():
                raise AccountError(f"no user {name}")
            connection.execute(
                "UPDATE users SET password = ?, must_change_password = 1 WHERE name = ?",
                (hash_password(password), name),
            )
            connection.execute("DELETE FROM sessions WHERE user = ?", (name,))
            connection.execute("DELETE FROM trusted_browsers WHERE user = ?", (name,))
        return password

    def reset_totp(self, name):
        """Remove the second factor of a user, who sets up a new one at the next login; the user's sessions end"""
        with self.connection() as connection:
            if not connection.execute("SELECT 1 FROM users WHERE name = ?", (name,)).fetchone():
                raise AccountError(f"no user {name}")
            connection.execute("UPDATE users SET totp_secret = NULL, totp_last_counter = 0 WHERE name = ?", (name,))
            connection.execute("DELETE FROM recovery_codes WHERE user = ?", (name,))
            connection.execute("DELETE FROM sessions WHERE user = ?", (name,))
            connection.execute("DELETE FROM trusted_browsers WHERE user = ?", (name,))

    def change_password(self, name, current, new, keep_session=None, keep_browser=None):
        user = self.user(name)
        if user is None or not verify_password(current or "", user["password"]):
            raise AccountError("the current password is wrong")
        problem = password_problem(new, name)
        if problem:
            raise AccountError(problem)
        if new == current:
            raise AccountError("the new password must be different")
        with self.connection() as connection:
            connection.execute(
                "UPDATE users SET password = ?, must_change_password = 0 WHERE name = ?", (hash_password(new), name)
            )
            # Other sessions end
            connection.execute(
                "DELETE FROM sessions WHERE user = ? AND id_hash != ?", (name, hash_token(keep_session or ""))
            )
            # And the browsers trusted for the second factor, but this one
            connection.execute(
                "DELETE FROM trusted_browsers WHERE user = ? AND token_hash != ?",
                (name, hash_token(keep_browser if isinstance(keep_browser, str) else "")),
            )

    # Logins

    def failures(self, connection, user=None, ip=None):
        since = time.time() - self.settings.lockout_time
        if user is not None:
            return connection.execute(
                "SELECT COUNT(*) FROM login_failures WHERE time > ? AND user = ?", (since, user)
            ).fetchone()[0]
        return connection.execute(
            "SELECT COUNT(*) FROM login_failures WHERE time > ? AND ip = ?", (since, ip)
        ).fetchone()[0]

    def check_limits(self, name, ip):
        ip = address_key(ip)
        with self.connection() as connection:
            connection.execute("DELETE FROM trusted_browsers WHERE expires < ?", (time.time(),))
            connection.execute("DELETE FROM login_failures WHERE time < ?", (time.time() - self.settings.lockout_time,))
            if self.failures(connection, ip=ip) >= self.settings.max_failed_logins_per_ip:
                raise AccountError("too many failed logins from this address: try again later")
            if name and self.failures(connection, user=name) >= self.settings.max_failed_logins:
                raise AccountError("too many failed logins for this account: try again later")

    def record_failure(self, name, ip, step):
        with self.connection() as connection:
            connection.execute("INSERT INTO login_failures VALUES (?, ?, ?)", (time.time(), name, address_key(ip)))
        self.audit("login_failed", name, ip, step=step)

    def new_session(self, connection, name, stage, ip, pending_totp_secret=None):
        token = secrets.token_urlsafe(32)
        now = time.time()
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (hash_token(token), name, stage, secrets.token_urlsafe(32), pending_totp_secret, now, now, ip),
        )
        return token

    def login(self, name, password, ip, trusted_browser=None):
        """Check a password: returns (session token, next step): "totp" to give the code of the second factor,
        "totp_setup" to set one up, or "done" in a browser trusted for the second factor (trusted_browser, the token of
        its cookie), which then gets a full session. Raises AccountError."""
        name = name if isinstance(name, str) else ""
        password = password if isinstance(password, str) else ""
        # The limits are checked while holding one of the few password checks: many logins at once can't all get past
        if not PASSWORD_CHECKS.acquire(timeout=30):
            raise AccountError("the server is busy: try again in a moment")
        try:
            self.check_limits(name, ip)
            user = self.user(name)
            if user is None or user["disabled"]:
                dummy_verify(password)
                self.record_failure(name, ip, "password")
                raise AccountError("wrong user name or password")
            if not verify_password(password, user["password"]):
                self.record_failure(name, ip, "password")
                raise AccountError("wrong user name or password")
        finally:
            PASSWORD_CHECKS.release()
        with self.connection() as connection:
            if user["totp_secret"] and self.is_trusted(connection, name, trusted_browser):
                connection.execute(
                    "UPDATE trusted_browsers SET last_used = ? WHERE token_hash = ?",
                    (time.time(), hash_token(trusted_browser)),
                )
                token, step = self.new_session(connection, name, "full", ip), "done"
            elif user["totp_secret"]:
                return self.new_session(connection, name, "password", ip), "totp"
            else:
                secret = new_totp_secret()
                return self.new_session(connection, name, "totp_setup", ip, secret), "totp_setup"
        self.audit("login", name, ip, trusted_browser=True)
        return token, step

    def is_trusted(self, connection, name, token):
        """Whether a browser (the token of its cookie) is trusted for the second factor of a user"""
        if not token or not isinstance(token, str):
            return False
        row = connection.execute(
            "SELECT expires FROM trusted_browsers WHERE token_hash = ? AND user = ?", (hash_token(token), name)
        ).fetchone()
        return row is not None and row["expires"] > time.time()

    def browser_trusted(self, name, token):
        with self.connection() as connection:
            return self.is_trusted(connection, name, token)

    def trust_browser(self, connection, name, ip):
        """The token of a browser trusted for the second factor of a user, for trusted_browser_days (None if 0)"""
        days = self.settings.trusted_browser_days
        if not days:
            return None
        token = secrets.token_urlsafe(32)
        now = time.time()
        connection.execute(
            "INSERT INTO trusted_browsers VALUES (?, ?, ?, ?, NULL, ?)",
            (hash_token(token), name, now, now + days * 86400, ip),
        )
        return token

    def forget_browsers(self, name):
        """No browser of a user is trusted for the second factor any more"""
        with self.connection() as connection:
            connection.execute("DELETE FROM trusted_browsers WHERE user = ?", (name,))

    def session(self, token, touch=True):
        """The session of a token, if valid, with its user: {user, role, stage, csrf, must_change_password, ...}"""
        if not token:
            return None
        now = time.time()
        with self.connection() as connection:
            row = connection.execute(
                """SELECT sessions.*, users.role, users.must_change_password, users.disabled FROM sessions
                JOIN users ON users.name = sessions.user WHERE id_hash = ?""",
                (hash_token(token),),
            ).fetchone()
            if row is None:
                return None
            idle_limit = PARTIAL_SESSION_TIME if row["stage"] != "full" else self.settings.session_idle_timeout
            if (
                row["disabled"]
                or now - row["last_seen"] > idle_limit
                or now - row["created"] > self.settings.session_max_age
            ):
                connection.execute("DELETE FROM sessions WHERE id_hash = ?", (row["id_hash"],))
                return None
            if touch and now - row["last_seen"] > 60:
                connection.execute("UPDATE sessions SET last_seen = ? WHERE id_hash = ?", (now, row["id_hash"]))
        return dict(row)

    def complete_login(self, connection, session, ip):
        """Replace a partial session by a full one (a new token, against session fixation)"""
        connection.execute("DELETE FROM sessions WHERE id_hash = ?", (session["id_hash"],))
        return self.new_session(connection, session["user"], "full", ip)

    def verify_second_factor(self, token, code, ip, trust=False):
        """Check the code of the second factor (or a recovery code) of a partial session: returns the token of the
        full session, and if trust, the token of the browser, then trusted for the second factor (or None).
        Raises AccountError."""
        session = self.session(token)
        if session is None or session["stage"] != "password":
            raise AccountError("log in again")
        name = session["user"]
        self.check_limits(name, ip)
        with self.connection() as connection:
            # In the transaction: a code can't be used by two logins at once
            user = connection.execute(
                "SELECT totp_secret, totp_last_counter FROM users WHERE name = ?", (name,)
            ).fetchone()
            counter = totp_counter(user["totp_secret"], code, user["totp_last_counter"])
            if counter is not None:
                updated = connection.execute(
                    "UPDATE users SET totp_last_counter = ? WHERE name = ? AND totp_last_counter < ?",
                    (counter, name, counter),
                ).rowcount
                new_token = self.complete_login(connection, session, ip) if updated else None
            else:
                code_hash = hash_token(normalize_recovery_code(code))
                row = connection.execute(
                    "SELECT rowid FROM recovery_codes WHERE user = ? AND code_hash = ? AND used IS NULL",
                    (name, code_hash),
                ).fetchone()
                if row is None:
                    new_token = None
                else:
                    connection.execute(
                        "UPDATE recovery_codes SET used = ? WHERE rowid = ?", (time.time(), row["rowid"])
                    )
                    new_token = self.complete_login(connection, session, ip)
        if new_token is None:
            self.record_failure(name, ip, "second factor")
            raise AccountError("wrong code")
        with self.connection() as connection:
            browser = self.trust_browser(connection, name, ip) if trust else None
        self.audit("login", name, ip, recovery_code=counter is None, trusted_browser_added=browser is not None)
        return new_token, browser

    def totp_setup(self, token):
        """The secret to set up in an authenticator app, for a partial session which must set one up"""
        session = self.session(token)
        if session is None or session["stage"] != "totp_setup":
            raise AccountError("log in again")
        secret = session["pending_totp_secret"]
        return {"secret": secret, "uri": totp_uri(secret, session["user"])}

    def confirm_totp_setup(self, token, code, ip, trust=False):
        """Enable the second factor of a partial session with a first code of it: returns (token of the full session,
        recovery codes, shown once, token of the browser if trusted)"""
        session = self.session(token)
        if session is None or session["stage"] != "totp_setup":
            raise AccountError("log in again")
        name = session["user"]
        self.check_limits(name, ip)
        secret = session["pending_totp_secret"]
        counter = totp_counter(secret, code, 0)
        if counter is None:
            self.record_failure(name, ip, "second factor setup")
            raise AccountError("wrong code: check the time of your device")
        codes = [
            "-".join(chunk for chunk in (raw[:5], raw[5:]))
            for raw in (base64.b32encode(os.urandom(10)).decode("ascii")[:10] for _ in range(RECOVERY_CODES))
        ]
        with self.connection() as connection:
            connection.execute(
                "UPDATE users SET totp_secret = ?, totp_last_counter = ? WHERE name = ?", (secret, counter, name)
            )
            connection.execute("DELETE FROM recovery_codes WHERE user = ?", (name,))
            connection.executemany(
                "INSERT INTO recovery_codes (user, code_hash) VALUES (?, ?)",
                [(name, hash_token(normalize_recovery_code(code))) for code in codes],
            )
            new_token = self.complete_login(connection, session, ip)
            browser = self.trust_browser(connection, name, ip) if trust else None
        self.audit("totp_setup", name, ip)
        self.audit("login", name, ip, trusted_browser_added=browser is not None)
        return new_token, codes, browser

    def logout(self, token):
        with self.connection() as connection:
            connection.execute("DELETE FROM sessions WHERE id_hash = ?", (hash_token(token or ""),))

    # Permissions

    def grant(self, name, dbname):
        with self.connection() as connection:
            if not connection.execute("SELECT 1 FROM users WHERE name = ?", (name,)).fetchone():
                raise AccountError(f"no user {name}")
            connection.execute("INSERT OR IGNORE INTO grants VALUES (?, ?)", (name, dbname))

    def revoke(self, name, dbname):
        with self.connection() as connection:
            connection.execute("DELETE FROM grants WHERE user = ? AND dbname = ?", (name, dbname))

    def set_owner(self, dbname, name, new_database=False):
        """Record who loaded a database with the service first: loading it again doesn't change it, unless it didn't
        exist (a database of the same name may have been deleted)"""
        with self.connection() as connection:
            verb = "INSERT OR REPLACE" if new_database else "INSERT OR IGNORE"
            connection.execute(f"{verb} INTO database_owners VALUES (?, ?, ?)", (dbname, name, time.time()))

    def owners(self):
        with self.connection() as connection:
            return {row["dbname"]: row["user"] for row in connection.execute("SELECT * FROM database_owners")}

    def may_edit(self, name, role, dbname):
        """Whether a user may edit the web config of a database, or load it again: admins, the user who loaded it
        with the service, and those it was granted to"""
        if role == "admin":
            return True
        with self.connection() as connection:
            owner = connection.execute("SELECT user FROM database_owners WHERE dbname = ?", (dbname,)).fetchone()
            if owner is not None and owner["user"] == name:
                return True
            return (
                connection.execute("SELECT 1 FROM grants WHERE user = ? AND dbname = ?", (name, dbname)).fetchone()
                is not None
            )
