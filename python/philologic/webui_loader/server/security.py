"""What every request of the philologic5-webui-loader server goes through before its resource: the URL prefix,
the Host and HTTPS checks, security headers, who is asking, and the Origin and CSRF checks of requests which change
something.

On your own machine, the token of the URL printed by philologic5-webui-loader is sent by the page as an Authorization
header, from its localStorage, which only pages of the same origin (port included) can read: browsers send cookies of
localhost to all its ports, so a cookie could be picked up by another local server. In the service, logins make a
session, kept in an HttpOnly cookie limited to the path of the service, with SameSite=Strict and a CSRF token."""

import hashlib
import hmac
import ipaddress
import os
import pwd

import falcon

SESSION_COOKIE = "philologic5_webui_session"
TRUSTED_COOKIE = "philologic5_webui_trusted"
CSRF_HEADER = "X-CSRF-Token"
UNSAFE_METHODS = ("POST", "PUT", "PATCH", "DELETE")

# Who can use a resource is set by its class attributes, not by paths (which the router normalizes differently):
#   public = True: without being logged in (the files of the client, the session, the steps of logins)
#   before_password_change = True: by a user who must change their password
# Any other resource needs a logged in user, and the CSRF token of the session for requests which change something.

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
    "font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; "
    "frame-ancestors 'none'"
)


class StripPrefix:
    """WSGI middleware taking the URL prefix of the service (SERVICE_PATH) off the path, whether the
    reverse proxy passed it on or not. The prefix itself is redirected to prefix/, where the relative URLs of the
    client's files lead."""

    def __init__(self, app, prefix):
        self.app = app
        self.prefix = prefix

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if self.prefix and path == self.prefix and environ.get("REQUEST_METHOD") in ("GET", "HEAD"):
            start_response(
                "301 Moved Permanently",
                [("Location", self.prefix + "/"), ("Content-Type", "text/plain"), ("Content-Length", "0")],
            )
            return [b""]
        if self.prefix and (path == self.prefix or path.startswith(self.prefix + "/")):
            environ["SCRIPT_NAME"] = environ.get("SCRIPT_NAME", "") + self.prefix
            environ["PATH_INFO"] = path[len(self.prefix) :] or "/"
        return self.app(environ, start_response)


def current_user():
    return pwd.getpwuid(os.geteuid()).pw_name


def personal_csrf(token):
    return hmac.new(token.encode("utf8"), b"csrf", hashlib.sha256).hexdigest()


def client_ip(req, settings):
    """Address of the client: the one a trusted reverse proxy gives, if it came through one"""
    remote = req.remote_addr or ""
    if settings.forwarded_allow_ips and remote in settings.forwarded_allow_ips:
        forwarded = req.get_header("X-Forwarded-For")
        if forwarded:
            candidate = forwarded.split(",")[-1].strip()
            try:
                return str(ipaddress.ip_address(candidate))
            except ValueError:
                pass
    return remote


def bind_port(bind):
    try:
        return int(bind.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return None


class SecurityMiddleware:
    def __init__(self, settings, accounts=None):
        self.settings = settings
        self.accounts = accounts
        if not settings.service:
            port = bind_port(settings.bind)
            self.hosts = {f"localhost:{port}", f"127.0.0.1:{port}", f"[::1]:{port}"}
            self.origins = {f"http://{host}" for host in self.hosts}
        else:
            # The service is on whatever host the web server serves it: requests must come from a page of that host
            self.hosts = None
            self.origins = None
        self.user = current_user()

    def process_request(self, req, resp):
        req.context.ip = client_ip(req, self.settings)
        req.context.user = None
        req.context.role = None
        req.context.session = None
        req.context.csrf = None
        req.context.token = None
        if "//" in req.path or "\\" in req.path:
            raise falcon.HTTPBadRequest(description="invalid path")
        if self.hosts is not None and req.get_header("Host") not in self.hosts:
            raise falcon.HTTPForbidden(description="this server only answers on localhost")
        if self.settings.service and req.scheme != "https":
            raise falcon.HTTPForbidden(description="HTTPS only")
        if req.method in UNSAFE_METHODS:
            origin = req.get_header("Origin")
            if origin is None:
                referer = req.get_header("Referer") or ""
                origin = "/".join(referer.split("/", 3)[:3]) if "://" in referer else None
            # In the service, the origin of the request itself: the host the browser asked the web server for (which
            # it passes on in Host or X-Forwarded-Host), over HTTPS
            origins = self.origins if self.origins is not None else {f"https://{req.forwarded_host}"}
            if origin not in origins:
                raise falcon.HTTPForbidden(description="cross-origin request refused")
        if self.settings.service:
            self.identify_service_user(req)
        else:
            self.identify_personal_user(req, resp)

    def process_resource(self, req, resp, resource, params):
        """Deny by default: only resources marked public can be used without being logged in"""
        if resource is None or getattr(resource, "public", False):
            return
        if req.context.user is None:
            raise falcon.HTTPUnauthorized(description="log in first")
        session = req.context.session
        if (
            session is not None
            and session["must_change_password"]
            and not getattr(resource, "before_password_change", False)
        ):
            raise falcon.HTTPForbidden(description="change your password first")
        if req.method in UNSAFE_METHODS:
            sent = req.get_header(CSRF_HEADER) or ""
            if not hmac.compare_digest(sent, req.context.csrf or ""):
                raise falcon.HTTPForbidden(description="missing or wrong CSRF token: reload the page")

    def identify_personal_user(self, req, resp):
        authorization = req.get_header("Authorization") or ""
        token = authorization[7:] if authorization.startswith("Bearer ") else ""
        if token and hmac.compare_digest(token, self.settings.token):
            req.context.user = self.user
            req.context.role = "admin"
            req.context.csrf = personal_csrf(self.settings.token)

    def identify_service_user(self, req):
        tokens = req.get_cookie_values(session_cookie(self.settings)) or []
        for token in tokens[:3]:
            session = self.accounts.session(token)
            if session is None:
                continue
            req.context.token = token
            req.context.session = session
            if session["stage"] == "full":
                req.context.user = session["user"]
                req.context.role = session["role"]
                req.context.csrf = session["csrf"]
            return

    def process_response(self, req, resp, resource, req_succeeded):
        resp.set_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        resp.set_header("X-Content-Type-Options", "nosniff")
        resp.set_header("X-Frame-Options", "DENY")
        resp.set_header("Referrer-Policy", "same-origin")
        resp.set_header("Cross-Origin-Opener-Policy", "same-origin")
        if req.path.startswith("/api/"):
            resp.set_header("Cache-Control", "no-store")


def session_cookie(settings):
    """The name of the session cookie: with the __Host- prefix when the UI is at the root of its host (cookies of other
    subdomains can't then replace it)"""
    return f"__Host-{SESSION_COOKIE}" if not settings.url_prefix else SESSION_COOKIE


def trusted_cookie(settings):
    """The name of the cookie of a browser trusted for the second factor (same prefix as the session cookie)"""
    return session_cookie(settings).replace(SESSION_COOKIE, TRUSTED_COOKIE)


def trusted_browser_token(req, settings):
    values = req.get_cookie_values(trusted_cookie(settings)) or []
    return values[0] if values else None


def set_trusted_cookie(resp, settings, token):
    resp.set_cookie(
        trusted_cookie(settings),
        token,
        max_age=settings.trusted_browser_days * 86400,
        path=(settings.url_prefix or "") + "/",
        secure=True,
        http_only=True,
        same_site="Strict",
    )


def clear_trusted_cookie(resp, settings):
    resp.unset_cookie(trusted_cookie(settings), path=(settings.url_prefix or "") + "/")


def set_session_cookie(resp, settings, token, max_age=None):
    resp.set_cookie(
        session_cookie(settings),
        token,
        max_age=max_age if max_age is not None else settings.session_max_age,
        path=(settings.url_prefix or "") + "/",
        secure=True,
        http_only=True,
        same_site="Strict",
    )


def clear_session_cookie(resp, settings):
    resp.unset_cookie(session_cookie(settings), path=(settings.url_prefix or "") + "/")
