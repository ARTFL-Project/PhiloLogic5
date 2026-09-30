"""The JSON API of philologic5-webui-loader, and the files of its web client. Permissions are checked here: on your
own machine you can do everything your account can; in the service, admins can do everything, and loaders can load
new databases, and edit or load again the databases they loaded or were granted, and only see and cancel their own
loads. Files are always checked against the directories which can be loaded from (see files.py)."""

import importlib.metadata
import json
import mimetypes
import os
import shutil
import socket

import falcon

from philologic.webui_loader import (
    databases,
    files,
    load_config_io,
    load_schema,
    preflight,
    previews,
    web_config_io,
)
from philologic.webui_loader.accounts import PARTIAL_SESSION_TIME, AccountError
from philologic.webui_loader.files import FilesError
from philologic.webui_loader.jobs import JobError, load_arguments
from philologic.webui_loader.server.security import (
    clear_session_cookie,
    clear_trusted_cookie,
    set_session_cookie,
    set_trusted_cookie,
    trusted_browser_token,
)
from philologic.webui_loader.uploads import MAX_CHUNK, UploadError, Uploads
from philologic.webui_loader.web_config_io import WebConfigError

MAX_BODY = 4 * 1024**2


def body(req):
    if req.content_length and req.content_length > MAX_BODY:
        raise falcon.HTTPContentTooLarge(description="request too large")
    media = req.get_media(default_when_empty={})
    if not isinstance(media, dict):
        raise falcon.HTTPBadRequest(description="a JSON object is expected")
    return media


def version():
    try:
        return importlib.metadata.version("philologic")
    except importlib.metadata.PackageNotFoundError:
        return None


class Api:
    """What the resources share: settings, accounts (service), jobs"""

    def __init__(self, settings, accounts, jobs):
        self.settings = settings
        self.accounts = accounts
        self.jobs = jobs

    def roots(self, req):
        """Directories this user can load files from: all on your own machine; in the service, the allowed roots and
        the user's upload area"""
        if not self.settings.service:
            return []
        roots = list(self.settings.allowed_roots)
        upload = self.upload_dir(req.context.user)
        if upload:
            roots.append(os.path.realpath(upload))
        return roots

    def upload_dir(self, user):
        if not self.settings.upload_dir or not user:
            return None
        return os.path.join(self.settings.upload_dir, user)

    def is_admin(self, req):
        return req.context.role == "admin"

    def require_admin(self, req):
        if not self.is_admin(req):
            raise falcon.HTTPForbidden(description="only admins can do this")

    def may_edit(self, req, dbname):
        if not self.settings.service:
            return True
        return self.accounts.may_edit(req.context.user, req.context.role, dbname)

    def audit(self, req, action, **detail):
        if self.accounts is not None:
            self.accounts.audit(action, req.context.user, req.context.ip, **detail)

    def database(self, name):
        db_path = databases.database_path(self.settings, name)
        if db_path is None:
            raise falcon.HTTPNotFound(description=f"no database {name}")
        return db_path

    def job_status(self, req, job_id, details=True):
        try:
            status = self.jobs.status(job_id, details)
        except JobError as error:
            raise falcon.HTTPNotFound(description=str(error)) from error
        if self.settings.service and not self.is_admin(req) and status["user"] != req.context.user:
            raise falcon.HTTPNotFound(description="no such load")
        return status


class SessionResource:
    public = True

    def __init__(self, api):
        self.api = api

    def on_get(self, req, resp):
        settings = self.api.settings
        session = req.context.session
        result = {"mode": settings.mode, "authenticated": req.context.user is not None}
        if settings.service:
            result["trusted_browser_days"] = settings.trusted_browser_days
        if req.context.user is None:
            if session is not None:
                result["stage"] = session["stage"]
            resp.media = result
            return
        result.update(
            {
                "user": req.context.user,
                "role": req.context.role,
                "csrf": req.context.csrf,
                "must_change_password": bool(session and session["must_change_password"]),
                "hostname": socket.gethostname(),
                "database_root": settings.database_root,
                "url_root": settings.url_root,
                "version": version(),
                "uploads": bool(settings.upload_dir),
                "browser_trusted": bool(
                    settings.service
                    and self.api.accounts.browser_trusted(req.context.user, trusted_browser_token(req, settings))
                ),
            }
        )
        resp.media = result


class LoginResource:
    public = True

    def __init__(self, api):
        self.api = api

    def service_only(self):
        if not self.api.settings.service:
            raise falcon.HTTPNotFound()

    def on_post(self, req, resp):
        self.service_only()
        data = body(req)
        settings = self.api.settings
        try:
            token, step = self.api.accounts.login(
                data.get("username"), data.get("password"), req.context.ip, trusted_browser_token(req, settings)
            )
        except AccountError as error:
            raise falcon.HTTPUnauthorized(description=str(error)) from error
        set_session_cookie(resp, settings, token, max_age=None if step == "done" else PARTIAL_SESSION_TIME)
        resp.media = {"next": step}


class TotpResource(LoginResource):
    def on_post(self, req, resp):
        self.service_only()
        data = body(req)
        try:
            token, browser = self.api.accounts.verify_second_factor(
                req.context.token, data.get("code"), req.context.ip, trust=data.get("trust") is True
            )
        except AccountError as error:
            raise falcon.HTTPUnauthorized(description=str(error)) from error
        set_session_cookie(resp, self.api.settings, token)
        if browser:
            set_trusted_cookie(resp, self.api.settings, browser)
        resp.media = {"ok": True}


class TotpSetupResource(LoginResource):
    def on_get(self, req, resp):
        self.service_only()
        try:
            resp.media = self.api.accounts.totp_setup(req.context.token)
        except AccountError as error:
            raise falcon.HTTPUnauthorized(description=str(error)) from error

    def on_post(self, req, resp):
        self.service_only()
        data = body(req)
        try:
            token, codes, browser = self.api.accounts.confirm_totp_setup(
                req.context.token, data.get("code"), req.context.ip, trust=data.get("trust") is True
            )
        except AccountError as error:
            raise falcon.HTTPUnauthorized(description=str(error)) from error
        set_session_cookie(resp, self.api.settings, token)
        if browser:
            set_trusted_cookie(resp, self.api.settings, browser)
        resp.media = {"recovery_codes": codes}


class LogoutResource(LoginResource):
    def on_post(self, req, resp):
        if self.api.settings.service:
            if req.context.token:
                self.api.accounts.logout(req.context.token)
                self.api.audit(req, "logout")
            clear_session_cookie(resp, self.api.settings)
        resp.media = {"ok": True}


class PasswordResource(LoginResource):
    public = False
    before_password_change = True

    def on_post(self, req, resp):
        self.service_only()
        data = body(req)
        try:
            self.api.accounts.change_password(
                req.context.user,
                data.get("current"),
                data.get("new"),
                keep_session=req.context.token,
                keep_browser=trusted_browser_token(req, self.api.settings),
            )
        except AccountError as error:
            raise falcon.HTTPBadRequest(description=str(error)) from error
        self.api.audit(req, "password_changed")
        resp.media = {"ok": True}


class TrustedBrowsersResource(LoginResource):
    """Forget the browsers trusted for the second factor of the user, this one included"""

    public = False
    before_password_change = True

    def on_delete(self, req, resp):
        self.service_only()
        self.api.accounts.forget_browsers(req.context.user)
        clear_trusted_cookie(resp, self.api.settings)
        self.api.audit(req, "trusted_browsers_forgotten")
        resp.media = {"ok": True}


class SystemResource:
    def __init__(self, api):
        self.api = api

    def on_get(self, req, resp):
        settings = self.api.settings
        try:
            free = shutil.disk_usage(settings.database_root).free
        except OSError:
            free = None
        resp.media = {
            "mode": settings.mode,
            "database_root": settings.database_root,
            "url_root": settings.url_root,
            "cpu_count": os.cpu_count(),
            "memory_available": preflight.memory_available(),
            "disk_free": free,
            "spacy_models": preflight.installed_spacy_models(),
            "roots": self.api.roots(req),
            # Where the file browser starts on your own machine: the first of webui_loader_allowed_roots, if set
            "home": (
                (settings.allowed_roots[0] if settings.allowed_roots else os.path.expanduser("~"))
                if not settings.service
                else None
            ),
            "max_cores": settings.max_cores,
            "upload_dir": self.api.upload_dir(req.context.user) if settings.service else None,
            "version": version(),
        }


class SchemaResource:
    def __init__(self, schema):
        self.schema = schema

    def on_get(self, req, resp):
        resp.media = self.schema()


class DatabasesResource:
    def __init__(self, api):
        self.api = api

    def on_get(self, req, resp):
        settings = self.api.settings
        owners = self.api.accounts.owners() if self.api.accounts else {}
        result = []
        for database in databases.list_databases(settings):
            may_edit = self.api.may_edit(req, database["name"])
            database["may_edit"] = may_edit
            database["loaded_by"] = owners.get(database["name"])
            lock = self.api.jobs.running_load(database["name"])
            database["loading"] = {"user": lock.get("user"), "job": lock.get("job")} if lock else None
            result.append(database)
        resp.media = {"databases": result}


class DatabaseResource:
    def __init__(self, api):
        self.api = api

    def on_get(self, req, resp, name):
        db_path = self.api.database(name)
        database = databases.info(self.api.settings, name)
        database["may_edit"] = self.api.may_edit(req, name)
        config_path = load_config_io.database_config_path(db_path)
        database["load_config"] = None
        if os.path.isfile(config_path):
            try:
                database["load_config"] = load_config_io.read_file(config_path)
            except (SyntaxError, OSError) as error:
                database["load_config"] = {"error": str(error)}
        resp.media = database


class WebConfigResource:
    def __init__(self, api):
        self.api = api

    def on_get(self, req, resp, name):
        db_path = self.api.database(name)
        config = web_config_io.read(db_path, service=self.api.settings.service)
        if not self.api.may_edit(req, name):
            config["writable"] = False
            config["reason"] = "you are not allowed to edit this database: ask an admin"
        resp.media = config

    def on_put(self, req, resp, name):
        db_path = self.api.database(name)
        if not self.api.may_edit(req, name):
            raise falcon.HTTPForbidden(description="you are not allowed to edit this database")
        data = body(req)
        changes = data.get("changes")
        if not isinstance(changes, dict):
            raise falcon.HTTPBadRequest(description="changes are expected")
        try:
            backup = web_config_io.save(
                db_path,
                changes,
                data.get("hash"),
                service=self.api.settings.service,
                restricted=self.api.settings.service and not self.api.is_admin(req),
            )
        except WebConfigError as error:
            raise falcon.HTTPConflict(description=str(error)) from error
        if backup:
            self.api.audit(req, "web_config_saved", database=name, changes=audit_changes(changes))
        config = web_config_io.read(db_path, service=self.api.settings.service)
        config["backup"] = backup
        resp.media = config


class LoadConfigResource:
    """Read a load config file, as the load pages start from it"""

    def __init__(self, api):
        self.api = api

    def on_post(self, req, resp):
        path = body(req).get("path") or ""
        try:
            files.check_allowed(path, self.api.roots(req))
            resp.media = load_config_io.read_file(path)
        except FilesError as error:
            raise falcon.HTTPBadRequest(description=str(error)) from error
        except OSError as error:
            raise falcon.HTTPBadRequest(description=f"{path} can't be read") from error
        except SyntaxError as error:
            raise falcon.HTTPBadRequest(description=f"{path} isn't valid Python: {error}") from error


class FilesResource:
    def __init__(self, api):
        self.api = api

    def on_get(self, req, resp):
        roots = self.api.roots(req)
        path = req.get_param("path") or (roots[0] if roots else os.path.expanduser("~"))
        try:
            listing = files.list_directory(os.path.abspath(path), roots, req.get_param("pattern") or "*")
        except FilesError as error:
            raise falcon.HTTPBadRequest(description=str(error)) from error
        listing["roots"] = roots
        resp.media = listing


def audit_changes(changes, limit=20000):
    """The changes of a web config for the audit log (their JSON, shortened if long)"""
    text = json.dumps(changes, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "…"


def load_request(data):
    request = data.get("request")
    if not isinstance(request, dict):
        raise falcon.HTTPBadRequest(description="a load request is expected")
    return request


class PreflightResource:
    def __init__(self, api):
        self.api = api

    def prepare(self, req, request):
        api = self.api
        return preflight.prepare(
            api.settings,
            api.jobs,
            request,
            api.roots(req),
            user=req.context.user,
            can_replace=lambda dbname: api.may_edit(req, dbname),
        )

    def on_post(self, req, resp):
        request = load_request(body(req))
        plan = self.prepare(req, request)
        result = plan.to_dict()
        result["command"] = "philoload5 " + " ".join(
            load_arguments(
                request.get("dbname") or "",
                "load_config.py",
                "files.txt",
                request.get("cores") or 4,
                request.get("header") or "tei",
                request.get("file_type") or "xml",
                request.get("bibliography"),
                bool(request.get("overwrite")),
            )
        )
        resp.media = result


class JobsResource(PreflightResource):
    def on_get(self, req, resp):
        user = None if (not self.api.settings.service or self.api.is_admin(req)) else req.context.user
        resp.media = {"jobs": self.api.jobs.list(user)}

    def on_post(self, req, resp):
        request = load_request(body(req))
        plan = self.prepare(req, request)
        if plan.errors:
            resp.status = falcon.HTTP_422
            resp.media = plan.to_dict()
            return
        new_database = databases.database_path(self.api.settings, request["dbname"]) is None
        try:
            job_id = self.api.jobs.launch(
                request["dbname"],
                plan.config_text,
                plan.files,
                request.get("cores") or 4,
                header=request.get("header") or "tei",
                file_type=request.get("file_type") or "xml",
                bibliography=request.get("bibliography"),
                overwrite=bool(request.get("overwrite")),
                cwd=plan.cwd,
                user=req.context.user,
                source=request.get("files"),
            )
        except JobError as error:
            raise falcon.HTTPConflict(description=str(error)) from error
        if self.api.accounts is not None:
            self.api.accounts.set_owner(request["dbname"], req.context.user, new_database)
        self.api.audit(req, "load_started", database=request["dbname"], job=job_id, files=len(plan.files))
        resp.status = falcon.HTTP_201
        resp.media = {"id": job_id}


class JobResource:
    def __init__(self, api):
        self.api = api

    def on_get(self, req, resp, job_id):
        resp.media = self.api.job_status(req, job_id)


class JobLogResource(JobResource):
    def on_get(self, req, resp, job_id):
        self.api.job_status(req, job_id, details=False)
        offset = req.get_param_as_int("offset", min_value=0) or 0
        resp.media = self.api.jobs.log(job_id, offset)


class JobCancelResource(JobResource):
    def on_post(self, req, resp, job_id):
        self.api.job_status(req, job_id, details=False)
        try:
            self.api.jobs.cancel(job_id)
        except JobError as error:
            raise falcon.HTTPConflict(description=str(error)) from error
        self.api.audit(req, "load_cancelled", job=job_id)
        resp.media = {"ok": True}


class JobConfigResource(JobResource):
    def on_get(self, req, resp, job_id):
        self.api.job_status(req, job_id, details=False)
        resp.media = {"config": self.api.jobs.load_config(job_id)}


class PreviewResource:
    def __init__(self, api):
        self.api = api

    def on_post(self, req, resp, kind):
        data = body(req)
        try:
            paths = files.resolve(data.get("files") or {}, self.api.roots(req))
        except FilesError as error:
            raise falcon.HTTPBadRequest(description=str(error)) from error
        if not paths:
            raise falcon.HTTPBadRequest(description="no files")
        options = data.get("options") or {}
        errors = load_config_io.validate(options)
        if errors:
            raise falcon.HTTPBadRequest(description="; ".join(f"{key}: {error}" for key, error in errors.items()))
        try:
            count = min(max(int(data.get("count") or 10), 1), 50)
        except (TypeError, ValueError) as error:
            raise falcon.HTTPBadRequest(description="count must be a number") from error
        if kind == "header":
            job = (previews.header_preview, paths, options, count)
        elif kind == "tags":
            job = (previews.tag_census, paths, options, count)
        elif kind == "tokens":
            name = data.get("file")
            path = next((path for path in paths if os.path.basename(path) == name), paths[0])
            job = (previews.tokens_preview, path, options)
        else:
            raise falcon.HTTPNotFound()
        try:
            resp.media = previews.run_isolated(*job)
        except previews.PreviewTimeout as error:
            raise falcon.HTTPServiceUnavailable(description=str(error)) from error
        except (OSError, UnicodeDecodeError, ValueError) as error:
            raise falcon.HTTPBadRequest(description=str(error)) from error


class UploadsResource:
    """Uploads of the files of loads, in the service"""

    def __init__(self, api):
        self.api = api

    def uploads(self, req):
        if not self.api.settings.upload_dir:
            raise falcon.HTTPNotFound()
        return Uploads(self.api.settings, req.context.user)

    def on_get(self, req, resp):
        resp.media = self.uploads(req).list()

    def on_post(self, req, resp):
        data = body(req)
        try:
            meta = self.uploads(req).create(
                data.get("name"), data.get("kind"), data.get("size"), data.get("file_count") or 1
            )
        except UploadError as error:
            raise falcon.HTTPBadRequest(description=str(error)) from error
        self.api.audit(req, "upload_started", name=meta["name"], size=meta["size"], upload=meta["id"])
        resp.status = falcon.HTTP_201
        resp.media = meta


class UploadResource(UploadsResource):
    def on_get(self, req, resp, upload_id):
        uploads = self.uploads(req)
        try:
            status = uploads.status(upload_id)
            if status["state"] == "receiving":
                status["files"] = uploads.received(upload_id)
        except UploadError as error:
            raise falcon.HTTPNotFound(description=str(error)) from error
        resp.media = status

    def on_delete(self, req, resp, upload_id):
        try:
            self.uploads(req).delete(upload_id)
        except UploadError as error:
            raise falcon.HTTPNotFound(description=str(error)) from error
        self.api.audit(req, "upload_deleted", upload=upload_id)
        resp.media = {"ok": True}


class UploadContentResource(UploadsResource):
    def on_put(self, req, resp, upload_id):
        length = req.content_length or 0
        if length > MAX_CHUNK:
            raise falcon.HTTPContentTooLarge(description=f"chunks are at most {MAX_CHUNK} bytes")
        offset = req.get_param_as_int("offset", min_value=0, required=True)
        try:
            received = self.uploads(req).write_chunk(
                upload_id, req.get_param("path") or "", offset, req.bounded_stream, length
            )
        except UploadError as error:
            raise falcon.HTTPConflict(description=str(error)) from error
        resp.media = {"received": received}


class UploadCompleteResource(UploadsResource):
    def on_post(self, req, resp, upload_id):
        try:
            status = self.uploads(req).complete(upload_id)
        except UploadError as error:
            raise falcon.HTTPConflict(description=str(error)) from error
        self.api.audit(req, "upload_complete", upload=upload_id, files=status["file_count"], size=status["size"])
        resp.media = status


class UsersResource:
    """Accounts of the service, for admins"""

    def __init__(self, api):
        self.api = api

    def check(self, req):
        if not self.api.settings.service:
            raise falcon.HTTPNotFound()
        self.api.require_admin(req)

    def on_get(self, req, resp):
        self.check(req)
        resp.media = {"users": self.api.accounts.users()}

    def on_post(self, req, resp):
        self.check(req)
        data = body(req)
        try:
            password = self.api.accounts.add_user(data.get("name"), data.get("role") or "loader")
        except AccountError as error:
            raise falcon.HTTPBadRequest(description=str(error)) from error
        self.api.audit(req, "user_added", name=data.get("name"), role=data.get("role") or "loader")
        resp.status = falcon.HTTP_201
        resp.media = {"temporary_password": password}


class UserResource(UsersResource):
    def on_patch(self, req, resp, name):
        self.check(req)
        data = body(req)
        try:
            self.api.accounts.update_user(name, role=data.get("role"), disabled=data.get("disabled"))
        except AccountError as error:
            raise falcon.HTTPBadRequest(description=str(error)) from error
        self.api.audit(req, "user_changed", name=name, role=data.get("role"), disabled=data.get("disabled"))
        resp.media = {"ok": True}

    def on_delete(self, req, resp, name):
        self.check(req)
        if name == req.context.user:
            raise falcon.HTTPBadRequest(description="you can't remove your own account")
        try:
            self.api.accounts.remove_user(name)
        except AccountError as error:
            raise falcon.HTTPBadRequest(description=str(error)) from error
        self.api.audit(req, "user_removed", name=name)
        resp.media = {"ok": True}


class UserActionResource(UsersResource):
    def on_post(self, req, resp, name, action):
        self.check(req)
        data = body(req)
        accounts = self.api.accounts
        try:
            if action == "reset_password":
                resp.media = {"temporary_password": accounts.reset_password(name)}
            elif action == "reset_totp":
                accounts.reset_totp(name)
                resp.media = {"ok": True}
            elif action == "grant":
                dbname = data.get("database") or ""
                if not databases.valid_name(dbname):
                    raise falcon.HTTPBadRequest(description="not a database name")
                accounts.grant(name, dbname)
                resp.media = {"ok": True}
            elif action == "revoke":
                accounts.revoke(name, data.get("database") or "")
                resp.media = {"ok": True}
            else:
                raise falcon.HTTPNotFound()
        except AccountError as error:
            raise falcon.HTTPBadRequest(description=str(error)) from error
        self.api.audit(req, f"user_{action}", name=name, database=data.get("database"))


class AuditResource(UsersResource):
    def on_get(self, req, resp):
        self.check(req)
        resp.media = {"entries": self.api.accounts.audit_log()}


class StaticResource:
    """The web client: index.html at /, and its files (the build of webui_loader/, in static_dir)"""

    public = True

    def __init__(self, settings):
        self.directory = os.path.realpath(settings.static_dir)

    def on_get(self, req, resp, path="index.html"):
        file_path = os.path.realpath(os.path.join(self.directory, path))
        if not file_path.startswith(self.directory + os.sep) or not os.path.isfile(file_path):
            if path == "index.html":
                resp.content_type = falcon.MEDIA_HTML
                resp.text = (
                    "<!doctype html><title>PhiloLogic loader</title><p>The web client of philologic5-webui-loader "
                    f"is not built: run <code>npm run build</code> in webui_loader/, or give --static-dir.</p>"
                )
                return
            raise falcon.HTTPNotFound()
        resp.content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        accept_encoding = req.get_header("Accept-Encoding") or ""
        if "br" in accept_encoding and os.path.isfile(file_path + ".br"):
            file_path += ".br"
            resp.set_header("Content-Encoding", "br")
            resp.set_header("Vary", "Accept-Encoding")
        if path == "index.html":
            resp.set_header("Cache-Control", "no-cache")
        else:
            resp.set_header("Cache-Control", "public, max-age=31536000, immutable")
        with open(file_path, "rb") as static_file:
            resp.data = static_file.read()
