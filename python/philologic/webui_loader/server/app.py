"""The server of philologic5-webui-loader: its Falcon application, run by gunicorn in one process with threads (the
loads themselves run apart, see jobs.py)."""

import os

import falcon
import gunicorn.app.base

from philologic.webui_loader.accounts import Accounts
from philologic.webui_loader.jobs import Jobs
from philologic.webui_loader import load_schema, web_config_io
from philologic.webui_loader.server import resources
from philologic.webui_loader.server.security import SecurityMiddleware, StripPrefix


def error_serializer(req, resp, exception):
    resp.content_type = falcon.MEDIA_JSON
    resp.media = {"title": exception.title, "description": exception.description}


def create_app(settings, accounts=None):
    """The WSGI application. accounts: the accounts of the service (from its state directory if not given)"""
    os.makedirs(settings.state_dir, mode=0o700, exist_ok=True)
    if settings.service and accounts is None:
        accounts = Accounts(os.path.join(settings.state_dir, "accounts.sqlite"), settings)
    api = resources.Api(settings, accounts, Jobs(settings))
    app = falcon.App(middleware=[SecurityMiddleware(settings, accounts)])
    app.set_error_serializer(error_serializer)
    app.req_options.auto_parse_qs_csv = False
    routes = {
        "/api/session": resources.SessionResource(api),
        "/api/login": resources.LoginResource(api),
        "/api/login/totp": resources.TotpResource(api),
        "/api/login/totp_setup": resources.TotpSetupResource(api),
        "/api/logout": resources.LogoutResource(api),
        "/api/account/password": resources.PasswordResource(api),
        "/api/account/trusted_browsers": resources.TrustedBrowsersResource(api),
        "/api/system": resources.SystemResource(api),
        "/api/load_options": resources.SchemaResource(load_schema.schema),
        "/api/web_config_options": resources.SchemaResource(web_config_io.schema),
        "/api/databases": resources.DatabasesResource(api),
        "/api/databases/{name}": resources.DatabaseResource(api),
        "/api/databases/{name}/web_config": resources.WebConfigResource(api),
        "/api/load_config": resources.LoadConfigResource(api),
        "/api/files": resources.FilesResource(api),
        "/api/preflight": resources.PreflightResource(api),
        "/api/jobs": resources.JobsResource(api),
        "/api/jobs/{job_id}": resources.JobResource(api),
        "/api/jobs/{job_id}/log": resources.JobLogResource(api),
        "/api/jobs/{job_id}/cancel": resources.JobCancelResource(api),
        "/api/jobs/{job_id}/load_config": resources.JobConfigResource(api),
        "/api/previews/{kind}": resources.PreviewResource(api),
        "/api/uploads": resources.UploadsResource(api),
        "/api/uploads/{upload_id}": resources.UploadResource(api),
        "/api/uploads/{upload_id}/content": resources.UploadContentResource(api),
        "/api/uploads/{upload_id}/complete": resources.UploadCompleteResource(api),
        "/api/users": resources.UsersResource(api),
        "/api/users/{name}": resources.UserResource(api),
        "/api/users/{name}/{action}": resources.UserActionResource(api),
        "/api/audit": resources.AuditResource(api),
    }
    for route, resource in routes.items():
        app.add_route(route, resource)
    static = resources.StaticResource(settings)
    app.add_route("/", static)
    app.add_route("/{path:path}", static)
    return StripPrefix(app, settings.url_prefix)


class Server(gunicorn.app.base.BaseApplication):
    """gunicorn running the application, configured from the settings rather than from its command line"""

    def __init__(self, application, options):
        self.application = application
        self.options = options
        super().__init__()

    def load_config(self):
        for key, value in self.options.items():
            if key in self.cfg.settings and value is not None:
                self.cfg.set(key, value)

    def load(self):
        return self.application


def gunicorn_options(settings, errorlog="-", accesslog=None):
    return {
        "bind": [settings.bind],
        "workers": 1,  # jobs, sessions and caches are shared by its threads
        "worker_class": "gthread",
        "threads": settings.threads,
        "timeout": 120,
        "graceful_timeout": 10,
        "errorlog": errorlog,
        "accesslog": accesslog,
        "proc_name": "philologic5-webui-loader",
        "control_socket_disable": True,
        # Only the web server of the settings is trusted for X-Forwarded-Proto (nothing on your own machine)
        "forwarded_allow_ips": ",".join(settings.forwarded_allow_ips) if settings.forwarded_allow_ips else "",
        "secure_scheme_headers": {"X-FORWARDED-PROTO": "https"},
    }


def run(settings, errorlog="-", accesslog=None):
    Server(create_app(settings), gunicorn_options(settings, errorlog, accesslog)).run()
