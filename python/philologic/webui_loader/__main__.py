"""philologic5-webui-loader: a web UI to load PhiloLogic databases and edit their web configs.

philologic5-webui-loader              start it on your own machine, and open it in the browser
philologic5-webui-loader --background   ... and keep it running after the terminal closes
philologic5-webui-loader url          print the URL of the running UI
philologic5-webui-loader stop         stop it
philologic5-webui-loader serve        run it as a service on a distant machine
philologic5-webui-loader users ...    manage the accounts of the service

Its settings are in PhiloLogic's global config (/etc/philologic/philologic5.cfg): webui_loader = False turns it off on
the machine, and webui_loader_* keys set how it runs (see docs/webui_loader.md).
"""

import argparse
import getpass
import html
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import webbrowser

from philologic.webui_loader.settings import (
    DEFAULT_PORT,
    GLOBAL_CONFIG,
    LoaderDisabled,
    SettingsError,
    personal_settings,
    personal_state_dir,
    service_settings,
)

PORTS_TRIED = 20


def server_file(state_dir):
    return os.path.join(state_dir, "server.json")


def running_server(state_dir):
    """The UI already running for this user ({pid, port, url}), or None"""
    try:
        with open(server_file(state_dir), encoding="utf8") as info_file:
            info = json.load(info_file)
        os.kill(info["pid"], 0)
    except (OSError, ValueError, KeyError):
        return None
    try:
        with socket.create_connection(("127.0.0.1", info["port"]), timeout=1):
            return info
    except OSError:
        return None


def free_port(first):
    for port in range(first, first + PORTS_TRIED):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise SystemExit(f"No free port between {first} and {first + PORTS_TRIED - 1}: give one with --port")


def ssh_hint(port):
    """How to reach the UI from your own machine, when started in an SSH session"""
    if not os.environ.get("SSH_CONNECTION"):
        return None
    user = getpass.getuser()
    host = socket.getfqdn()
    return (
        f"You are connected with SSH: to open the UI in your browser, forward its port from your own machine:\n"
        f"    ssh -N -L {port}:localhost:{port} {user}@{host}\n"
        f"or next time, connect and start the UI in one go:\n"
        f"    ssh -L {port}:localhost:{port} {user}@{host} philologic5-webui-loader"
    )


def open_in_browser(url, state_dir):
    """Open the UI through a local page readable only by you, which goes to its URL: the token isn't then on the
    command line of the browser, which other users can see"""
    page = os.path.join(state_dir, "open.html")
    fd = os.open(page, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf8") as page_file:
        page_file.write(
            f'<!doctype html><meta http-equiv="refresh" content="0;url={html.escape(url)}">'
            f"<script>location.replace({json.dumps(url)})</script>"
        )
    webbrowser.open(f"file://{page}")


def announce(url, port, open_browser, state_dir):
    print(f"\nPhiloLogic loader: {url}\n", flush=True)
    hint = ssh_hint(port)
    if hint:
        print(hint + "\n", flush=True)
    elif open_browser:
        open_in_browser(url, state_dir)


def start_personal(args):
    state_dir = args.state_dir or personal_state_dir()
    os.makedirs(state_dir, mode=0o700, exist_ok=True)
    running = running_server(state_dir)
    if running:
        print("The loader is already running.")
        announce(running["url"], running["port"], not args.no_browser, state_dir)
        return
    port = free_port(args.port)
    try:
        settings = personal_settings(port, args.global_config or GLOBAL_CONFIG, args.static_dir, state_dir)
    except SettingsError as error:
        raise SystemExit(str(error)) from error
    settings.token = secrets.token_urlsafe(32)
    url = f"http://localhost:{port}/#/?token={settings.token}"
    if args.background:
        log = open(os.path.join(state_dir, "server.log"), "ab")
        command = [
            sys.executable,
            "-m",
            "philologic.webui_loader",
            "--port",
            str(port),
            "--no-browser",
            "--foreground-child",
        ]
        for option, value in (
            ("--global-config", args.global_config),
            ("--static-dir", args.static_dir),
            ("--state-dir", args.state_dir),
        ):
            if value:
                command += [option, value]
        env = dict(os.environ, PHILOLOGIC_WEBUI_TOKEN=settings.token)
        subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, env=env)
        for _ in range(100):
            if running_server(state_dir):
                break
            time.sleep(0.1)
        announce(url, port, not args.no_browser, state_dir)
        print("It keeps running in the background: stop it with philologic5-webui-loader stop")
        return
    if args.foreground_child:
        settings.token = os.environ.pop("PHILOLOGIC_WEBUI_TOKEN")
        url = f"http://localhost:{port}/#/?token={settings.token}"
    info = {"pid": os.getpid(), "port": port, "url": url}
    fd = os.open(server_file(state_dir), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf8") as info_file:
        json.dump(info, info_file)
    if not args.foreground_child:
        announce(url, port, not args.no_browser, state_dir)
        print("Stop it with Control-C. Loads keep running when it stops.", flush=True)
    from philologic.webui_loader.server.app import run

    try:
        run(settings, errorlog=os.path.join(state_dir, "server.log") if args.foreground_child else "-")
    finally:
        try:
            os.remove(server_file(state_dir))
        except OSError:
            pass


def show_url(args):
    running = running_server(args.state_dir or personal_state_dir())
    if running is None:
        raise SystemExit("The loader isn't running: start it with philologic5-webui-loader")
    print(running["url"])


def stop(args):
    running = running_server(args.state_dir or personal_state_dir())
    if running is None:
        raise SystemExit("The loader isn't running")
    os.kill(running["pid"], signal.SIGTERM)
    print("Stopped (the loads it started keep running).")


def serve(args):
    try:
        settings = service_settings(args.global_config or GLOBAL_CONFIG)
    except LoaderDisabled as error:
        print(error)
        return  # a clean exit: systemd doesn't start it again
    except SettingsError as error:
        raise SystemExit(str(error)) from error
    from philologic.webui_loader.server.app import run

    run(settings, errorlog="-", accesslog="-")


def accounts_of(args):
    from philologic.webui_loader.accounts import Accounts

    try:
        settings = service_settings(args.global_config or GLOBAL_CONFIG, check=False)
    except SettingsError as error:
        raise SystemExit(str(error)) from error
    return Accounts(os.path.join(settings.state_dir, "accounts.sqlite"), settings)


def users(args):
    from philologic.webui_loader.accounts import AccountError

    accounts = accounts_of(args)
    try:
        if args.action == "list":
            for user in accounts.users():
                flags = [user["role"]]
                if not user["totp_enabled"]:
                    flags.append("no second factor yet")
                if user["must_change_password"]:
                    flags.append("temporary password")
                if user["disabled"]:
                    flags.append("disabled")
                grants = f"; granted: {', '.join(user['grants'])}" if user["grants"] else ""
                print(f"{user['name']}: {', '.join(flags)}{grants}")
        elif args.action == "add":
            password = accounts.add_user(args.name, "admin" if args.admin else "loader")
            accounts.audit("user_added", None, "command line", name=args.name)
            print(f"Temporary password of {args.name}, to change at the first login: {password}")
            print("A second factor (an authenticator app) will be set up at the first login.")
        elif args.action == "remove":
            accounts.remove_user(args.name)
            accounts.audit("user_removed", None, "command line", name=args.name)
        elif args.action == "reset-password":
            password = accounts.reset_password(args.name)
            accounts.audit("user_reset_password", None, "command line", name=args.name)
            print(f"Temporary password of {args.name}, to change at the next login: {password}")
        elif args.action == "reset-2fa":
            accounts.reset_totp(args.name)
            accounts.audit("user_reset_totp", None, "command line", name=args.name)
            print(f"{args.name} will set up a new second factor at the next login.")
        elif args.action in ("admin", "loader"):
            accounts.update_user(args.name, role=args.action)
        elif args.action in ("disable", "enable"):
            accounts.update_user(args.name, disabled=args.action == "disable")
        elif args.action in ("grant", "revoke"):
            getattr(accounts, args.action)(args.name, args.database)
    except AccountError as error:
        raise SystemExit(str(error)) from error


def parser():
    parser = argparse.ArgumentParser(
        prog="philologic5-webui-loader", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="first port tried (default %(default)s)")
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    parser.add_argument("--background", action="store_true", help="keep running after the terminal closes")
    parser.add_argument("--foreground-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--global-config", default=None, help=f"PhiloLogic's global config (default {GLOBAL_CONFIG})")
    parser.add_argument(
        "--static-dir", default=None, help="the built web client (default /var/lib/philologic5/webui_loader/dist)"
    )
    parser.add_argument("--state-dir", default=None, help=argparse.SUPPRESS)
    # --global-config can also come after a command, or an action of users
    config = argparse.ArgumentParser(add_help=False)
    config.add_argument("--global-config", default=argparse.SUPPRESS, help="PhiloLogic's global config")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("url", help="print the URL of the running UI")
    commands.add_parser("stop", help="stop the running UI")
    for name, help_text in (("serve", "run the service"), ("users", "manage the accounts of the service")):
        command = commands.add_parser(name, help=help_text, parents=[config])
        if name == "users":
            actions = command.add_subparsers(dest="action", required=True)
            actions.add_parser("list", help="list the accounts", parents=[config])
            add = actions.add_parser("add", help="add an account, with a temporary password", parents=[config])
            add.add_argument("name")
            add.add_argument("--admin", action="store_true", help="an admin, who manages accounts and all databases")
            for action, help_text in (
                ("remove", "remove an account"),
                ("reset-password", "give a new temporary password"),
                ("reset-2fa", "remove the second factor, to set up again at the next login"),
                ("admin", "make an admin"),
                ("loader", "make a loader (not an admin)"),
                ("disable", "disable an account"),
                ("enable", "enable a disabled account"),
            ):
                actions.add_parser(action, help=help_text, parents=[config]).add_argument("name")
            for action, help_text in (("grant", "let a loader edit a database"), ("revoke", "take it back")):
                grant = actions.add_parser(action, help=help_text, parents=[config])
                grant.add_argument("name")
                grant.add_argument("database")
    return parser


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "serve":
        serve(args)
    elif args.command == "users":
        users(args)
    elif args.command == "url":
        show_url(args)
    elif args.command == "stop":
        stop(args)
    else:
        start_personal(args)


if __name__ == "__main__":
    main()
