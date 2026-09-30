"""The databases in database_root, for the databases page of philologic5-webui-loader"""

import os
import re
import stat

from philologic.webui_loader.pyconfig import ConfigFile
from philologic.webui_loader.web_config_io import group_name, user_name, write_permission

DBNAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")

_cache = {}  # path: (modification times, value)


def valid_name(dbname):
    return bool(DBNAME.match(dbname or "")) and ".." not in dbname


def cached(key, paths, compute):
    """compute(), again only when one of the paths has changed"""
    stamps = []
    for path in paths:
        try:
            info = os.stat(path)
            stamps.append((info.st_mtime_ns, info.st_size))
        except OSError:
            stamps.append(None)
    entry = _cache.get(key)
    if entry is not None and entry[0] == stamps:
        return entry[1]
    value = compute()
    _cache[key] = (stamps, value)
    return value


def title(db_path):
    """The dbname of the web config: the title of the database"""

    def compute():
        try:
            with open(os.path.join(db_path, "data", "web_config.cfg"), encoding="utf8") as config_file:
                value = ConfigFile(config_file.read()).values.get("dbname")
        except (OSError, SyntaxError):
            return None
        return value if isinstance(value, str) else None

    return cached(("title", db_path), [os.path.join(db_path, "data", "web_config.cfg")], compute)


def document_count(db_path):
    text_dir = os.path.join(db_path, "data", "TEXT")

    def compute():
        try:
            return sum(1 for entry in os.scandir(text_dir) if entry.is_file())
        except OSError:
            return None

    return cached(("documents", db_path), [text_dir], compute)


def can_replace(settings, db_path):
    """(can, reason): whether this process can delete a database, to load it again"""
    for path in (settings.database_root, db_path, os.path.join(db_path, "data")):
        if os.path.exists(path) and not os.access(path, os.W_OK | os.X_OK):
            info = os.stat(path)
            return (
                False,
                f"{path} belongs to {user_name(info.st_uid)} and can't be written by {user_name(os.geteuid())}",
            )
    return True, ""


def info(settings, name):
    db_path = os.path.join(settings.database_root, name)
    details = os.stat(db_path)
    db_locals = os.path.join(db_path, "data", "db.locals.py")
    writable, reason = write_permission(db_path)
    replaceable, replace_reason = can_replace(settings, db_path)
    return {
        "name": name,
        "title": title(db_path),
        "path": db_path,
        "url": settings.url_root.rstrip("/") + "/" + name,
        "owner": user_name(details.st_uid),
        "group": group_name(details.st_gid),
        "group_writable": bool(details.st_mode & stat.S_IWGRP),
        "loaded": os.path.getmtime(db_locals) if os.path.exists(db_locals) else None,
        "documents": document_count(db_path),
        "has_load_config": os.path.isfile(os.path.join(db_path, "data", "load_config.py")),
        "web_config_writable": writable,
        "web_config_reason": reason,
        "replaceable": replaceable,
        "replace_reason": replace_reason,
    }


def list_databases(settings):
    databases = []
    try:
        entries = sorted(os.scandir(settings.database_root), key=lambda entry: entry.name.lower())
    except OSError:
        return []
    for entry in entries:
        if entry.name.startswith(".") or not entry.is_dir():
            continue
        if not os.path.isdir(os.path.join(entry.path, "data")):
            continue  # not a database
        try:
            databases.append(info(settings, entry.name))
        except OSError:
            continue
    return databases


def database_path(settings, name):
    """Path of an existing database, or None"""
    if not valid_name(name):
        return None
    path = os.path.join(settings.database_root, name)
    return path if os.path.isdir(os.path.join(path, "data")) else None
