"""Where a database's hitlists go: the only files written at request time, so databases can be read-only.

In <hitlist_dir>/<database name>/<load>/, created when first needed. hitlist_dir can be set in the database's
db.locals.py, for that database, or in the global config (philologic5.cfg), for all of them; the database's own setting
comes first. Without either, it is DEFAULT_ROOT. install.sh creates the global one, for the web server's user. Every
file cached along with a hitlist (.done and .terms flags, sorted copies, KWIC sort caches, collocation caches...) goes
in the same directory.

Some of those caches are named after the query alone, not the database, so each database needs its own directory:
databases with the same name (under different database roots) must not share a hitlist_dir. And each load of a
database has its own (<load>, named after its db.locals.py, which every load writes anew), so that nothing cached for
the database a reload replaced is used for the new one: the cleanup removes the old directories.
"""

import os
import sys
from functools import cache

from philologic.Config import DB_LOCALS_DEFAULTS, DB_LOCALS_HEADER, Config
from philologic.utils import load_module

DEFAULT_ROOT = "/Library/Caches/philologic5/hitlists" if sys.platform == "darwin" else "/var/cache/philologic5/hitlists"


@cache
def global_hitlist_root():
    """The hitlist_dir set in the global config, or DEFAULT_ROOT."""
    config_path = os.environ.get("PHILOLOGIC_CONFIG", "/etc/philologic/philologic5.cfg")
    root = getattr(load_module("philologic5", config_path), "hitlist_dir", None) if os.path.isfile(config_path) else None
    return os.path.abspath(root or DEFAULT_ROOT)


def database_hitlist_dir(data_path, db_locals=None):
    """The directory of the hitlists of every load of the database whose data/ directory is data_path (as DB.path).
    db_locals is the database's db.locals.py, if already loaded (as DB.locals)."""
    if db_locals is None:
        db_locals = Config(os.path.join(data_path, "db.locals.py"), DB_LOCALS_DEFAULTS, DB_LOCALS_HEADER)
    root = os.path.abspath(db_locals["hitlist_dir"]) if db_locals["hitlist_dir"] else global_hitlist_root()
    return os.path.join(root, os.path.basename(os.path.dirname(os.path.normpath(data_path))))


def load_name(data_path):
    """The name of the directory of the hitlists of the database's current load: the modification time and inode of
    its db.locals.py, or None without one."""
    try:
        stat = os.stat(os.path.join(data_path, "db.locals.py"))
    except FileNotFoundError:
        return None
    return f"load-{stat.st_mtime_ns}-{stat.st_ino}"


def get_hitlist_dir(data_path, db_locals=None, create=True):
    """The directory for the hitlists of the current load of the database whose data/ directory is data_path (as
    DB.path), created unless create is False. db_locals is the database's db.locals.py, if already loaded (as
    DB.locals)."""
    path = database_hitlist_dir(data_path, db_locals)
    name = load_name(data_path)
    if name is not None:
        path = os.path.join(path, name)
    if create:
        os.makedirs(path, exist_ok=True)
    return path
