"""Where a database's hitlists go: the only files written at request time.

By default, in the database's own data/hitlists/, which the loader creates. When hitlist_dir is set, in
<hitlist_dir>/<database name>/ instead, created on first use, so that databases can be read-only. It can be set in
the database's db.locals.py, for that database, or in the global config (philologic5.cfg), for all of them; the
database's own setting comes first. Every file cached along with a hitlist (.done and .terms flags, sorted copies,
KWIC sort caches, collocation caches...) goes in the same directory.

Some of those caches are named after the query alone, not the database, so each database needs its own directory:
databases with the same name (under different database roots) must not share a hitlist_dir.
"""

import os
from functools import cache

from philologic.Config import DB_LOCALS_DEFAULTS, DB_LOCALS_HEADER, Config
from philologic.utils import load_module


def default_hitlist_dir(data_path):
    """The database's own hitlist directory, data/hitlists/. data_path is the database's data/ directory."""
    return os.path.join(data_path, "hitlists")


@cache
def _configured_root():
    """The hitlist_dir set in the global config, or None if hitlists stay in each database."""
    config_path = os.environ.get("PHILOLOGIC_CONFIG", "/etc/philologic/philologic5.cfg")
    if not os.path.isfile(config_path):
        return None
    root = getattr(load_module("philologic5", config_path), "hitlist_dir", None)
    return os.path.abspath(root) if root else None


def get_hitlist_dir(data_path, db_locals=None):
    """The directory for the hitlists of the database whose data/ directory is data_path (as DB.path).
    db_locals is the database's db.locals.py, if already loaded (as DB.locals)."""
    if db_locals is None:
        db_locals = Config(os.path.join(data_path, "db.locals.py"), DB_LOCALS_DEFAULTS, DB_LOCALS_HEADER)
    root = os.path.abspath(db_locals["hitlist_dir"]) if db_locals["hitlist_dir"] else _configured_root()
    if root is None:
        return default_hitlist_dir(data_path)
    path = os.path.join(root, os.path.basename(os.path.dirname(os.path.normpath(data_path))))
    os.makedirs(path, exist_ok=True)
    return path
