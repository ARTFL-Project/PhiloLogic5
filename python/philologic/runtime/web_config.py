#!/var/lib/philologic5/philologic_env/bin/python3

import os
import sys
from functools import lru_cache

from philologic.Config import MakeWebConfig


class brokenConfig(object):
    """Broken config returned with some default values"""

    def __init__(self, db_path, traceback):
        self.web_config_path = db_path + "/data/web_config.cfg"
        self.valid_config = False
        self.traceback = traceback
        self.db_path = db_path

    def __getitem__(self, _):
        return ""

    def to_dict(self):
        """Return dict representation of config"""
        return {"valid_config": False, "traceback": self.traceback, "web_config_path": self.web_config_path}


def WebConfig(db_path):
    """Build runtime web config object: a copy, its own to change, of the database's web_config.cfg (with its
    db.locals.py) as parsed, which is parsed again only once one of them has changed. Parsing them was most of the time
    of short requests, as autocompletes."""
    path = db_path + "/data/web_config.cfg"
    try:
        return _parsed(path, _stamp(path), _stamp(os.path.join(db_path, "data", "db.locals.py"))).copy()
    except Exception as err:
        print(err, file=sys.stderr)
        return brokenConfig(db_path, str(err))


@lru_cache(maxsize=128)
def _parsed(path, *stamps):
    """The web config at path, parsed once for each version of its files (stamps): failures are not kept"""
    return MakeWebConfig(path)


def _stamp(path):
    """What changes when a file is written: its modification time, size and inode (None if there is no file)"""
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size, stat.st_ino
