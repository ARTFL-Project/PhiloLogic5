#!/var/lib/philologic5/philologic_env/bin/python3

import sys

from orjson import dumps
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

    def to_json(self):
        """Return JSON representation of config"""
        return dumps({"valid_config": False, "traceback": self.traceback, "web_config_path": self.web_config_path})


def WebConfig(db_path):
    """Build runtime web config object"""
    try:
        return MakeWebConfig(db_path + "/data/web_config.cfg")
    except Exception as err:
        print(err, file=sys.stderr)
        return brokenConfig(db_path, str(err))
