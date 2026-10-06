import os
import sqlite3
from contextlib import closing
from pathlib import Path

from philologic.Config import MakeDBConfig

def get_web_config(request, config):
    """Retrieve Web Config data, but for where the access file is: a server path the client has no use for"""
    if config.valid_config is False:
        return config.to_dict()
    config.time_series_status = time_series_tester(config)
    db_locals = MakeDBConfig(os.path.join(config.db_path, "data/db.locals.py"))
    config.data["available_metadata"] = db_locals.metadata_fields
    web_config = config.to_dict()
    web_config.pop("access_file", None)
    # Only fields toms has: the web configs of databases loaded before October 2026 list those no text has, which toms
    # has no column for, in the search form and the facets, and searching or faceting by one was a 500
    columns = toms_columns(config.db_path)
    for key in ("metadata", "facets"):
        web_config[key] = [field for field in web_config[key] if not isinstance(field, str) or field in columns]
    return web_config


def toms_columns(db_path):
    """The columns of the database's toms table"""
    uri = f"{(Path(db_path) / 'data' / 'toms.db').resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as dbh:
        return {row[1] for row in dbh.execute("PRAGMA table_info(toms)")}


def time_series_tester(config):
    """Test if we have at least two distinct values for time series"""
    frequencies_file = os.path.join(config.db_path, f"data/frequencies/{config.time_series_year_field}_frequencies")
    if os.path.exists(frequencies_file):
        with open(frequencies_file, encoding="utf8") as input_file:
            line_count = sum(1 for _ in input_file)
        if line_count > 1:
            return True
    return False
