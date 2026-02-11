#!/var/lib/philologic5/philologic_env/bin/python3

import os
from wsgiref.handlers import CGIHandler

from philologic.Config import MakeDBConfig
from philologic.runtime.DB import DB

from philologic.runtime import WebConfig

from custom_functions_loader import get_custom


def get_web_config(environ, start_response):
    """Retrieve Web Config data"""
    status = "200 OK"
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response(status, headers)
    db_path = environ.get("PHILOLOGIC_DBPATH", os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    _WebConfig = get_custom(db_path, "WebConfig", WebConfig)
    config = _WebConfig(db_path)
    if config.valid_config is False:
        yield config.to_json()
    else:
        config.time_series_status = time_series_tester(config)
        db_locals = MakeDBConfig(os.path.join(db_path, "data/db.locals.py"))
        config.data["available_metadata"] = db_locals.metadata_fields
        yield config.to_json()


def time_series_tester(config):
    """Test if we have at least two distinct values for time series"""
    frequencies_file = os.path.join(config.db_path, f"data/frequencies/{config.time_series_year_field}_frequencies")
    if os.path.exists(frequencies_file):
        with open(frequencies_file) as input_file:
            line_count = sum(1 for _ in input_file)
        if line_count > 1:
            return True
    return False


if __name__ == "__main__":
    CGIHandler().run(get_web_config)
