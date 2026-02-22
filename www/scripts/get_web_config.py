import os

from philologic.Config import MakeDBConfig

def get_web_config(request, config):
    """Retrieve Web Config data"""
    if config.valid_config is False:
        return config.to_dict()
    config.time_series_status = time_series_tester(config)
    db_locals = MakeDBConfig(os.path.join(config.db_path, "data/db.locals.py"))
    config.data["available_metadata"] = db_locals.metadata_fields
    return config.to_dict()


def time_series_tester(config):
    """Test if we have at least two distinct values for time series"""
    frequencies_file = os.path.join(config.db_path, f"data/frequencies/{config.time_series_year_field}_frequencies")
    if os.path.exists(frequencies_file):
        with open(frequencies_file) as input_file:
            line_count = sum(1 for _ in input_file)
        if line_count > 1:
            return True
    return False
