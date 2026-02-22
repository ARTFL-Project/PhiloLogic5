from philologic.runtime import generate_time_series

from wsgi_helpers import resolve
from wsgi_helpers import json_endpoint


@json_endpoint
def time_series(request, config):
    _generate_time_series = resolve(config.db_path, "generate_time_series", generate_time_series)
    return _generate_time_series(request, config)
