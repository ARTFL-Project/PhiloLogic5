from philologic.runtime import frequency_results

from wsgi_helpers import resolve
from wsgi_helpers import json_endpoint


@json_endpoint
def get_sorted_frequency(request, config):
    """reads through a hitlist. looks up q.frequency_field in each hit, and builds up a list of
    unique values and their frequencies."""
    _frequency_results = resolve(config.db_path, "frequency_results", frequency_results)
    return _frequency_results(request, config)
