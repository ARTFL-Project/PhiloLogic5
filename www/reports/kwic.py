from philologic.runtime import kwic_results

from wsgi_helpers import resolve
from wsgi_helpers import json_endpoint


@json_endpoint
def kwic(request, config):
    _kwic_results = resolve(config.db_path, "kwic_results", kwic_results)
    return _kwic_results(request, config)
