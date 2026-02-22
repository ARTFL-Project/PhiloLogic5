from philologic.runtime import collocation_results

from wsgi_helpers import resolve
from wsgi_helpers import json_endpoint


@json_endpoint
def collocation(request, config):
    _collocation_results = resolve(config.db_path, "collocation_results", collocation_results)
    return _collocation_results(request, config)
