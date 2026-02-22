from philologic.runtime import concordance_results

from wsgi_helpers import resolve
from wsgi_helpers import json_endpoint


@json_endpoint
def concordance(request, config):
    _concordance_results = resolve(config.db_path, "concordance_results", concordance_results)
    return _concordance_results(request, config)
