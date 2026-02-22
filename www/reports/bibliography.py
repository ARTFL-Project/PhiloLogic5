from philologic.runtime import bibliography_results

from wsgi_helpers import resolve
from wsgi_helpers import json_endpoint


@json_endpoint
def bibliography(request, config):
    _bibliography_results = resolve(config.db_path, "bibliography_results", bibliography_results)
    bibliography_object, _ = _bibliography_results(request, config)
    return bibliography_object
