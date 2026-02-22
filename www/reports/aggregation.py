from philologic.runtime import aggregation_by_field

from wsgi_helpers import resolve
from wsgi_helpers import json_endpoint


@json_endpoint
def aggregation(request, config):
    _aggregation_by_field = resolve(config.db_path, "aggregation_by_field", aggregation_by_field)
    return _aggregation_by_field(request, config)
