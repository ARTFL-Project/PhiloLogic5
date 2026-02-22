from philologic.runtime import generate_toc_object

from wsgi_helpers import resolve
from wsgi_helpers import json_endpoint


@json_endpoint
def table_of_contents(request, config):
    _generate_toc_object = resolve(config.db_path, "generate_toc_object", generate_toc_object)
    return _generate_toc_object(request, config)
