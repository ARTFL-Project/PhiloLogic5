from philologic.runtime import generate_text_object

from wsgi_helpers import resolve
from wsgi_helpers import json_endpoint


@json_endpoint
def navigation(request, config):
    _generate_text_object = resolve(config.db_path, "generate_text_object", generate_text_object)
    return _generate_text_object(request, config)
