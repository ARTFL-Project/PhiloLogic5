from philologic.runtime import generate_text_object

from wsgi_helpers import json_endpoint


@json_endpoint
def get_notes(request, config):
    return generate_text_object(request, config, note=True)
