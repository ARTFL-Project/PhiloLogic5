from philologic.runtime import landing_page_bibliography

from wsgi_helpers import json_endpoint


@json_endpoint
def get_bibliography(request, config):
    return landing_page_bibliography(request, config)
