from philologic.runtime import get_tei_header

from wsgi_helpers import html_endpoint


@html_endpoint
def get_header(request, config):
    return get_tei_header(request, config)
