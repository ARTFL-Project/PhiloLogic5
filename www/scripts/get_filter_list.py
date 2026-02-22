from philologic.runtime.reports.collocation import build_filter_list

from wsgi_helpers import json_endpoint


@json_endpoint
def get_filter_list(request, config):
    return build_filter_list(request, config)
