from philologic.runtime import group_by_metadata, group_by_range

from wsgi_helpers import json_endpoint


@json_endpoint
def get_landing_page_content(request, config):
    if request.is_range == "true":
        if isinstance(request.query, bytes):
            request_range = request.query.decode("utf8")
        request_range = [item.strip() for item in request.query.lower().split("-")]
        if len(request_range) == 1:
            request_range.append(request_range[0])
        return group_by_range(request_range, request, config)
    return group_by_metadata(request, config)
