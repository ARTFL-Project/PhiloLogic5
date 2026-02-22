from philologic.runtime.DB import DB

from wsgi_helpers import json_endpoint


@json_endpoint
def get_total_results(request, config):
    db = DB(config.db_path + "/data/")
    if request.no_q:
        if request.no_metadata:
            hits = db.get_all(db.locals["default_object_level"], request["sort_order"])
        else:
            hits = db.query(sort_order=request["sort_order"], **request.metadata)
    else:
        hits = db.query(request["q"], request["method"], request["arg"], **request.metadata)
    hits.finish()
    total_results = len(hits)
    return total_results
