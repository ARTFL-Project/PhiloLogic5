from philologic.runtime.DB import DB
from philologic.runtime.reports.collocation import collocate_distance, collocation_search_method

def get_total_results(request, config):
    db = DB(config.db_path + "/data/")
    if request.no_q:
        if request.no_metadata:
            hits = db.get_all(db.locals["default_object_level"], request["sort_order"])
        else:
            hits = db.query(sort_order=request["sort_order"], **request.metadata)
    elif request.report == "collocation":  # the hits collocates are counted around
        method, method_arg = collocation_search_method(
            request["q"], db.locals.query_patterns, collocate_distance(request)
        )
        hits = db.query(request["q"], method, method_arg, **request.metadata)
    else:
        hits = db.query(request["q"], request["method"], request["arg"], **request.metadata)
    hits.finish()
    total_results = len(hits)
    return total_results
