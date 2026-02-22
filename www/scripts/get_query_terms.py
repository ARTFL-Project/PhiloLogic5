from philologic.runtime.DB import DB
from philologic.runtime.Query import get_expanded_query

def get_query_terms(request, config):
    db = DB(config.db_path + "/data/")
    hits = db.query(request["q"], request["method"], request["arg"], **request.metadata)
    hits.finish()
    expanded_terms = get_expanded_query(hits)
    return expanded_terms[0]
