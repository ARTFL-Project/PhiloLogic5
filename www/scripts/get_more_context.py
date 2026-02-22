from philologic.runtime import get_concordance_text
from philologic.runtime.DB import DB

from wsgi_helpers import json_endpoint


@json_endpoint
def get_more_context(request, config):
    db = DB(config.db_path + "/data/")
    hit_num = int(request.hit_num)
    hits = db.query(request["q"], request["method"], request["arg"], **request.metadata)
    context_size = config["concordance_length"] * 3
    hit_context = get_concordance_text(db, hits[hit_num], config.db_path, context_size)
    return hit_context
