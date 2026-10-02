from philologic.runtime import get_concordance_text
from philologic.runtime.DB import DB
from philologic.runtime.exceptions import BadRequest, NotFound
from philologic.runtime.WSGIHandler import whole_number

def get_more_context(request, config):
    db = DB(config.db_path + "/data/")
    hit_num = whole_number("hit_num", request.hit_num, None)
    if hit_num is None or hit_num < 0:
        raise BadRequest("hit_num must be the number of a hit, from 0")
    hits = db.query(request["q"], request["method"], request["arg"], sort_order=request["sort_order"], **request.metadata)
    context_size = config["concordance_length"] * 3
    try:
        hit = hits[hit_num]
    except IndexError:
        raise NotFound(f"No hit {hit_num}: there are {len(hits)}") from None
    hit_context = get_concordance_text(db, hit, config.db_path, context_size)
    return hit_context
