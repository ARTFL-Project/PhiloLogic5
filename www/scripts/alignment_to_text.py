from philologic.runtime.DB import DB
from philologic.runtime.link import byte_range_to_link

from wsgi_helpers import json_endpoint


@json_endpoint
def alignment_to_text(request, config):
    db = DB(config.db_path + "/data/")
    link = byte_range_to_link(db, config, request)
    return {"link": link}
