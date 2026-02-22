from philologic.runtime import generate_text_object
from philologic.runtime.DB import DB
from philologic.runtime.HitWrapper import ObjectWrapper

from wsgi_helpers import resolve
from wsgi_helpers import json_endpoint


@json_endpoint
def get_text_object(request, config):
    _generate_text_object = resolve(config.db_path, "generate_text_object", generate_text_object)
    db = DB(config.db_path + "/data/")
    zeros = 7 - len(request.philo_id)
    if zeros:
        request.philo_id += zeros * " 0"
    obj = ObjectWrapper(request["philo_id"].split(), db)
    text_object = _generate_text_object(request, config)
    return text_object
