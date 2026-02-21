import os
import orjson

from philologic.runtime import WebConfig, WSGIHandler, generate_toc_object

from custom_functions_loader import get_custom


def get_table_of_contents(environ, start_response):
    status = "200 OK"
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response(status, headers)
    db_path = environ.get("PHILOLOGIC_DBPATH", os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    _WebConfig = get_custom(db_path, "WebConfig", WebConfig)
    _WSGIHandler = get_custom(db_path, "WSGIHandler", WSGIHandler)
    _generate_toc_object = get_custom(db_path, "generate_toc_object", generate_toc_object)
    config = _WebConfig(db_path)
    request = _WSGIHandler(environ, config)
    philo_id = request["philo_id"].split()
    toc_object = _generate_toc_object(request, config)
    current_obj_position = 0
    philo_id = " ".join(philo_id)
    for pos, toc_element in enumerate(toc_object["toc"]):
        if toc_element["philo_id"] == philo_id:
            current_obj_position = pos
            break
    toc_object["current_obj_position"] = current_obj_position
    yield orjson.dumps(toc_object)

