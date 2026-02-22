from philologic.runtime import generate_toc_object

from wsgi_helpers import resolve
from wsgi_helpers import json_endpoint


@json_endpoint
def get_table_of_contents(request, config):
    _generate_toc_object = resolve(config.db_path, "generate_toc_object", generate_toc_object)
    philo_id = request["philo_id"].split()
    toc_object = _generate_toc_object(request, config)
    current_obj_position = 0
    philo_id = " ".join(philo_id)
    for pos, toc_element in enumerate(toc_object["toc"]):
        if toc_element["philo_id"] == philo_id:
            current_obj_position = pos
            break
    toc_object["current_obj_position"] = current_obj_position
    return toc_object
