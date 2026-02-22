import hashlib
import os

from philologic.runtime.reports.collocation import (
    atomic_pickle_dump,
    decode_group_collocates,
    load_map_field_cache,
)

def get_collocate_distribution(request, config):
    """Get collocate distribution for a single field value from a map_field numpy cache."""
    tids, counts, group_bounds, group_names, count_lemmas, attribute, attribute_value = load_map_field_cache(
        request.file_path
    )

    # Find the requested group
    group_index = group_names.index(request.field)
    field_counter = decode_group_collocates(
        tids, counts, group_bounds, group_index,
        config.db_path, count_lemmas, attribute, attribute_value,
    )

    collocates = sorted(field_counter.items(), key=lambda x: x[1], reverse=True)

    # Cache the field's Counter to disk for downstream use (e.g. comparative_collocations)
    h = hashlib.sha1(f"{request.file_path}:{request.field}".encode("utf-8")).hexdigest()
    field_file_path = os.path.join(config.db_path, "data", "hitlists", f"{h}.pickle")
    atomic_pickle_dump(field_counter, field_file_path)

    return {"collocates": collocates[:100], "file_path": field_file_path}
