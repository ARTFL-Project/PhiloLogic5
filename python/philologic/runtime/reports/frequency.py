#!/usr/bin env python3
"""Frequency results for facets"""

import numpy as np
from urllib.parse import quote_plus

from philologic.runtime.DB import DB
from philologic.runtime.MetadataQuery import bulk_load_metadata
from philologic.runtime.link import make_absolute_query_link
from philologic.runtime.sql_validation import validate_request_column

OBJ_DICT = {"doc": 1, "div1": 2, "div2": 3, "div3": 4, "para": 5, "sent": 6, "word": 7}


def _object_level(philo_id):
    """The philo_id of an object, without its trailing zeros: (5, 2) for division 2 of document 5."""
    depth = len(philo_id)
    while depth > 1 and philo_id[depth - 1] == 0:
        depth -= 1
    return tuple(int(x) for x in philo_id[:depth])


def _filtered_word_counts(db, filters, field_cache, prefix_len):
    """The words of each value of a field among the objects filters selects (the denominators of relative
    frequencies). field_cache maps the field's objects (philo_id prefixes of prefix_len, padded with 0) to
    (value, word_count). A selected object finer than the field's counts its own words, under the value of the field's
    object it is in; a coarser one the words of the field's objects it holds."""
    selected = [_object_level(row) for row in db.query(**filters).read_array()[:, :7].tolist()]
    counts = {}
    finer = [o for o in selected if len(o) >= prefix_len or (prefix_len == 4 and len(o) >= 2)]
    coarser = {}
    for obj in selected:
        if not (len(obj) >= prefix_len or (prefix_len == 4 and len(obj) >= 2)):
            coarser.setdefault(len(obj), set()).add(obj)
    if finer:
        cursor = db.dbh.cursor()
        cursor.execute("CREATE TEMP TABLE IF NOT EXISTS _facet_objects (philo_id TEXT)")
        cursor.execute("DELETE FROM _facet_objects")
        cursor.executemany("INSERT INTO _facet_objects VALUES (?)", ((" ".join(map(str, o + (0,) * (7 - len(o)))),) for o in finer))
        cursor.execute("SELECT toms.philo_id, toms.word_count FROM toms JOIN _facet_objects USING (philo_id)")
        for philo_id, word_count in cursor.fetchall():
            obj = tuple(int(x) for x in philo_id.split())
            if prefix_len == 4:  # div fields: the finest division with a value, as hits are counted
                prefixes = [p for p in (obj[:level] + (0,) * (4 - level) for level in (4, 3, 2)) if field_cache.get(p, ("",))[0]]
            else:  # with no value too, counted as "" (the NULL bucket's words)
                prefixes = [obj[:prefix_len]] if obj[:prefix_len] in field_cache else []
            if prefixes:
                value = f"{field_cache[prefixes[0]][0]}"
                counts[value] = counts.get(value, 0) + int(word_count or 0)
        cursor.execute("DELETE FROM _facet_objects")
    if coarser:
        for prefix, (value, word_count) in field_cache.items():
            if any(prefix[:depth] in objects for depth, objects in coarser.items()):
                counts[f"{value}"] = counts.get(f"{value}", 0) + int(word_count or 0)
    return counts


def frequency_results(request, config):
    """reads through a hitlist. looks up request.frequency_field in each hit, and builds up a list of
    unique values and their frequencies."""
    db = DB(config.db_path + "/data/")
    frequency_field = validate_request_column(request.frequency_field, db)
    biblio_search = False
    if request.q == "" and request.no_q:
        biblio_search = True
        if request.no_metadata:
            hits = db.get_all(
                db.locals["default_object_level"],
                sort_order=["rowid"],
                raw_results=True,
            )
        else:
            hits = db.query(sort_order=["rowid"], raw_results=True, **request.metadata)
    else:
        hits = db.query(
            request["q"],
            request["method"],
            request["arg"],
            raw_results=True,
            **request.metadata,
        )

    metadata_type = db.locals["metadata_types"][frequency_field]
    has_metadata_filter = any(v for v in request.metadata.values())

    # Build metadata_dict and word_counts via bulk_load_metadata.
    # When no metadata filters are active, load word_count in the same scan.
    # With filters, word_counts need a separate filtered query.
    metadata_dict = {}
    word_counts_by_field_name = {}
    prefix_len, cache = bulk_load_metadata(db, [frequency_field], extra_columns=["word_count"])[frequency_field]
    for prefix, (field_name, word_count) in cache.items():
        if field_name:
            metadata_dict[prefix] = field_name
        if not biblio_search and not has_metadata_filter:  # with no value too, under "": the NULL bucket's words
            wc = int(word_count) if word_count else 0
            word_counts_by_field_name[f"{field_name}"] = word_counts_by_field_name.get(f"{field_name}", 0) + wc
    if not biblio_search and has_metadata_filter:  # of the objects the filters select, not of all
        filters = {k: v for k, v in request.metadata.items() if v}
        word_counts_by_field_name = _filtered_word_counts(db, filters, cache, prefix_len)

    base_url = make_absolute_query_link(
        config,
        request,
        frequency_field="",
        start="0",
        end="0",
        report=request.report,
        script="",
    )

    hits.finish()

    # Use numpy to count hits per object-level ID
    if metadata_type != "div":
        object_level = OBJ_DICT[metadata_type]
    else:
        object_level = OBJ_DICT["div3"]  # Extract at finest div level
    id_counts, total_hits = __count_hits_by_level(hits, object_level)

    # Build frequency counts from distinct IDs with pre-computed hit counts
    counts = {}
    for philo_id, hit_count in id_counts.items():
        if metadata_type == "div":
            key = ""
            for div in ["div3", "div2", "div1"]:
                prefix = philo_id[: OBJ_DICT[div]]
                prefix = prefix + (0,) * (4 - len(prefix))
                if prefix in metadata_dict:
                    key = metadata_dict[prefix]
                    break
            if not key:
                continue
        elif philo_id in metadata_dict:
            key = metadata_dict[philo_id]
        elif philo_id in cache:  # an object with no value of the field: the NULL bucket
            if "NULL" not in counts:
                counts["NULL"] = {"count": 0, "metadata": {frequency_field: "NULL"}, "url": f"{base_url}&{frequency_field}=NULL"}
                if not biblio_search:
                    counts["NULL"]["total_word_count"] = word_counts_by_field_name.get("", 0)
            counts["NULL"]["count"] += hit_count
            continue
        else:
            continue
        key = f"{key}"  # convert potential integers to strings
        if key not in counts:
            counts[key] = {"count": 0, "metadata": {frequency_field: key}}
            counts[key]["url"] = f'{base_url}&{frequency_field}="{quote_plus(key)}"'
            if not biblio_search:
                try:
                    counts[key]["total_word_count"] = word_counts_by_field_name[key]
                except KeyError:
                    # Worst case when there are different values for the field in div1, div2, and div3
                    query_metadata = {k: v for k, v in request.metadata.items() if v}
                    query_metadata[frequency_field] = f'"{key}"'
                    local_hits = db.query(**query_metadata)
                    counts[key]["total_word_count"] = local_hits.get_total_word_count()
        counts[key]["count"] += hit_count

    return _sorted_result(counts, hits, request, biblio_search)


def _sorted_result(counts, hits, request, biblio_search):
    """The report: the top 100 values by hit count, and their relative frequencies."""
    results_list = []
    for label, data in sorted(counts.items(), key=lambda x: x[1]["count"], reverse=True)[:100]:
        entry = dict(data)
        entry["label"] = label
        results_list.append(entry)

    result = {
        "results": results_list,
        "results_length": len(hits),
        "query": dict([i for i in request]),
    }

    # Compute relative frequency (per 10,000 words) — top 100 by relative frequency
    if not biblio_search:
        relative_list = []
        for label, data in counts.items():
            total_wc = data.get("total_word_count", 0)
            if total_wc > 0:
                relative_list.append({
                    "label": label,
                    "count": round((data["count"] / total_wc) * 10000, 2),
                    "absolute_count": data["count"],
                    "total_word_count": total_wc,
                    "metadata": data["metadata"],
                    "url": data["url"],
                })
        relative_list.sort(key=lambda x: x["count"], reverse=True)
        result["relative_results"] = relative_list[:100]

    return result


def __count_hits_by_level(hits, object_level):
    """Stream sorted hitlist with numpy, return per-ID hit counts.

    Exploits sorted hitlist order: uses vectorized diff to find boundaries
    and computes run lengths without converting every hit to Python.

    Returns:
        (id_counts, total_results) where id_counts is {philo_id_tuple: hit_count}
    """
    CHUNK_SIZE = 100_000
    id_counts = {}
    total_results = 0
    prev_id = None
    prev_count = 0

    with hits.open_raw() as f:
        while True:
            chunk = f.read(hits.length * 4 * CHUNK_SIZE)
            if not chunk:
                break
            arr = np.frombuffer(chunk, dtype="u4").reshape(-1, hits.length)
            total_results += arr.shape[0]

            if object_level == 1:
                col = arr[:, 0]
                change_indices = np.where(col[1:] != col[:-1])[0] + 1
                boundaries = np.concatenate([[0], change_indices, [len(col)]])
                run_lengths = np.diff(boundaries)
                unique_vals = col[boundaries[:-1]]

                for val, rlen in zip(unique_vals.tolist(), run_lengths.tolist()):
                    key = (val,)
                    if key == prev_id:
                        prev_count += rlen
                    else:
                        if prev_id is not None:
                            id_counts[prev_id] = prev_count
                        prev_id = key
                        prev_count = rlen
            else:
                cols = np.ascontiguousarray(arr[:, :object_level])
                void_col = cols.view(np.dtype((np.void, object_level * 4))).ravel()
                change_indices = np.where(void_col[1:] != void_col[:-1])[0] + 1
                boundaries = np.concatenate([[0], change_indices, [len(void_col)]])
                run_lengths = np.diff(boundaries)

                for idx, rlen in zip(boundaries[:-1].tolist(), run_lengths.tolist()):
                    key = tuple(cols[idx].tolist())
                    if key == prev_id:
                        prev_count += rlen
                    else:
                        if prev_id is not None:
                            id_counts[prev_id] = prev_count
                        prev_id = key
                        prev_count = rlen

    if prev_id is not None:
        id_counts[prev_id] = prev_count

    return id_counts, total_results


if __name__ == "__main__":
    import sys

    from philologic.runtime import WebConfig

    class Request:
        def __init__(self, q, field, metadata):
            self.q = q
            self.frequency_field = field
            self.no_metadata = False
            self.no_q = False
            self.metadata = metadata
            self.method = "proxy"
            self.report = "frequency"
            self.arg = ""
            self.start = 0

        def __getitem__(self, item):
            return getattr(self, item)

        def __iter__(self):
            for item in ["q", "frequency_field", "report"]:
                yield item, self[item]

    query_term, field, db_path = sys.argv[1:]
    config = WebConfig(db_path)
    request = Request(query_term, field, {})
    frequency_results(request, config)
