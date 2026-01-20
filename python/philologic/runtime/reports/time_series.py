#!/var/lib/philologic5/philologic_env/bin/python3
"""Time series"""

import time
from collections import defaultdict

import regex as re

from philologic.runtime.DB import DB
from philologic.runtime.link import make_absolute_query_link
from philologic.runtime.sql_validation import validate_column, validate_object_level


def generate_time_series(request, config):
    db = DB(config.db_path + "/data/")
    year_field = validate_column(config.time_series_year_field, db)
    time_series_object = {"query": dict([i for i in request]), "query_done": False}

    # Invalid date range
    if request.start_date == "invalid" or request.end_date == "invalid":
        time_series_object["results_length"] = 0
        time_series_object["more_results"] = False
        time_series_object["new_start_date"] = 0
        time_series_object["results"] = {"absolute_count": {}, "date_count": {}}
        return time_series_object

    start_date, end_date = get_start_end_date(
        db, config, start_date=request.start_date or None, end_date=request.end_date or None
    )

    # Generate date ranges
    interval = int(request.year_interval)
    date_ranges = []
    # Make sure last date gets included in for loop below by adding one to last step
    for start in range(start_date, end_date + 1, interval):
        end = start + interval - 1
        if end > end_date:
            end = end_date
        date_range = "%d-%d" % (start, end)
        date_ranges.append((start, date_range))

        # Start running queries concurrently to avoid waiting for results in the below loop
        request.metadata[year_field] = date_range
        hits = db.query(request["q"], request["method"], request["arg"], raw_results=True, **request.metadata)

    absolute_count = defaultdict(int)
    date_counts = {}
    total_hits = 0
    last_date_done = start_date
    start_time = time.time()
    max_time = int(request.max_time) or 2
    cursor = db.dbh.cursor()
    object_level = validate_object_level(db.locals.default_object_level)
    for start_range, date_range in date_ranges:
        params = {"report": "concordance", "start": "0", "end": "0"}
        params[year_field] = date_range
        url = make_absolute_query_link(config, request, **params)

        # Get date total count
        if interval != 1:
            end_range = start_range + (int(request["year_interval"]) - 1)
            if request.q:
                cursor.execute(
                    f"select sum(word_count) from toms where {year_field} between ? and ?",
                    (str(start_range), str(end_range)),
                )
            else:
                cursor.execute(
                    f"SELECT COUNT(*) FROM toms WHERE philo_type=? AND {year_field} BETWEEN ? AND ?",
                    (object_level, start_range, end_range),
                )
        else:
            if request.q:
                cursor.execute(
                    f"select sum(word_count) from toms where {year_field}=?",
                    (str(start_range),),
                )
            else:
                cursor.execute(
                    f"SELECT COUNT(*) FROM toms WHERE philo_type=? AND {year_field}=?",
                    (object_level, str(start_range)),
                )
        date_counts[start_range] = cursor.fetchone()[0] or 0

        # Get absolute count
        request.metadata[year_field] = date_range
        hits = db.query(request["q"], request["method"], request["arg"], raw_results=True, **request.metadata)
        hits.finish()
        hit_len = len(hits)

        absolute_count[str(start_range)] = {"label": start_range, "count": hit_len, "url": url}
        total_hits += hit_len

        last_date_done = start_range
        # avoid timeouts by splitting the query if more than request.max_time
        # (in seconds) has been spent in the loop
        if time.time() - start_time > max_time:
            break

    time_series_object["results_length"] = total_hits
    if (last_date_done + int(request.year_interval)) >= end_date:
        time_series_object["more_results"] = False
    else:
        time_series_object["more_results"] = True
        time_series_object["new_start_date"] = last_date_done + int(request.year_interval)
    time_series_object["results"] = {
        "absolute_count": absolute_count,
        "date_count": {str(date): count for date, count in date_counts.items()},
    }

    return time_series_object


def get_start_end_date(db, config, start_date=None, end_date=None):
    """Get start and end date of dataset"""
    year_field = validate_column(config.time_series_year_field, db)
    date_finder = re.compile(r"^.*?(\d{1,}).*")
    cursor = db.dbh.cursor()
    object_type = db.locals["metadata_types"][year_field]
    if object_type == "div":
        year_field_type = ("div1", "div2", "div3")
    else:
        year_field_type = (object_type,)
    cursor.execute(
        f"select {year_field} from toms where {year_field} is not null AND philo_type IN ({','.join('?' for _ in range(len(year_field_type)))})",
        year_field_type,
    )
    dates = []
    for i in cursor:
        try:
            dates.append(int(i[0]))
        except:
            date_match = date_finder.search(i[0])
            if date_match:
                dates.append(int(date_match.groups()[0]))
            else:
                pass
    min_date = min(dates)
    if not start_date:
        start_date = min_date
    max_date = max(dates)
    if not end_date:
        end_date = max_date
    return start_date, end_date
