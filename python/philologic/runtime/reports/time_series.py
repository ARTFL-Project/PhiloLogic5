#!/var/lib/philologic5/philologic_env/bin/python3
"""Time series"""

import os

import numba
import numpy as np
import regex as re

from philologic.runtime.DB import DB
from philologic.runtime.link import make_absolute_query_link
from philologic.runtime.sql_validation import validate_column


def _get_doc_year_data(db, year_field):
    """Return (year_array, year_word_counts, year_doc_counts, min_date, max_date).

    Cached as .npz in the hitlists directory. First request per database
    computes from SQL and saves; subsequent requests load from disk.
    """
    cache_path = os.path.join(db.hitlist_dir, "time_series_year_data.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path)
        year_array = data["year_array"]
        years = data["years"]
        word_counts = data["word_counts"]
        doc_counts = data["doc_counts"]
        min_date, max_date = int(data["min_max"][0]), int(data["min_max"][1])
        year_word_counts = dict(zip(years.tolist(), word_counts.tolist()))
        year_doc_counts = dict(zip(years.tolist(), doc_counts.tolist()))
        return year_array, year_word_counts, year_doc_counts, min_date, max_date

    cursor = db.dbh.cursor()
    cursor.execute(
        f"SELECT philo_id, CAST({year_field} AS INTEGER), word_count FROM toms "
        f"WHERE philo_type='doc' AND {year_field} IS NOT NULL"
    )
    max_doc_id = 0
    doc_year_list = []
    year_word_counts = {}
    year_doc_counts = {}
    min_date = None
    max_date = None
    for row in cursor:
        doc_id = int(row[0].split()[0])
        try:
            year = int(row[1])
        except (TypeError, ValueError):
            continue
        doc_year_list.append((doc_id, year))
        if doc_id > max_doc_id:
            max_doc_id = doc_id
        if min_date is None or year < min_date:
            min_date = year
        if max_date is None or year > max_date:
            max_date = year
        wc = int(row[2]) if row[2] else 0
        year_word_counts[year] = year_word_counts.get(year, 0) + wc
        year_doc_counts[year] = year_doc_counts.get(year, 0) + 1

    year_array = np.zeros(max_doc_id + 1, dtype=np.int32)
    for doc_id, year in doc_year_list:
        year_array[doc_id] = year

    # Save to disk for subsequent requests
    years = np.array(sorted(year_word_counts.keys()), dtype=np.int32)
    np.savez(
        cache_path,
        year_array=year_array,
        years=years,
        word_counts=np.array([year_word_counts[y] for y in years], dtype=np.int64),
        doc_counts=np.array([year_doc_counts[y] for y in years], dtype=np.int64),
        min_max=np.array([min_date, max_date], dtype=np.int32),
    )
    return year_array, year_word_counts, year_doc_counts, min_date, max_date


def _corpus_year_counts(db, year_array, metadata):
    """The words and the objects of each year among those metadata selects, as _get_doc_year_data's year_word_counts
    and year_doc_counts are for the whole database: by the year of their document."""
    corpus = db.query(**metadata)
    ids = corpus.read_array()
    cursor = db.dbh.cursor()
    cursor.execute("CREATE TEMP TABLE IF NOT EXISTS _time_series_corpus (philo_id TEXT)")
    cursor.execute("DELETE FROM _time_series_corpus")
    cursor.executemany("INSERT INTO _time_series_corpus VALUES (?)", ((" ".join(map(str, row[:7])),) for row in ids))
    cursor.execute("SELECT toms.philo_id, toms.word_count FROM toms JOIN _time_series_corpus USING (philo_id)")
    year_word_counts, year_object_counts = {}, {}
    for philo_id, word_count in cursor:
        doc_id = int(philo_id.split()[0])
        year = int(year_array[doc_id]) if doc_id < len(year_array) else 0
        if year:
            year_word_counts[year] = year_word_counts.get(year, 0) + int(word_count or 0)
            year_object_counts[year] = year_object_counts.get(year, 0) + 1
    cursor.execute("DELETE FROM _time_series_corpus")
    return year_word_counts, year_object_counts


@numba.jit(nopython=True, nogil=True, cache=True)
def _bucket_hits_by_year(doc_ids, year_array, start_date, interval, n_ranges):
    """Single-pass: read doc_id → look up year → bucket into range."""
    bin_counts = np.zeros(n_ranges, dtype=np.int64)
    total = 0
    year_len = len(year_array)
    for i in range(len(doc_ids)):
        doc_id = doc_ids[i]
        if doc_id < year_len:
            year = year_array[doc_id]
            if year > 0:
                idx = (year - start_date) // interval
                if 0 <= idx < n_ranges:
                    bin_counts[idx] += 1
                    total += 1
    return bin_counts, total


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

    try:
        interval = int(request.year_interval)
    except (ValueError, TypeError):
        interval = int(config.time_series_interval)
    if interval < 1:  # makes no ranges (range() fails on 0)
        interval = int(config.time_series_interval)

    # Get cached doc→year mapping (SQL only on first request per worker)
    year_array, year_word_counts, year_doc_counts, min_date, max_date = _get_doc_year_data(db, year_field)

    # Resolve start/end dates
    start_date = int(request.start_date) if request.start_date else min_date
    end_date = int(request.end_date) if request.end_date else max_date

    # The metadata the request filters by: its own year filter too, which the range of the time series is added to
    filters = {field: value for field, value in request.metadata.items() if value}

    # Fire the word query now that we have start/end dates
    hits = None
    if request.q:
        metadata = dict(request.metadata)
        metadata[year_field] = [value for value in (filters.get(year_field), f"{start_date}-{end_date}") if value]
        hits = db.query(request["q"], request["method"], request["arg"], raw_results=True, **metadata)

    # Generate date ranges for output
    date_ranges = []
    for start in range(start_date, end_date + 1, interval):
        end = start + interval - 1
        if end > end_date:
            end = end_date
        date_ranges.append((start, "%d-%d" % (start, end)))
    n_ranges = len(date_ranges)

    # Aggregate word counts / doc counts into date ranges: of the objects the metadata selects, if it does
    if filters:
        year_word_counts, year_doc_counts = _corpus_year_counts(db, year_array, filters)
    year_totals = year_word_counts if request.q else year_doc_counts
    date_counts = {}
    for range_start, _ in date_ranges:
        total = 0
        range_end = min(range_start + interval, end_date + 1)  # the last range may be cut by end_date
        for y in range(range_start, range_end):
            total += year_totals.get(y, 0)
        date_counts[range_start] = total

    # Absolute hit counts: wait for search, then vectorized bucketing
    if hits is not None:
        hits.finish()
        total_hits = len(hits)

        if total_hits > 0:
            hit_length = hits.length
            # From the file hits has open, by chunks: by name, the hitlist cleanup could remove it in between
            chunks = []
            with hits.open_raw() as f:
                while chunk := f.read(hit_length * 4 * 1_000_000):
                    chunks.append(np.frombuffer(chunk, dtype="u4").reshape(-1, hit_length)[:, 0].copy())
            doc_ids = np.concatenate(chunks)

            bin_counts, total_hits = _bucket_hits_by_year(
                doc_ids, year_array, start_date, interval, n_ranges
            )
        else:
            bin_counts = np.zeros(n_ranges, dtype=np.int64)
    else:
        # Metadata-only (no word query): count docs per range from SQL
        total_hits = 0
        bin_counts = np.zeros(n_ranges, dtype=np.int64)
        for i, (range_start, _) in enumerate(date_ranges):
            bin_counts[i] = date_counts.get(range_start, 0)
            total_hits += int(bin_counts[i])

    # Build absolute_count output matching expected format
    absolute_count = {}
    for i, (range_start, date_range) in enumerate(date_ranges):
        params = {"report": "concordance", "start": "0", "end": "0"}
        params[year_field] = date_range
        url = make_absolute_query_link(config, request, **params)
        absolute_count[str(range_start)] = {
            "label": range_start,
            "count": int(bin_counts[i]),
            "url": url,
        }

    time_series_object["results_length"] = int(total_hits)
    time_series_object["more_results"] = False
    time_series_object["results"] = {
        "absolute_count": absolute_count,
        "date_count": {str(date): count for date, count in date_counts.items()},
    }

    return time_series_object


def time_series_to_csv(results):
    """Convert time series results to CSV string."""
    import csv
    import io

    absolute_count = results.get("absolute_count", {})
    date_count = results.get("date_count", {})
    if not absolute_count:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["period", "count", "total_words"])
    writer.writeheader()
    for period_start in sorted(absolute_count.keys(), key=int):
        entry = absolute_count[period_start]
        writer.writerow({
            "period": entry["label"],
            "count": entry["count"],
            "total_words": date_count.get(period_start, ""),
        })
    return output.getvalue()


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
