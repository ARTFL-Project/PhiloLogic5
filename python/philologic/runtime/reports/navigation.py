#!/var/lib/philologic5/philologic_env/bin/python3
"""Navigate inside objects"""

from bisect import bisect_right

import regex as re

from philologic.runtime.citations import citation_links, citations
from philologic.runtime.DB import DB
from philologic.runtime.exceptions import BadRequest, NotFound
from philologic.runtime.get_text import get_text_obj, window_bounds
from philologic.runtime.WSGIHandler import whole_number


def window_length(config):
    """The navigation_window_length of the web config: 0 (objects shown whole) if it is no positive whole number."""
    try:
        length = int(config["navigation_window_length"] or 0)
    except (KeyError, TypeError, ValueError):
        return 0
    return max(length, 0)


def text_window(obj, config, request):
    """The part of a text object the navigation report shows, if the object is longer than navigation_window_length:
    the window of its fixed partition (see get_text.window_bounds) holding the byte of window_start, else the first
    highlighted word (byte) in the object, else the start of the highlighted passage (start_byte), else the first.
    None for an object shown whole."""
    length = window_length(config)
    if not length or int(obj.end_byte) - int(obj.start_byte) <= length:
        return None
    bounds = window_bounds(obj, config, length)
    starts = bounds[:-1]
    target = None
    if request["window_start"] != "":
        target = whole_number("window_start", request["window_start"], None)
    else:
        try:
            offsets = sorted(int(byte) for byte in request.byte)
        except ValueError:  # an empty byte: no highlight, as get_text_obj has it
            offsets = []
        in_object = [offset for offset in offsets if bounds[0] <= offset < bounds[-1]]
        if in_object:
            target = in_object[0]
        elif isinstance(request.start_byte, int) and bounds[0] <= request.start_byte < bounds[-1]:
            target = request.start_byte
    part = 0 if target is None else max(bisect_right(starts, target) - 1, 0)
    return {
        "start_byte": bounds[part],
        "end_byte": bounds[part + 1],
        "prev_start_byte": bounds[part - 1] if part > 0 else None,
        "next_start_byte": bounds[part + 1] if part + 1 < len(starts) else None,
        "part": part + 1,
        "part_count": len(starts),
    }


def generate_text_object(request, config, note=False):
    """Return text object given an philo_id. A client that can show it a part at a time (windowed=1) gets, of an
    object longer than navigation_window_length, the part text_window picks, and its bounds and neighbors as window."""
    # Block doc-level text rendering to prevent scraping
    parts = request.philo_id.split()
    non_zero = [p for p in parts if p != "0"]
    if len(non_zero) <= 1:
        raise BadRequest("Document-level text rendering is not allowed.")

    # verify this isn't a page ID or if this is a note
    if len(request.philo_id.split()) == 9 and note is not True:
        width = 9
    else:
        width = 7
    db = DB(config.db_path + "/data/", width=width)
    if note:
        target = request.target.replace("#", "")
        doc_id = request.philo_id.split()[0] + " %"
        cursor = db.dbh.cursor()
        cursor.execute("select philo_id from toms where id=? and philo_id like ? limit 1", (target, doc_id))
        row = cursor.fetchone()
        if row is None:
            raise NotFound(f"No note {target} in {request.philo_id}")
        philo_id = row["philo_id"].split()[:7]
        obj = db[philo_id]
    else:
        obj = db[request.philo_id]
        philo_id = obj.philo_id
    if width != 9:
        while obj["philo_name"] == "__philo_virtual" and obj["philo_type"] != "div1":
            philo_id.pop()
            obj = db[philo_id]
    philo_id = list(obj.philo_id)
    while int(philo_id[-1]) == 0:
        philo_id.pop()
    text_object = {"query": dict([i for i in request]), "philo_id": " ".join([str(i) for i in philo_id])}
    text_object["prev"] = neighboring_object_id(db, obj.prev, width)
    text_object["next"] = neighboring_object_id(db, obj.next, width)
    metadata_fields = {}
    for metadata in db.locals["metadata_fields"]:
        metadata_fields[metadata] = obj[metadata]
    text_object["metadata_fields"] = metadata_fields
    if width != 9:
        citation_hrefs = citation_links(db, config, obj)
        citation = citations(obj, citation_hrefs, config, report="navigation")
    else:
        db = DB(config.db_path + "/data/", width=7)
        doc_obj = db[obj.philo_id[0]]
        citation_hrefs = citation_links(db, config, doc_obj)
        citation = citations(doc_obj, citation_hrefs, config, report="navigation")
    text_object["citation"] = citation
    window = None
    if request["windowed"] not in ("", "0", "false") and note is not True:
        window = text_window(obj, config, request)
    if window is None:
        text, imgs = get_text_obj(obj, config, request, db.locals["token_regex"], note=note)
    else:
        text, imgs = get_text_obj(
            obj, config, request, db.locals["token_regex"], note=note, window=(window["start_byte"], window["end_byte"])
        )
    if config.navigation_formatting_regex:
        for pattern, replacement in config.navigation_formatting_regex:
            text = re.sub(r"%s" % pattern, "%s" % replacement, text)
    text_object["text"] = text
    text_object["imgs"] = imgs
    if window is not None:
        text_object["window"] = window
    return text_object


def neighboring_object_id(db, philo_id, width):
    """Get neighboring object ID"""
    if not philo_id:
        return ""
    philo_id = philo_id.split()[:width]
    while philo_id[-1] == "0":
        philo_id.pop()
    philo_id = str(" ".join(philo_id))
    obj = db[philo_id]
    if obj["philo_name"] == "__philo_virtual" and obj["philo_type"] != "div1":
        # Remove the last number (1) in the philo_id and point to one object
        # level lower
        philo_id = " ".join(philo_id.split()[:-1])
    return philo_id
