#!/var/lib/philologic5/philologic_env/bin/python3


import mmap
import os
import threading
from collections import OrderedDict

import numpy as np
import regex as re
from lxml import etree
from philologic.runtime.DB import DB
from philologic.runtime.exceptions import BadRequest
from philologic.runtime.HitWrapper import ObjectWrapper

from .ObjectFormatter import adjust_bytes, format_concordance, format_text_object

# Module-level bounded mmap cache. Keyed by absolute file path so entries
# are unique across different databases served by the same worker process.
# Each entry keeps the (inode, mtime, size) of the file it maps: a database reloaded in place replaces the file,
# and the old mapping would go on serving the deleted file's bytes at the new offsets.
_mmap_cache = OrderedDict()  # path -> (file signature, mmap object)
_MMAP_CACHE_MAX = 64

# The bounds of the windows of the text objects too long to show whole, by database, object, window length and the
# signatures of toms.db and of the text file: a reader stepping through the parts of an object doesn't wait each time
# for its paragraphs to be read again, and a database reloaded in place gets new ones. Locked for threaded workers.
_window_cache = OrderedDict()  # key -> (signatures, bounds)
_WINDOW_CACHE_MAX = 256
_window_cache_lock = threading.Lock()

_WHITESPACE = (b" ", b"\t", b"\r", b"\x0b", b"\x0c")
_TAG_OPENERS = re.compile(rb"[A-Za-z/!?]")  # the bytes after a < that opens a tag: OCR leaves "a < b" in plain text


def _file_signature(file_path):
    st = os.stat(file_path)
    return (st.st_ino, st.st_mtime_ns, st.st_size)


def _get_mmap(file_path):
    """Return a read-only mmap for *file_path*, using a bounded LRU cache."""
    signature = _file_signature(file_path)
    if file_path in _mmap_cache:
        cached_signature, mm = _mmap_cache[file_path]
        if cached_signature == signature:
            _mmap_cache.move_to_end(file_path)
            return mm
        # The file was replaced or rewritten. The old mmap is not closed: a thread may still be slicing it, and
        # it is unmapped once nothing refers to it.
        del _mmap_cache[file_path]
    fh = open(file_path, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    fh.close()  # mmap keeps its own reference to the fd
    if len(_mmap_cache) >= _MMAP_CACHE_MAX:
        _, (_, old_mm) = _mmap_cache.popitem(last=False)
        old_mm.close()
    _mmap_cache[file_path] = (signature, mm)
    return mm


def get_text(hit, start_byte, length, path):
    """Returns the text of a hit as a string."""
    file_path = path + "/data/TEXT/" + hit.doc.filename
    mm = _get_mmap(file_path)
    return mm[start_byte:start_byte + length]


def get_concordance_text(db, hit, path, context_size):
    """Returns the text of a concordance as a string."""
    # Determine length of text needed
    byte_offsets = sorted(hit.bytes)
    byte_distance = byte_offsets[-1] - byte_offsets[0]
    length = context_size + byte_distance + context_size
    byte_offsets, start_byte = adjust_bytes(byte_offsets, context_size)
    conc_text = get_text(hit, start_byte, length, path)
    conc_text = format_concordance(conc_text, db.locals["token_regex"], byte_offsets)
    return conc_text


def text_file_path(obj, config):
    """The path of the file of the object's text."""
    path = config.db_path
    filename = obj.doc.filename
    if filename and os.path.exists(path + "/data/TEXT/" + filename):
        path += "/data/TEXT/" + filename
    else:
        # workaround for when no filename is returned with the full philo_id of the object
        philo_id = str(obj.philo_id[0]) + " 0 0 0 0 0 0"
        c = obj.db.dbh.cursor()
        c.execute("select filename from toms where philo_type='doc' and philo_id =? limit 1", (philo_id,))
        path += "/data/TEXT/" + c.fetchone()["filename"]
    return path


def _object_rowids(cursor, philo_id):
    """The rowids between which (both excluded) lie the toms rows of the objects inside the object of philo_id. toms
    has the rows of a document in document order, each object followed by those inside it, up to the next sibling of
    the object or of one of its ancestors. A page's are its document's. philo_id LIKE would read every row."""

    def rowid(ids):
        row = cursor.execute(
            "SELECT rowid FROM toms WHERE philo_id=?", (" ".join(map(str, ids + [0] * (7 - len(ids)))),)
        ).fetchone()
        return None if row is None else row[0]

    ids = [int(i) for i in philo_id[:7]]
    while len(ids) > 1 and ids[-1] == 0:
        ids.pop()
    first = rowid(ids)
    if first is None:  # a page
        ids = ids[:1]
        first = rowid(ids) or 0
    for depth in range(len(ids), 0, -1):
        following = rowid(ids[: depth - 1] + [ids[depth - 1] + 1])
        if following is not None:
            return first, following
    return first, (cursor.execute("SELECT max(rowid) FROM toms").fetchone()[0] or 0) + 1


def _paragraph_starts(cursor, philo_id, start_byte, end_byte):
    """The sorted start bytes of the paragraphs (para objects) inside the object, past its first byte. Their start
    bytes are text in toms, and the 110 MB books of plain-text databases have hundreds of thousands of them: one string
    of them all reads in a fraction of the time of as many rows."""
    first, following = _object_rowids(cursor, philo_id)
    row = cursor.execute(
        "SELECT group_concat(CAST(start_byte AS INTEGER), ' ') FROM toms "
        "WHERE rowid > ? AND rowid < ? AND philo_type = 'para'",
        (first, following),
    ).fetchone()
    starts = np.fromstring(row[0] or "", dtype=np.int64, sep=" ")
    starts = starts[(starts > start_byte) & (starts < end_byte)]
    starts.sort()
    return starts


def _open_tag_start(chunk, position):
    """Where the markup tag that position (in chunk, which starts outside any tag) is inside of starts, or -1."""
    tag_end = chunk.rfind(b">", 0, position)
    tag_start = chunk.rfind(b"<", tag_end + 1, position)
    while tag_start != -1 and not _TAG_OPENERS.match(chunk, tag_start + 1):
        tag_start = chunk.rfind(b"<", tag_end + 1, tag_start)
    return tag_start


def _end_without_paragraph(mm, window_start, limit, end_byte):
    """Where to end a window starting at window_start, which no paragraph starting past it and by limit can end: after
    its last line break outside a tag, else after its last space, else at the last character outside a tag and an
    entity. A window with none of these up to limit (a single word or tag longer) ends at the first one past it."""
    chunk = mm[window_start : limit + 1]  # the byte after limit says whether a < there opens a tag
    length = limit - window_start
    for separators in ((b"\n",), _WHITESPACE + (b"\n",)):
        position = length
        while position > 0:
            index = max(chunk.rfind(separator, 0, position) for separator in separators)
            if index == -1:
                break
            tag_start = _open_tag_start(chunk, index + 1)
            if tag_start == -1:
                return window_start + index + 1
            position = tag_start
    position = length
    tag_start = _open_tag_start(chunk, position)
    if tag_start != -1:
        position = tag_start
    entity_start = chunk.rfind(b"&", max(0, position - 12), position)
    if entity_start != -1 and chunk.find(b";", entity_start, position) == -1:
        position = entity_start
    while 0 < position < len(chunk) and 0x80 <= chunk[position] < 0xC0:  # no UTF-8 character cut in two
        position -= 1
    if position > 0:
        return window_start + position
    match = re.compile(rb">" if tag_start != -1 else rb"\s").search(mm, limit, end_byte)
    return end_byte if match is None else match.end()


def window_bounds(obj, config, length):
    """The bounds of the windows a text object is cut into to be shown length bytes at a time at most: [start of the
    object, start of its second window, ..., end of the object]. Cut from the object's start, so that the same bytes
    always get the same window. Each ends where the last paragraph that fits starts, else at a line break or a space
    (see _end_without_paragraph), never inside a tag. An object of length bytes or less is one window."""
    start_byte, end_byte = int(obj.start_byte), int(obj.end_byte)
    if end_byte - start_byte <= length:
        return [start_byte, end_byte]
    path = text_file_path(obj, config)
    key = (config.db_path, " ".join(map(str, obj.philo_id)), start_byte, end_byte, length)
    signatures = (_file_signature(os.path.join(config.db_path, "data", "toms.db")), _file_signature(path))
    with _window_cache_lock:
        cached = _window_cache.get(key)
        if cached is not None and cached[0] == signatures:
            _window_cache.move_to_end(key)
            return cached[1]
    mm = _get_mmap(path)
    paragraphs = _paragraph_starts(obj.db.dbh.cursor(), obj.philo_id, start_byte, end_byte)
    bounds = [start_byte]
    window_start = start_byte
    while end_byte - window_start > length:
        limit = window_start + length
        last = int(np.searchsorted(paragraphs, limit, side="right")) - 1
        if last >= 0 and paragraphs[last] > window_start:
            window_start = int(paragraphs[last])
        else:
            window_start = _end_without_paragraph(mm, window_start, limit, end_byte)
        if window_start < end_byte:
            bounds.append(window_start)
    bounds.append(end_byte)
    with _window_cache_lock:
        _window_cache[key] = (signatures, bounds)
        _window_cache.move_to_end(key)
        if len(_window_cache) > _WINDOW_CACHE_MAX:
            _window_cache.popitem(last=False)
    return bounds


def get_text_obj(obj, config, request, word_regex, note=False, images=True, window=None):
    """Returns the text of an object as a string, formatted according to the request parameters. A window, a
    (start_byte, end_byte) pair within the object, gives only those bytes of it: the highlights outside it are left
    out, and the passages cut to it."""
    path = text_file_path(obj, config)
    mm = _get_mmap(path)
    if window is None:
        obj_start_byte = int(obj.start_byte)
        width = int(obj.end_byte) - obj_start_byte
    else:
        obj_start_byte = window[0]
        width = window[1] - obj_start_byte
    raw_text = mm[obj_start_byte:obj_start_byte + width]
    try:
        byte_offsets = sorted([int(byte) - obj_start_byte for byte in request.byte])
    except ValueError:  ## request.byte contains an empty string
        byte_offsets = []
    start_end_pairs = []
    if request.start_byte:
        start_end_pairs = [(int(request.start_byte) - obj_start_byte, int(request.end_byte) - obj_start_byte)]
    elif request.passages:
        start_end_pairs = sorted(
            [
                (passage["start_byte"] - obj_start_byte, passage["end_byte"] - obj_start_byte)
                for passage in request.passages
            ]
        )
    if window is not None:
        byte_offsets = [offset for offset in byte_offsets if 0 <= offset < width]
        start_end_pairs = [
            (max(start, 0), min(end, width)) for start, end in start_end_pairs if start < width and end > 0
        ]
    formatted_text, imgs = format_text_object(
        obj,
        raw_text,
        config,
        request,
        word_regex,
        obj_start_byte,
        byte_offsets=byte_offsets,
        note=note,
        images=images,
        start_end_pairs=start_end_pairs,
        window=window is not None,
    )
    if images:
        return formatted_text, imgs
    else:
        return formatted_text


def get_tei_header(request, config):
    """Returns the TEI header of a text as a string."""
    path = config.db_path
    db = DB(path + "/data")
    if not request["philo_id"]:
        raise BadRequest("A philo_id is required")
    db[request["philo_id"]]  # BadRequest for no philo_id, NotFound for no such object: they gave a 500
    obj = ObjectWrapper(request["philo_id"].split(), db)
    filename = path + "/data/TEXT/" + obj.filename
    parser = etree.XMLParser(remove_blank_text=True, recover=True)
    xml_tree = etree.parse(filename, parser)
    header = xml_tree.find("teiHeader")
    try:
        header_text = etree.tostring(header, pretty_print=True).decode("utf8")
    except TypeError:  # workaround for when lxml doesn't find the header for whatever reason
        header_text = ""
        with open(filename, encoding="utf8") as file:
            file_content = file.read()
            try:
                start_header_index = re.search(r"<teiheader", file_content, re.I).start()
                end_header_index = re.search(r"</teiheader>", file_content, re.I).end()
            except AttributeError:  # tag not found
                return ""
            header_text = file_content[start_header_index:end_header_index]
    return header_text.replace("<", "&lt;").replace(">", "&gt;")
