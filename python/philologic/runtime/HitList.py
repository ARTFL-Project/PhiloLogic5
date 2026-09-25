#!/var/lib/philologic5/philologic_env/bin/python3

import fcntl
import os
import struct
import threading
import time
from contextlib import contextmanager

import numpy as np
from unidecode import unidecode

from .HitWrapper import HitWrapper
from .sql_validation import validate_column, validate_philo_type

obj_dict = {"doc": 1, "div1": 2, "div2": 3, "div3": 4, "para": 5, "sent": 6, "word": 7}


def sort_key(value, ascii_conversion):
    """Sort key of a metadata value: numbers in numeric order, then text regardless of case (and of accents with
    ascii_conversion), then missing values."""
    if value is None or value == "":
        return (2, "")
    if isinstance(value, (int, float)):
        return (0, value)
    value = str(value)
    return (1, (unidecode(value) if ascii_conversion else value).casefold())


def _id_keys(ids, depth):
    """The first `depth` columns of rows of object ids, as byte strings that compare like the id tuples."""
    return np.ascontiguousarray(ids[:, :depth], dtype=">u4").view(f"S{4 * depth}").ravel()


def sort_hits(hits, dbh, sort_order, ascii_conversion):
    """Order of hits (rows of object ids) sorted by the sort_order metadata of the objects containing them.

    Sort fields describe one object type (e.g. doc for author and title), so rather than hits, we sort those
    objects, then give each hit the rank of the deepest one containing it. Ties, and hits within an object, keep
    their load order; hits in no such object come last.

    Like HitWrapper, a hit's value for a field is that of the deepest object of the field's type containing it
    which has one: a div3 without a head shows its div2's or div1's, so it is sorted by that value too.
    """
    field_types = []
    for field in sort_order:
        philo_type = dbh.locals["metadata_types"][field]
        field_types.append({"div1", "div2", "div3"} if philo_type == "div" else {philo_type})
    philo_types = sorted(validate_philo_type(t) for t in set().union(*field_types))
    cursor = dbh.dbh.cursor()
    cursor.execute(
        f"select philo_id, philo_type, {', '.join(sort_order)} from toms "
        f"where philo_type in ({', '.join('?' for _ in philo_types)}) order by rowid",
        philo_types,
    )
    rows = cursor.fetchall()
    object_ids = np.array([[int(i) for i in row["philo_id"].split()[:7]] for row in rows], dtype=np.uint32).reshape(-1, 7)
    depths = np.array([obj_dict[row["philo_type"]] for row in rows])

    missing = sort_key(None, ascii_conversion)
    keys_by_id = {}  # object id -> its sort keys, with the ones it lacks taken from the objects containing it
    object_keys = [None] * len(rows)
    id_lists = object_ids.tolist()
    for i in np.argsort(depths, kind="stable").tolist():  # containing objects first
        row = rows[i]
        philo_id = tuple(id_lists[i][: depths[i]])
        keys = [
            sort_key(row[f], ascii_conversion) if row["philo_type"] in types else missing
            for f, types in zip(sort_order, field_types)
        ]
        if missing in keys:
            for d in range(len(philo_id) - 1, 0, -1):  # the deepest containing object, its keys inherited already
                parent = keys_by_id.get(philo_id[:d])
                if parent is not None:
                    keys = [p if k == missing else k for k, p in zip(keys, parent)]
                    break
        keys_by_id[philo_id] = object_keys[i] = keys
    object_order = sorted(range(len(rows)), key=object_keys.__getitem__)
    object_ranks = np.empty(len(rows), dtype=np.int64)
    object_ranks[object_order] = np.arange(len(rows))

    ranks = np.full(len(hits), len(rows), dtype=np.int64)
    unranked = np.ones(len(hits), dtype=bool)
    for depth in sorted(set(depths.tolist()), reverse=True):  # the deepest object containing a hit decides
        at_depth = depths == depth
        keys = _id_keys(object_ids[at_depth], depth)
        by_key = np.argsort(keys)
        keys, ranks_at_depth = keys[by_key], object_ranks[at_depth][by_key]
        hit_keys = _id_keys(hits, depth)
        found = np.minimum(np.searchsorted(keys, hit_keys), len(keys) - 1)
        matched = unranked & (keys[found] == hit_keys)
        ranks[matched] = ranks_at_depth[found[matched]]
        unranked &= ~matched
    return np.argsort(ranks, kind="stable")


def sorted_hitlist_file(filename, length, dbh, sort_order, ascii_conversion):
    """Path of a copy of a complete hitlist file with its hits sorted by sort_order, sorting them unless done
    already. It is written whole under a temporary name, so readers only ever see it complete."""
    path = f"{filename}.sorted.{','.join(sort_order)}"
    if not os.path.exists(path):
        hits = np.fromfile(filename, dtype=np.uint32).reshape(-1, length)
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        hits[sort_hits(hits, dbh, sort_order, ascii_conversion)].tofile(tmp)
        os.replace(tmp, path)
    return path


class HitlistClaim:
    """The right to produce a hitlist: an exclusive flock on it, held until finish_hitlist() writes its .done flag."""

    def __init__(self, fd):
        self.fd = fd
        self.handed_over = False

    def hand_over(self):
        """Record that a producer now owns the claim and will finish_hitlist() it on every path, so that the claimer
        failing afterwards (e.g. while building its HitList) no longer undoes it under the producer."""
        self.handed_over = True

    def release(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def _create_locked(filename):
    """Create filename with its flock already held, so that no other claimer ever sees it unlocked. Returns the file
    descriptor, or None if filename already exists."""
    tmp = f"{filename}.{os.urandom(8).hex()}.claim"
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)  # nobody else knows tmp, so this never waits
        os.link(tmp, filename)
        return fd
    except FileExistsError:
        os.close(fd)
        return None
    except OSError:  # no hard links on this filesystem: create it in place, where it is briefly visible unlocked
        os.close(fd)
        try:
            fd = os.open(filename, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        except FileExistsError:
            return None
        fcntl.flock(fd, fcntl.LOCK_EX)
        return fd
    except BaseException:
        os.close(fd)
        raise
    finally:
        os.remove(tmp)


def _same_file(fd, path):
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return False
    fst = os.fstat(fd)
    return (st.st_dev, st.st_ino) == (fst.st_dev, fst.st_ino)


@contextmanager
def claim_hitlist(filename):
    """Try to become the producer of a hitlist file.

    Yields a HitlistClaim if the caller must produce the hitlist, None if it is done or being produced. The claim
    holds an exclusive flock on the hitlist, which the producer keeps until finish_hitlist() has written the .done
    flag. If the producer dies before that, the kernel releases the lock and the next claim takes the unfinished
    hitlist over, so orphaned hitlists are told apart from ones still being produced, however long that takes. If
    the caller fails before handing the claim over to its producer, the claim is undone.
    """
    while True:
        fd = _create_locked(filename)
        if fd is not None:
            break
        try:
            # Read-only is enough for flock, and lets processes that can read a hitlist but not write it use it
            fd = os.open(filename, os.O_RDONLY)
        except FileNotFoundError:  # removed in between: try again
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:  # a live producer holds it
            os.close(fd)
            yield None
            return
        if not _same_file(fd, filename):  # replaced while we were locking it: try again
            os.close(fd)
            continue
        if os.path.exists(filename + ".done"):
            os.close(fd)
            yield None
            return
        # Orphaned by a producer that died: empty it in place and produce it again. Its readers keep it open and read
        # the new content, which is the same since searches are deterministic.
        try:
            writable = os.open(filename, os.O_WRONLY)
        except (PermissionError, FileNotFoundError):  # not ours to produce
            os.close(fd)
            yield None
            return
        try:
            replaced = not _same_file(writable, filename) or os.fstat(writable).st_ino != os.fstat(fd).st_ino
            if not replaced:
                os.ftruncate(writable, 0)
        finally:
            os.close(writable)
        if replaced:
            os.close(fd)
            continue
        break
    for suffix in (".done", ".terms"):  # left behind by an earlier producer, or by a hitlist cleanup removed
        try:
            os.remove(filename + suffix)
        except FileNotFoundError:
            pass
    claim = HitlistClaim(fd)
    try:
        yield claim
    except BaseException:
        if not claim.handed_over:
            claim.release()
            try:
                os.remove(filename)
            except FileNotFoundError:
                pass
        raise


def being_produced(fh):
    """Whether a producer holds its claim on the hitlist file open as fh."""
    try:
        fcntl.flock(fh, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    fcntl.flock(fh, fcntl.LOCK_UN)
    return False


def finish_hitlist(filename, lock, message="1"):
    """Mark a hitlist produced under claim_hitlist() as complete, then release its claim."""
    if lock is not None:
        lock.hand_over()  # complete (or, if writing .done fails, orphaned) from here on: never undone by the claimer
    try:
        with open(filename + ".done", "w") as flag:
            flag.write(message)
    finally:
        if lock is not None:
            lock.release()


class HitList(object):
    """Iterable containing philologic hits"""

    def __init__(
        self,
        filename,
        words,
        dbh,
        encoding=None,
        doc=0,
        byte=6,
        method="proxy",
        methodarg=3,
        sort_order=None,
        raw=False,
        raw_bytes=False,
        ascii_conversion=True,
        produce=None,
    ):
        self.filename = filename
        # produce(lock=...) produces this hitlist again, if its producer dies before finishing it
        self.produce = produce
        self.next_producer_check = time.monotonic() + 1
        self.words = words
        self.method = method
        self.methodarg = methodarg
        self.sort_order = sort_order
        if self.sort_order == ["rowid"]:
            self.sort_order = None
        self.raw = raw  # if true, this return the raw hitlist consisting of an iterable of philo_ids
        self.raw_bytes = raw_bytes  # if true, this returns the raw hitlist consisting of an iterable of bytes
        self.dbh = dbh
        self.encoding = encoding or "utf-8"
        self.length = 7 + 2 * (words)
        self.fh = open(self.filename, "rb")  # need a full path here.
        self.format = "%dI" % self.length  # short for object id's, int for byte offset.
        self.hitsize = struct.calcsize(self.format)
        self.doc = doc
        self.byte = byte
        self.position = 0
        self.done = False
        self.update()

        # The file hits are read from: the hitlist itself, or once complete, a copy sorted by sort_order
        self.data_file = self.filename
        if self.sort_order:
            self.sort_order = [validate_column(col, dbh) for col in self.sort_order]
            self.finish()
            self.data_file = sorted_hitlist_file(self.filename, self.length, dbh, self.sort_order, ascii_conversion)
            self.fh.close()
            self.fh = open(self.data_file, "rb")
            self.position = 0

    def __getitem__(self, n):
        self.update()
        if isinstance(n, slice):
            return self.get_slice(n)
        else:
            if self.raw:
                return self.readhit(n)
            else:
                return HitWrapper(self.readhit(n), self.dbh)

    def get_slice(self, n):
        self.update()
        # need to handle negative offsets.
        slice_position = n.start or 0
        self.seek(slice_position)
        while True:
            if n.stop is not None:
                if slice_position >= n.stop:
                    break
            try:
                hit = self.readhit(slice_position)
            except IndexError as IOError:
                break
            if self.raw:
                yield hit
            else:
                yield HitWrapper(hit, self.dbh)
            slice_position += 1

    def __len__(self):
        self.update()
        return self.count

    def __iter__(self):
        self.update()
        iter_position = 0
        self.seek(iter_position)
        while True:
            try:
                hit = self.readhit(iter_position)
            except IndexError as IOError:
                break
            if self.raw:
                yield hit
            else:
                yield HitWrapper(hit, self.dbh)
            iter_position += 1

    def seek(self, n):
        if self.position == n:
            pass
        else:
            while n >= len(self):
                if self.done:
                    raise IndexError
                else:
                    time.sleep(0.01)
                    self.update()
            offset = self.hitsize * n
            self.fh.seek(offset)
            self.position = n

    def update(self):
        # Since the file could be growing, we should frequently check size/ if it's finished yet.
        if self.done:
            pass
        else:
            if os.path.exists(self.filename + ".done"):
                # Only once its producer has let go: until then, the flag is one left behind by an earlier
                # hitlist of the same name, which the producer is about to remove.
                self.done = not being_produced(self.fh)
            elif self.produce is not None and time.monotonic() >= self.next_producer_check:
                self.next_producer_check = time.monotonic() + 1
                with claim_hitlist(self.filename) as lock:
                    if lock is not None:  # its producer died before finishing it
                        self.produce(lock=lock)
            self.size = os.stat(self.filename).st_size  # in bytes
            self.count = int(self.size / self.hitsize)

    def finish(self):
        while not self.done:
            self.update()
            time.sleep(0.01)

    def readhit(self, n):
        # reads hitlist into buffer, unpacks
        # should do some work to read k at once, track buffer state.
        if not self.done:
            self.update()
        while n >= len(self):
            if self.done:
                raise IndexError
            else:
                time.sleep(0.01)
                self.update()
        if n != self.position:
            offset = self.hitsize * n
            self.fh.seek(offset)
            self.position = n
        buffer = self.fh.read(self.hitsize)
        self.position += 1
        if self.raw_bytes:
            return buffer
        return struct.unpack(self.format, buffer)

    def get_total_word_count(self):
        philo_ids = []
        total_count = 0
        iter_position = 0
        self.seek(iter_position)
        while True:
            try:
                hit = self.readhit(iter_position)
                philo_ids.append(hit)
            except IndexError as IOError:
                break
            iter_position += 1
        c = self.dbh.dbh.cursor()
        ids = []
        for id in philo_ids:
            ids.append(" ".join(map(str, id)))
            if len(ids) == 999:  # max expression tree in sqlite is 1000
                placeholders = " OR ".join("philo_id=?" for _ in ids)
                c.execute(f"SELECT SUM(word_count) FROM toms WHERE {placeholders}", ids)
                total_count += int(c.fetchone()[0])
                ids = []
        if ids:
            placeholders = " OR ".join("philo_id=?" for _ in ids)
            c.execute(f"SELECT SUM(word_count) FROM toms WHERE {placeholders}", ids)
            total_count += int(c.fetchone()[0])
        return total_count


# TODO: check if we still need this...
class CombinedHitlist(object):
    """A combined hitlists used for binding collocation hits"""

    def __init__(self, *hitlists):
        self.combined_hitlist = []
        # sentence_ids = set()
        # for hit in sorted(chain(*hitlists), key=lambda x: x.date):
        #     sentence_id = hit.philo_id[:6]
        #     if sentence_id not in sentence_ids:
        #         self.combined_hitlist.append(hit)
        #         sentence_ids.add(sentence_id)
        from collections import defaultdict

        sentence_counts = defaultdict(int)
        for pos, hitlist in enumerate(hitlists):
            max_sent_count = 2
            for hit in hitlist:
                sentence_id = repr(hit.philo_id[:6])
                if sentence_id not in sentence_counts or sentence_counts[sentence_id] == max_sent_count:
                    self.combined_hitlist.append(hit)
                    sentence_counts[sentence_id] += 1

        self.done = True

    def __len__(self):
        return len(self.combined_hitlist)

    def __getitem__(self, key):
        return self.combined_hitlist[key]

    def __getattr__(self, name):
        return self.combined_hitlist[name]


class WordPropertyHitlist(object):
    def __init__(self, hitlist):
        self.done = True
        self.hitlist = hitlist

    def __getitem__(self, key):
        return self.hitlist[key]

    def __getattr__(self, name):
        return self.hitlist[name]

    def __len__(self):
        return len(self.hitlist)


class NoHits(object):
    def __init__(self):
        self.done = True

    def __len__(self):
        return 0

    def __getitem__(self, item):
        return ""

    def __iter__(self):
        yield ""

    def finish(self):
        return

    def update(self):
        return

    def get_total_word_count(self):
        return 0
