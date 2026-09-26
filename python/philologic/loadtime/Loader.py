#!/var/lib/philologic5/philologic_env/bin/python3
"""Standard PhiloLogic5 loader.
Calls all parsing functions and stores data in index"""

import array
import collections
import datetime
import hashlib
import heapq
import os
import pickle
import shutil
import sqlite3
import struct
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, as_completed, wait
from glob import iglob
from json import dump
from types import FunctionType

import dill
import lmdb
import lxml.etree
import lz4.frame
import numpy as np
import pandas as pd
import regex as re
from black import FileMode, format_str
from orjson import loads
from tqdm import tqdm

from philologic.Config import MakeDBConfig, MakeWebConfig
from philologic.loadtime.PostFilters import (
    frequency_file_key,
    make_collocation_database,
    make_sql_table,
    open_lz4_lines,
    write_lemma_frequencies,
    write_unique_word_attributes,
)
from philologic.utils import (
    convert_entities,
    count_lines,
    extract_full_date,
    extract_integer,
    load_module,
    pretty_print,
    process_pool,
    run_shell,
    shared_value,
    sort_list,
    start_worker_server,
    thread_pool,
)

SORT_BY_WORD = "-k 2,2"
SORT_BY_ID = "-k 3,3n -k 4,4n -k 5,5n -k 6,6n -k 7,7n -k 8,8n -k 9,9n"
OBJECT_TYPES = ["doc", "div1", "div2", "div3", "para", "sent", "word"]

BLOCKSIZE = 2048  # index block size.  Don't alter.
INDEX_CUTOFF = 10  # index frequency cutoff.  Don't alter.

DEFAULT_TABLES = ("toms", "pages", "refs", "graphics", "lines")

DEFAULT_OBJECT_LEVEL = "doc"

NAVIGABLE_OBJECTS = ("doc", "div1", "div2", "div3", "para")

ASCII_CONVERSION = True

PARSER_OPTIONS = [
    "parser_factory",
    "doc_xpaths",
    "token_regex",
    "tag_to_obj_map",
    "metadata_to_parse",
    "suppress_tags",
    "load_filters",
    "break_apost",
    "chars_not_to_index",
    "break_sent_in_line_group",
    "tag_exceptions",
    "join_hyphen_in_words",
    "abbrev_expand",
    "long_word_limit",
    "flatten_ligatures",
    "sentence_breakers",
    "file_type",
    "metadata_sql_types",
    "lowercase_index",
]


class ParserError(Exception):
    """Parser exception"""

    def __init___(self, *error_args):
        super().__init__(error_args)


# Stored hit layout: sentence philo_id (first 6 ints), then page (pos_8), word position (pos_6) and byte offset (pos_7)
HIT_COLUMN_ORDER = [0, 1, 2, 3, 4, 5, 8, 6, 7]
DIGITS_AND_SPACE = b"0123456789 "
PHILO_ID_PACK_CHUNK = 100000  # philo_ids held before packing them, bounding the memory used on top of the packed bytes


def pack_philo_id(philo_id):
    """Pack a philo_id (9 space-separated integers) into the 36 bytes stored in the index"""
    pos_0, pos_1, pos_2, pos_3, pos_4, pos_5, pos_6, pos_7, pos_8 = map(int, philo_id.split())
    return struct.pack("9I", pos_0, pos_1, pos_2, pos_3, pos_4, pos_5, pos_8, pos_6, pos_7)


def pack_philo_ids(philo_ids):
    """Pack a list of philo_ids (bytes) into the index format.
    Same bytes as b"".join(pack_philo_id(philo_id.decode("utf-8")) for philo_id in philo_ids),
    but the integers of longer lists are parsed all at once."""
    if len(philo_ids) >= 16 and all(philo_id.count(b" ") == 8 for philo_id in philo_ids):
        joined = b" ".join(philo_ids)
        # Only digits and single spaces, 9 integers per philo_id, all fitting in a uint32
        if not joined.translate(None, DIGITS_AND_SPACE):
            try:
                ids = np.fromstring(joined, dtype=np.int64, sep=" ")
            except ValueError:
                ids = None
            if ids is not None and ids.size == 9 * len(philo_ids) and ids.max() <= 0xFFFFFFFF:
                return ids.astype(np.uint32).reshape(-1, 9)[:, HIT_COLUMN_ORDER].tobytes()
    # Short lists or unusual philo_ids: one by one, raising the same errors as before
    return b"".join([pack_philo_id(philo_id.decode("utf-8")) for philo_id in philo_ids])


LEMMA_LOOKUP_BATCH = 20000000  # lemma lookup entries sorted in memory at once (about 2 GB)
LEMMA_LOOKUP_COMMIT = 500000  # lemma lookup entries stored per transaction


def sort_lemma_lookups(keys, positions):
    """Sort lemma lookup keys (a bytes-like object of 36-byte keys) with the positions of their values, in the byte
    order of LMDB keys. Of equal keys (a philo_id found twice), only the last one is kept, as it would overwrite the
    others."""
    keys = np.frombuffer(keys, dtype="S36")  # fixed-size strings: compared byte by byte, as LMDB keys
    order = np.argsort(keys, kind="stable")
    keys, positions = keys[order], positions[order]
    last = np.ones(len(keys), dtype=bool)
    last[:-1] = keys[1:] != keys[:-1]
    if last.all():
        return keys, positions
    return keys[last], positions[last]


def merge_lemma_lookup_runs(runs, chunk_size=LEMMA_LOOKUP_COMMIT):
    """Merge sorted runs of lemma lookups (paths of the .npy files of their keys and value positions), yielding
    (keys, positions) in key order, chunk by chunk. Of equal keys, the one of the last run is kept."""
    keys = [np.load(keys_path, mmap_mode="r") for keys_path, _ in runs]
    positions = [np.load(positions_path, mmap_mode="r") for _, positions_path in runs]
    starts = [0] * len(runs)
    while any(start < len(run_keys) for start, run_keys in zip(starts, keys)):
        heads = [run_keys[start : start + chunk_size] for start, run_keys in zip(starts, keys)]
        # Keys can be output up to the smallest last key of a head with more keys after it (excluded: its equals
        # in other runs may not be in their heads)
        bounds = [head[-1] for head, start, run_keys in zip(heads, starts, keys) if start + len(head) < len(run_keys)]
        taken_keys, taken_positions = [], []
        for run, head in enumerate(heads):
            taken = len(head) if not bounds else np.searchsorted(head, min(bounds), side="left")
            taken_keys.append(head[:taken])
            taken_positions.append(positions[run][starts[run] : starts[run] + taken])
            starts[run] += taken
        # Runs are concatenated in file order, so the stable sort keeps equal keys in file order too
        chunk_keys, chunk_positions = np.concatenate(taken_keys), np.concatenate(taken_positions)
        yield sort_lemma_lookups(chunk_keys.tobytes(), chunk_positions)


def build_lemma_lookup_index(workdir, destination, lemma_count):
    """Create a lemma lookup table where keys are philo_ids as bytes and values are lemmas in the form lemma:word.
    Only reads the sorted lemmas file and writes its own database, so it runs alongside the rest of the index build.
    Entries are sorted by key (in batches, merged afterwards, for large corpora) and stored in that order: this fills
    the database pages, without having to compact it."""
    print(f"{time.ctime()}: Creating lemma lookup index...", flush=True)
    values = []  # lemma:word values, in file order
    keys = bytearray()  # keys of the batch being read, 36 bytes each
    positions = array.array("I")  # positions of their values
    runs = []  # sorted batches saved to disk
    count = 0

    def save_run():
        nonlocal keys, positions
        run_keys, run_positions = sort_lemma_lookups(keys, np.frombuffer(positions, dtype=np.uint32))
        run = (f"{workdir}/lemma_lookup_keys_{len(runs)}.npy", f"{workdir}/lemma_lookup_positions_{len(runs)}.npy")
        np.save(run[0], run_keys)
        np.save(run[1], run_positions)
        runs.append(run)
        keys, positions = bytearray(), array.array("I")

    def store_lemma_lookups(philo_ids):
        nonlocal count
        packed = pack_philo_ids(philo_ids)
        keys.extend(packed)
        positions.extend([len(values) - 1] * (len(packed) // 36))
        count += len(packed) // 36

    with open_lz4_lines(f"{workdir}/all_lemmas_sorted.lz4") as input_file:
        current_lemma = None
        philo_ids = []
        for line in input_file:  # no progress bar: this runs alongside the other ones
            _, word, philo_id, _ = line.strip().split(b"\t")
            if word != current_lemma or len(philo_ids) == PHILO_ID_PACK_CHUNK:
                if current_lemma is not None:
                    store_lemma_lookups(philo_ids)
                if len(positions) >= LEMMA_LOOKUP_BATCH:
                    save_run()
                if word != current_lemma:
                    values.append(b"lemma:" + word)
                current_lemma = word
                philo_ids = []
            philo_ids.append(philo_id)
        if current_lemma is not None:
            store_lemma_lookups(philo_ids)
    if runs:  # several batches: merge them
        save_run()
        sorted_chunks = merge_lemma_lookup_runs(runs)
    else:
        sorted_chunks = [sort_lemma_lookups(keys, np.frombuffer(positions, dtype=np.uint32))]
    del keys, positions

    lemma_db_env = lmdb.open(f"{destination}/lemmas.lmdb", map_size=2 * 1024 * 1024 * 1024 * 1024, sync=False)
    for chunk_keys, chunk_positions in sorted_chunks:
        chunk_keys = chunk_keys.tobytes()
        for start in range(0, len(chunk_positions), LEMMA_LOOKUP_COMMIT):
            with lemma_db_env.begin(write=True) as lemma_txn:
                lemma_txn.cursor().putmulti(
                    (
                        (chunk_keys[position * 36 : position * 36 + 36], values[chunk_positions[position]])
                        for position in range(start, min(start + LEMMA_LOOKUP_COMMIT, len(chunk_positions)))
                    ),
                    append=True,
                )
    lemma_db_env.sync(True)
    lemma_db_env.close()
    for run in runs:
        for path in run:
            os.remove(path)
    print(f"{time.ctime()}: Stored {count} lemma lookup entries.", flush=True)

OVERFLOW_LIMIT = 360000000  # 36 bytes per philo_id, 10,000,000 philo_ids: more go to an overflow file
PROGRESS_INTERVAL = 100000  # lines read by an index worker between progress reports
index_progress = None  # count of the lines read by index workers, set up by init_index_worker


def init_index_worker(progress):
    """Give an index worker the count of lines read by all of them (a shared_value), for the progress bar"""
    global index_progress
    index_progress = progress


def report_progress(lines):
    """Add lines to the count of lines read by index workers"""
    if index_progress is not None:
        with index_progress.get_lock():
            index_progress.value += lines


def open_index(path):
    """Open a new LMDB database to store index entries in"""
    return lmdb.open(path, map_size=2 * 1024 * 1024 * 1024 * 1024, writemap=True, sync=False)  # 2TB limit


def write_overflow_file(overflow_dir, key, philo_ids):
    """Write the philo_ids of a key to a binary file, as they would overflow the limit for LMDB values"""
    filename = f'{hashlib.sha256(key.encode("utf-8")).hexdigest()}.bin'
    with open(os.path.join(overflow_dir, filename), "wb") as overflow_file:
        overflow_file.write(philo_ids)


def index_words(words_file, index_path, overflow_dir, has_attributes, attributes_to_skip, commit_interval):
    """Store the philo_ids of each word of a sorted words file in a new LMDB database, under the word.
    Returns the number of entries, the keys written to overflow files instead, and whether words have attributes
    other than attributes_to_skip (checked unless has_attributes is already True)."""
    db_env = open_index(index_path)
    overflow_keys = []
    # Lines are handled as bytes: the files are UTF-8, so splitting and comparing bytes gives the same results
    # as on decoded strings, and keys are encoded back to the same bytes.
    with open_lz4_lines(words_file) as input_file:
        current_word = None
        count = 0
        line_number = 0
        philo_ids = []  # philo_ids (bytes) not packed yet
        packed_philo_ids = bytearray()
        txn = db_env.begin(write=True)
        for line_number, line in enumerate(input_file, 1):
            _, word, philo_id, attribs = line.split(b"\t", 3)
            if not has_attributes:  # stop checking once we know
                if any(k not in attributes_to_skip for k in loads(attribs)):
                    has_attributes = True
            if word != current_word:
                if current_word is not None:
                    packed_philo_ids += pack_philo_ids(philo_ids)
                    if len(packed_philo_ids) > OVERFLOW_LIMIT:
                        write_overflow_file(overflow_dir, current_word.decode("utf-8"), packed_philo_ids)
                        overflow_keys.append(current_word.decode("utf-8"))
                    else:
                        txn.put(current_word, packed_philo_ids)
                    count += 1
                    if count % commit_interval == 0:
                        txn.commit()
                        txn = db_env.begin(write=True)
                current_word = word
                philo_ids = []
                packed_philo_ids = bytearray()
            philo_ids.append(philo_id)
            if len(philo_ids) == PHILO_ID_PACK_CHUNK:
                packed_philo_ids += pack_philo_ids(philo_ids)
                philo_ids = []
            if line_number % PROGRESS_INTERVAL == 0:
                report_progress(PROGRESS_INTERVAL)

        # Commit any remaining words
        packed_philo_ids += pack_philo_ids(philo_ids)
        if packed_philo_ids:
            if len(packed_philo_ids) > OVERFLOW_LIMIT:
                write_overflow_file(overflow_dir, current_word.decode("utf-8"), packed_philo_ids)
                overflow_keys.append(current_word.decode("utf-8"))
            else:
                txn.put(current_word, packed_philo_ids)
            count += 1
        txn.commit()
    report_progress(line_number % PROGRESS_INTERVAL)
    db_env.close()
    return count, overflow_keys, has_attributes


def index_lemmas(lemmas_file, index_path, overflow_dir, commit_interval):
    """Store the philo_ids of each lemma of a sorted lemmas file in a new LMDB database, under lemma:{lemma}.
    Returns the number of entries and the keys written to overflow files instead."""
    db_env = open_index(index_path)
    overflow_keys = []
    with open_lz4_lines(lemmas_file) as input_file:
        txn = db_env.begin(write=True)
        current_lemma = None
        count = 0
        line_number = 0
        philo_ids = []
        packed_philo_ids = bytearray()
        for line_number, line in enumerate(input_file, 1):
            _, lemma, philo_id, _ = line.strip().split(b"\t")
            if lemma != current_lemma:
                if current_lemma is not None:
                    packed_philo_ids += pack_philo_ids(philo_ids)
                    if len(packed_philo_ids) > OVERFLOW_LIMIT:
                        write_overflow_file(overflow_dir, f"lemma:{current_lemma.decode('utf-8')}", packed_philo_ids)
                        overflow_keys.append(f"lemma:{current_lemma.decode('utf-8')}")
                    else:
                        txn.put(b"lemma:" + current_lemma, packed_philo_ids)
                    count += 1
                    if count % commit_interval == 0:
                        txn.commit()
                        txn = db_env.begin(write=True)
                current_lemma = lemma
                philo_ids = []
                packed_philo_ids = bytearray()
            philo_ids.append(philo_id)
            if len(philo_ids) == PHILO_ID_PACK_CHUNK:
                packed_philo_ids += pack_philo_ids(philo_ids)
                philo_ids = []
            if line_number % PROGRESS_INTERVAL == 0:
                report_progress(PROGRESS_INTERVAL)
        # Commit any remaining lemmas
        packed_philo_ids += pack_philo_ids(philo_ids)
        if packed_philo_ids:
            if len(packed_philo_ids) > OVERFLOW_LIMIT:
                write_overflow_file(overflow_dir, f"lemma:{current_lemma.decode('utf-8')}", packed_philo_ids)
                overflow_keys.append(f"lemma:{current_lemma.decode('utf-8')}")
            else:
                txn.put(b"lemma:" + current_lemma, packed_philo_ids)
            count += 1
        txn.commit()
    report_progress(line_number % PROGRESS_INTERVAL)
    db_env.close()
    return count, overflow_keys


def index_word_attributes(
    sorted_file, index_path, overflow_dir, key_prefix, attributes_to_skip, commit_interval, collect_attribute_names
):
    """Store the philo_ids of each word (or lemma) and attribute value found in a sorted words (or lemmas) file in a
    new LMDB database, under {key_prefix}{word}:{attribute}:{value} keys. Returns the number of entries, the keys
    written to overflow files instead, and if collect_attribute_names, the names of all attributes in the file."""
    db_env = open_index(index_path)
    overflow_keys = []
    attribute_names = set() if collect_attribute_names else None
    count = 0
    txn = db_env.begin(write=True)

    def store(word, word_attributes, packed_attributes):
        nonlocal count, txn
        word_string = word.decode("utf-8")
        for attribute, attribute_dict in word_attributes.items():
            for attribute_value, philo_ids in attribute_dict.items():
                key = f"{key_prefix}{word_string}:{attribute}:{attribute_value}"
                packed_philo_ids = pack_philo_ids(philo_ids)
                if (attribute, attribute_value) in packed_attributes:
                    packed_philo_ids = bytes(packed_attributes[(attribute, attribute_value)]) + packed_philo_ids
                if len(packed_philo_ids) > OVERFLOW_LIMIT:
                    write_overflow_file(overflow_dir, key, packed_philo_ids)
                    overflow_keys.append(key)
                else:
                    txn.put(key.encode("utf-8"), packed_philo_ids)
                count += 1
                if count % commit_interval == 0:
                    txn.commit()
                    txn = db_env.begin(write=True)

    with open_lz4_lines(sorted_file) as input_file:
        word_attributes = {}  # attribute -> attribute value -> philo_ids (bytes) not packed yet
        packed_attributes = {}  # (attribute, attribute value) -> philo_ids already packed, for long words
        lines_in_word = 0
        line_number = 0
        current_word = None
        for line_number, line in enumerate(input_file, 1):
            _, word, philo_id, attributes = line.split(b"\t", 3)
            attributes = loads(attributes)
            if attribute_names is not None:
                attribute_names.update(attributes)
            if word != current_word:
                if current_word is not None:
                    store(current_word, word_attributes, packed_attributes)
                current_word = word
                word_attributes = {}
                packed_attributes = {}
                lines_in_word = 0
            for attribute, attribute_value in attributes.items():
                if attribute in attributes_to_skip:
                    continue
                if attribute not in word_attributes:
                    word_attributes[attribute] = defaultdict(list)
                word_attributes[attribute][attribute_value].append(philo_id)
            lines_in_word += 1
            if lines_in_word % PHILO_ID_PACK_CHUNK == 0:  # very frequent word: pack what we have so far
                for attribute, attribute_dict in word_attributes.items():
                    for attribute_value, philo_ids in attribute_dict.items():
                        if philo_ids:
                            packed = packed_attributes.setdefault((attribute, attribute_value), bytearray())
                            packed += pack_philo_ids(philo_ids)
                            philo_ids.clear()
            if line_number % PROGRESS_INTERVAL == 0:
                report_progress(PROGRESS_INTERVAL)
        # Handle the last set of words
        if current_word is not None:
            store(current_word, word_attributes, packed_attributes)
    txn.commit()
    report_progress(line_number % PROGRESS_INTERVAL)
    db_env.close()
    return count, overflow_keys, attribute_names


# Bytes stored per transaction when merging index parts, plus one more value of at most OVERFLOW_LIMIT bytes:
# without a writemap, a transaction keeps the pages it writes in memory, up to a limit
MERGE_COMMIT_BYTES = 64 * 1024 * 1024


def merge_indexes(index_paths, merged_path):
    """Store the entries of several LMDB databases in a new one, in key order. A key found in several of them gets
    its value in the last one, as when they were all built in a single database, one after the other.
    Stored in key order, the entries fill the new database's pages: it doesn't need compacting."""
    envs = [lmdb.open(path, readonly=True, lock=False) for path in index_paths]
    txns = [env.begin() for env in envs]
    merged_env = lmdb.open(merged_path, map_size=2 * 1024 * 1024 * 1024 * 1024, sync=False)
    entries = heapq.merge(
        *(((key, -position, value) for key, value in txn.cursor()) for position, txn in enumerate(txns))
    )
    txn = merged_env.begin(write=True)
    previous_key = None
    uncommitted_bytes = 0
    for key, _, value in entries:
        if key == previous_key:  # already stored, with its value in a later database
            continue
        if uncommitted_bytes + len(value) > MERGE_COMMIT_BYTES and uncommitted_bytes:
            txn.commit()
            txn = merged_env.begin(write=True)
            uncommitted_bytes = 0
        txn.put(key, value, append=True)
        uncommitted_bytes += len(key) + len(value)
        previous_key = key
    txn.commit()
    for read_txn, env in zip(txns, envs):
        read_txn.abort()
        env.close()
    merged_env.sync(True)
    merged_env.close()

# Loader class attributes which parse workers don't get: they are sent the files to parse one at a time, and don't use
# the spaCy model (with one, files are parsed in the loading process)
ATTRIBUTES_NOT_SENT_TO_WORKERS = {"filequeue", "data_dicts", "nlp"}
worker_loader = None  # Loader class of a parse worker, set up by init_parse_worker


def pickle_loader_state(loader_class):
    """Pickle a Loader class and its class attributes (set by set_class_attributes, set_file_data...), so that
    parse workers get the same state as the loading process. Attributes pickle can't handle (closures in load filters,
    classes and functions of load configs...) are pickled by dill, by value."""
    attributes = {}
    for klass in reversed(loader_class.__mro__[: loader_class.__mro__.index(Loader) + 1]):
        for name, value in vars(klass).items():
            if name.startswith("__") or name in ATTRIBUTES_NOT_SENT_TO_WORKERS:
                continue
            if not isinstance(value, (FunctionType, classmethod, staticmethod, property)):
                attributes[name] = value
    pickled_attributes = {}
    for name, value in attributes.items():
        try:
            pickled_attributes[name] = pickle.dumps(value)
        except (pickle.PicklingError, AttributeError, TypeError):
            pickled_attributes[name] = dill.dumps(value)
    return dill.dumps(loader_class), pickled_attributes


def init_parse_worker(pickled_class, pickled_attributes, workers):
    """Give the Loader class of a parse worker the state pickled by pickle_loader_state, and its own spaCy model"""
    global worker_loader
    worker_loader = dill.loads(pickled_class)
    for name, pickled_value in pickled_attributes.items():
        setattr(worker_loader, name, dill.loads(pickled_value))
    if worker_loader.spacy_model:
        import spacy

        worker_loader.nlp = spacy.load(worker_loader.spacy_model, disable=["tokenizer"])
        if "torch" in sys.modules:  # share out the threads torch would use in a single process among the workers
            import torch

            torch.set_num_threads(max(1, torch.get_num_threads() // workers))


def parse_in_worker(text, metadata):
    """Parse a file in a parse worker. Returns the number of lines of its words and lemmas files."""
    worker_loader.parse_text(text, metadata)
    return text["word_lines"], text["lemma_lines"]


def count_newlines(path, lz4_compressed=False):
    """Number of lines of a file, counted as wc -l does"""
    with (lz4.frame.open if lz4_compressed else open)(path, "rb") as input_file:
        return sum(block.count(b"\n") for block in iter(lambda: input_file.read(1 << 24), b""))


class Loader:
    """Loader class"""

    sort_by_word = SORT_BY_WORD
    sort_by_id = SORT_BY_ID
    types = OBJECT_TYPES
    tables = DEFAULT_TABLES
    omax = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    parser_config = {}
    words_to_index = set()
    data_dicts = []
    filequeue = []
    raw_files = []
    textdir = ""
    workdir = ""
    web_app_dir = ""
    metadata_fields = []
    metadata_types = {}
    metadata_hierarchy = []
    metadata_fields_not_found = []
    debug = False
    default_object_level = "doc"
    post_filters = []
    token_regex = ""
    url_root = ""
    cores = 2
    ascii_conversion = ASCII_CONVERSION
    lemmas = None
    attributes_to_skip = {
        "start_byte",
        "end_byte",
        "doc_ancestor",
        "div1_ancestor",
        "div2_ancestor",
        "div3_ancestor",
        "para_ancestor",
        "parent",
        "page",
        "lemma",
    }
    word_count = 0
    lemma_count = 0
    precomputed_files = {}  # frequency file key -> file written while building the index, see frequency_file_key
    parsed_line_counts = None  # [lines of the words files, of the lemmas files] of the files parsed by parse_files
    has_attributes = False
    nlp = None
    spacy_model = None  # spaCy model loaded by each parse worker, when not running on the GPU
    suppress_word_attributes = set()
    word_attributes = []
    overflow_words = set()  # words which would overflow the limit for LMDB values
    # (words file path, size, mtime, names of all attributes in its lines), as found by build_inverted_index
    all_word_attribute_names = None

    @classmethod
    def set_class_attributes(cls, loader_options):
        """Set initial class attributes and return Loader object"""
        start_worker_server(["philologic.loadtime.Loader"])
        cls.all_word_attribute_names = None
        cls.post_filters = list(loader_options["post_filters"])
        cls.debug = loader_options["debug"]
        cls.words_to_index = loader_options["words_to_index"]
        cls.destination = loader_options["data_destination"]
        cls.workdir = os.path.join(loader_options["data_destination"], "WORK/")
        cls.textdir = os.path.join(loader_options["data_destination"], "TEXT/")
        cls.web_app_dir = os.path.join(loader_options["db_destination"], "app/")
        cls.debug = loader_options["debug"]
        cls.default_object_level = loader_options["default_object_level"]
        cls.token_regex = loader_options["token_regex"]
        cls.url_root = loader_options["url_root"]
        cls.cores = loader_options["cores"]
        cls.ascii_conversion = loader_options["ascii_conversion"]
        cls.metadata_sql_types = loader_options["metadata_sql_types"]
        if loader_options["lemma_file"] is not None:
            cls.lemmas = {}
            lowercase_lemma_keys = loader_options.get("lowercase_index", True)
            with open(loader_options["lemma_file"], encoding="utf8") as lemma_file:
                for line in lemma_file:
                    word, lemma = line.strip().split("\t")
                    if lowercase_lemma_keys:
                        word = word.lower()
                    cls.lemmas[word] = lemma
        for option in PARSER_OPTIONS:
            try:
                cls.parser_config[option] = loader_options[option]
            except KeyError:  # option hasn't been set
                pass
        cls.spacy_model = None
        if loader_options["spacy_model"]:
            import spacy  # only imported when used: importing it takes about a second

            if spacy.prefer_gpu():  # files are then tagged in this process only, see parse_files
                cls.nlp = spacy.load(loader_options["spacy_model"], disable=["tokenizer"])
            else:
                cls.spacy_model = loader_options["spacy_model"]
        cls.suppress_word_attributes = set(loader_options["suppress_word_attributes"])
        return cls(**loader_options)

    def __init__(self, **loader_options):
        os.system(f"mkdir -p {self.destination}")
        os.mkdir(self.workdir)
        os.mkdir(self.textdir)

        load_config_path = os.path.join(loader_options["data_destination"], "load_config.py")
        # Loading these from a load_config would crash the parser for a number of reasons...
        values_to_ignore = [
            "load_filters",
            "post_filters",
            "parser_factory",
            "data_destination",
            "db_destination",
            "dbname",
        ]
        if loader_options["load_config"]:
            shutil.copy(loader_options["load_config"], load_config_path)
            config_obj = load_module("external_load_config", loader_options["load_config"])
            already_configured_values = {}
            for attribute in dir(config_obj):
                if not attribute.startswith("__") and not isinstance(
                    getattr(config_obj, attribute), collections.abc.Callable
                ):
                    already_configured_values[attribute] = getattr(config_obj, attribute)
            with open(load_config_path, "a", encoding="utf8") as load_config_copy:
                print(
                    "\n\n## The values below were also used for loading ##",
                    file=load_config_copy,
                )
                for option, option_value in loader_options.items():
                    if (
                        option not in already_configured_values
                        and option not in values_to_ignore
                        and option != "web_config"
                    ):
                        print(
                            "%s = %s\n" % (option, repr(option_value)),
                            file=load_config_copy,
                        )
        else:
            with open(load_config_path, "w", encoding="utf8") as load_config_copy:
                print("#!/var/lib/philologic5/philologic_env/bin/python3", file=load_config_copy)
                print(
                    '"""This is a dump of the default configuration used to load this database,',
                    file=load_config_copy,
                )
                print(
                    "including non-configurable options. You can use this file to reload",
                    file=load_config_copy,
                )
                print(
                    'the current database using the -l flag. See load documentation for more details"""\n\n',
                    file=load_config_copy,
                )
                for option, option_value in loader_options.items():
                    if option not in values_to_ignore and option != "web_config":
                        print(
                            "%s = %s\n" % (option, repr(option_value)),
                            file=load_config_copy,
                        )

        if "web_config" in loader_options:
            web_config_path = os.path.join(loader_options["data_destination"], "web_config.cfg")
            print("\nSaving predefined web_config.cfg file to %s..." % web_config_path)
            with open(web_config_path, "w", encoding="utf8") as w:
                w.write(loader_options["web_config"])
            self.predefined_web_config = True
        else:
            self.predefined_web_config = False

        self.filenames = []
        self.raw_files = []
        self.deleted_files = []
        self.metadata_fields = []
        self.metadata_hierarchy = []
        self.metadata_types = {}
        self.normalized_fields = []
        self.metadata_fields_not_found = []
        self.sort_order = ""

    def add_files(self, files):
        """Copy files to database directory"""
        for f in tqdm(
            files,
            total=len(files),
            leave=False,
            desc="Copying files to database directory",
        ):
            new_file_path = os.path.join(self.textdir, os.path.basename(f).replace(" ", "_").replace("'", "_"))
            shutil.copy2(f, new_file_path)
            os.chmod(new_file_path, 775)
            self.filenames.append(f)
        os.system(f"chmod -R 775 {self.textdir}")
        print("Copying files to database directory... done.", flush=True)

    def parse_bibliography_file(self, bibliography_file, sort_by_field):
        """Parse tab delimited bibliography file: tsv, tab, or csv"""

        # Detect delimiter based on file extension
        if bibliography_file.endswith(".tab") or bibliography_file.endswith(".tsv"):
            delimiter = "\t"
        else:
            delimiter = ","

        try:
            df = pd.read_csv(
                bibliography_file,
                delimiter=delimiter,
                encoding="utf8",
                dtype=str,  # Read all as strings initially to match current behavior
                keep_default_na=False,  # Don't convert empty strings to NaN
                skipinitialspace=True,  # Strip leading whitespace from column names and values
            )
        except FileNotFoundError:
            print(f"Error: Bibliography file not found: {bibliography_file}")
            sys.exit(1)
        except pd.errors.EmptyDataError:
            print(f"Error: Bibliography file is empty: {bibliography_file}")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading bibliography file {bibliography_file}: {e}")
            sys.exit(1)

        # Strip whitespace from column names to match csv.DictReader behavior
        df.columns = df.columns.str.strip()

        # Strip whitespace from all string values to match csv.DictReader behavior.
        # Use is_string_dtype so this keeps working when pandas 3.0 makes StringDtype
        # (rather than object) the default for string columns.
        df = df.apply(lambda x: x.str.strip() if pd.api.types.is_string_dtype(x) else x)

        # Validate that required 'filename' column exists
        if "filename" not in df.columns:
            print("Error: Bibliography file must contain a 'filename' column")
            print(f"Available columns: {', '.join(df.columns)}")
            sys.exit(1)

        # Convert year column to int if present, otherwise keep as string
        if "year" in df.columns:
            # Convert year to int, replacing empty/invalid values with 0
            df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)

        if self.debug:
            print(f"Found {len(df)} records in bibliography file")
            print(f"Columns: {', '.join(df.columns)}")

        # Convert DataFrame to list of dictionaries
        load_metadata = df.to_dict("records")

        # Process year field for each record only if year column doesn't exist
        for metadata in load_metadata:
            if "year" not in metadata or not metadata["year"]:
                metadata = self.create_year_field(metadata)
            if "year" not in metadata or not metadata["year"]:
                metadata["year"] = 0

        # Sort metadata
        print(
            "Sorting files by the following metadata fields: %s..." % ", ".join([i for i in sort_by_field]),
            end=" ",
        )

        self.sort_order = sort_by_field  # to be used for the sort by concordance biblio key in web config
        load_metadata = sort_list(load_metadata, sort_by_field)
        print("done.")

        return load_metadata

    def parse_tei_header(self, verbose):
        """Parse header in TEI files"""
        load_metadata = []
        deleted_files_error_cause = []
        metadata_xpaths = self.parser_config["doc_xpaths"]
        doc_count = len(os.listdir(self.textdir))
        if verbose:
            prefix = f"{time.ctime()}: Parsing document level metadata"
        else:
            prefix = "Parsing document level metadata"
        for file in tqdm(
            os.scandir(self.textdir),
            total=doc_count,
            desc=prefix,
            leave=False,
        ):
            data = {"filename": file.name}
            header = ""
            with open(file.path, encoding="utf8") as text_file:
                try:
                    file_content = "".join(text_file.readlines())
                except UnicodeDecodeError:
                    self.deleted_files.append(file.name)
                    deleted_files_error_cause.append((file.name, "invalid characters"))
                    continue
            try:
                start_header_index = re.search(r"<teiheader", file_content, re.I).start()
                end_header_index = re.search(r"</teiheader", file_content, re.I).start()
            except AttributeError:  # tag not found
                if self.debug:
                    print(f"File {file.name} contains no TEI header and will be skipped.")
                self.deleted_files.append(file.name)
                deleted_files_error_cause.append((file.name, "no TEI header"))
                continue
            header = file_content[start_header_index:end_header_index]
            header = convert_entities(header)
            if self.debug:
                print("parsing %s header..." % file.name)
            parser = lxml.etree.XMLParser(recover=True)
            try:
                tree = lxml.etree.fromstring(header, parser)
                trimmed_metadata_xpaths = []
                for field in metadata_xpaths:
                    for xpath in metadata_xpaths[field]:
                        xpath = xpath.rstrip("/")  # make sure there are no trailing slashes which make lxml die
                        try:
                            elements = tree.xpath(xpath)
                        except lxml.etree.XPathEvalError:
                            continue
                        for element in elements:
                            if element is not None:
                                value = ""
                                if isinstance(element, lxml.etree._Element) and element.text is not None:
                                    value = element.text.strip()
                                elif isinstance(element, lxml.etree._ElementUnicodeResult):
                                    value = str(element).strip()
                                if value:
                                    if field not in self.parser_config["metadata_sql_types"]:
                                        data[field] = value
                                        if (
                                            field in ("create_date", "pub_date") and re.search(r"\d", value) is None
                                        ):  # make sure we have a number in there
                                            del data[field]
                                            continue
                                    elif self.parser_config["metadata_sql_types"][field] == "int":
                                        data[field] = extract_integer(value)
                                    elif self.parser_config["metadata_sql_types"][field] == "date":
                                        data[field] = extract_full_date(value)
                                    break
                        else:  # only continue looping over xpaths if no break in inner loop
                            continue
                        break
                trimmed_metadata_xpaths = [
                    (metadata_type, xpath, field)
                    for metadata_type in ["div", "para", "sent", "word", "page"]
                    if metadata_type in metadata_xpaths
                    for field in metadata_xpaths[metadata_type]
                    for xpath in metadata_xpaths[metadata_type][field]
                ]
                data = self.create_year_field(data)
                if self.debug:
                    print(pretty_print(data))
                data["options"] = {"metadata_xpaths": trimmed_metadata_xpaths}
                load_metadata.append(data)
            except lxml.etree.XMLSyntaxError:
                self.deleted_files.append(file.name)
                deleted_files_error_cause.append((file.name, "invalid XML"))
        print(f"{prefix}... done.", flush=True)
        if self.deleted_files:
            print("\nThe following files have been removed from the load:")
            for filename, cause in deleted_files_error_cause:
                print(f"File {filename}: {cause}")
            print()
        return load_metadata

    def parse_dc_header(self):
        """Parse Dublin Core header"""
        load_metadata = []
        doc_count = len(os.listdir(self.textdir))
        prefix = f"{time.ctime()}: Parsing document level metadata"
        for file in tqdm(os.scandir(self.textdir), total=doc_count, leave=False, desc=prefix):
            data = {}
            header = ""
            with open(file.path, encoding="utf8") as fh:
                for line in fh:
                    start_scan = re.search(r"<teiheader>|<temphead>|<head>", line, re.IGNORECASE)
                    end_scan = re.search(r"</teiheader>|<\/?temphead>|</head>", line, re.IGNORECASE)
                    if start_scan:
                        header += line[start_scan.start() :]
                    elif end_scan:
                        header += line[: end_scan.end()]
                        break
                    else:
                        header += line
            matches = re.findall(r'<meta name="DC\.([^"]+)" content="([^"]+)"', header)
            if not matches:
                matches = re.findall(r"<dc:([^>]+)>([^>]+)>", header)
            for metadata_name, metadata_value in matches:
                metadata_value = convert_entities(metadata_value)
                metadata_name = metadata_name.lower()
                data[metadata_name] = metadata_value
            data["filename"] = file.name  # place at the end in case the value was in the header
            data = self.create_year_field(data)
            if self.debug:
                print(pretty_print(data))
            load_metadata.append(data)
        print(f"{prefix}... done.", flush=True)
        return load_metadata

    def create_year_field(self, metadata):
        """Create year field from date fields in header"""
        # Matches a year with an optional single leading minus for BCE dates (ISO 8601/TEI).
        # A single hyphen before digits = negative year (e.g. "-0044" for 44 BC).
        # Double hyphens ("--") indicate a partial date with unknown year (W3C "--mm-dd" format)
        # or junk delimiters (e.g. "--1798--"), so we skip those.
        year_finder = re.compile(r"(?<!\-)(\-?)(\d{1,})")
        earliest_year = float("inf")
        metadata_with_year = ""
        for field in ["date", "create_date", "pub_date", "period"]:
            if field in metadata:
                if isinstance(metadata[field], datetime.date):
                    metadata[field] = str(metadata[field].year)
                elif isinstance(metadata[field], int):
                    metadata[field] = str(metadata[field])
                year_match = year_finder.search(metadata[field])
                if year_match:
                    sign, digits = year_match.groups()
                    year = int(f"{sign}{digits}")
                    metadata_with_year = field
                    if field == "create_date":  # this should be the canonical date
                        earliest_year = year
                        break
                    if year < earliest_year:
                        earliest_year = year
        if earliest_year != float("inf"):
            if re.search(r"BC", metadata[metadata_with_year], re.I) and earliest_year > 0:
                metadata["year"] = -earliest_year
            else:
                metadata["year"] = earliest_year
        return metadata

    def parse_metadata(self, sort_by_field, header="tei", verbose=True):
        """Parsing metadata fields in TEI or Dublin Core headers"""
        if verbose is True:  # Turn off output when called from other libs such as TextPAIR
            print("### Parsing metadata ###", flush=True)
        if header == "tei":
            load_metadata = self.parse_tei_header(verbose)
        else:
            load_metadata = self.parse_dc_header()

        print(
            f'{time.ctime()}: Sorting files by the following metadata fields: {", ".join([i for i in sort_by_field])}...',
            end=" ",
            flush=True,
        )

        self.sort_order = sort_by_field  # to be used for the sort by concordance biblio key in web config
        if sort_by_field:
            sorted_load_metadata = sort_list(load_metadata, sort_by_field)
        else:
            sorted_load_metadata = []
            for filename in self.filenames:
                for m in load_metadata:
                    if m["filename"] == os.path.basename(filename):
                        sorted_load_metadata.append(m)
                        break
        if self.debug is True:
            print("Files sorted in following order:")
            for metadata in sorted_load_metadata:
                metadata = collections.defaultdict(str, metadata)
                print(f"File {metadata['filename']}:")
                print({field: metadata[field] for field in sort_by_field}, "\n")
        return sorted_load_metadata

    @classmethod
    def set_file_data(cls, load_metadata, textdir, workdir):
        """Set file data"""
        if load_metadata is None:
            cls.data_dicts = [{"filename": fn.name} for fn in os.scandir(textdir)]
        else:
            cls.data_dicts = load_metadata
        cls.filequeue = [
            {
                "name": d["filename"],
                "size": os.path.getsize(os.path.join(textdir, d["filename"])),
                "id": n + 1,
                "options": d["options"] if "options" in d else {},
                "newpath": textdir + d["filename"],
                "raw": workdir + d["filename"] + ".raw",
                "words": workdir + d["filename"] + ".words.sorted",
                "toms": workdir + d["filename"] + ".toms",
                "sortedtoms": workdir + d["filename"] + ".toms.sorted",
                "pages": workdir + d["filename"] + ".pages",
                "refs": workdir + d["filename"] + ".refs",
                "graphics": workdir + d["filename"] + ".graphics",
                "lines": workdir + d["filename"] + ".lines",
                "results": workdir + d["filename"] + ".results",
            }
            for n, d in enumerate(cls.data_dicts)
        ]
        cls.metadata_hierarchy.append([])
        # Adding in doc level metadata
        for d in cls.data_dicts:
            for k in list(d.keys()):
                if k not in cls.metadata_fields:
                    cls.metadata_fields.append(k)
                    cls.metadata_hierarchy[0].append(k)
                if k not in cls.metadata_types:
                    cls.metadata_types[k] = "doc"
                    # don't need to check for conflicts, since doc is first.

        # Adding non-doc level metadata
        for element_type in cls.parser_config["metadata_to_parse"]:
            if element_type != "page" and element_type != "ref" and element_type != "line":
                cls.metadata_hierarchy.append([])
                for param in cls.parser_config["metadata_to_parse"][element_type]:
                    if param not in cls.metadata_fields:
                        cls.metadata_fields.append(param)
                        cls.metadata_hierarchy[-1].append(param)
                    if param not in cls.metadata_types:
                        cls.metadata_types[param] = element_type
                    else:  # we have a serious error here!  Should raise going forward.
                        pass

        # Add unique philo ids for top level text objects
        cls.metadata_fields.extend(["philo_doc_id", "philo_div1_id", "philo_div2_id", "philo_div3_id"])
        for pos, object_level in enumerate(["doc", "div1", "div2", "div3"]):
            cls.metadata_hierarchy[pos].append(f"philo_{object_level}_id")
            cls.metadata_types[f"philo_{object_level}_id"] = object_level

    @classmethod
    def parse_files(cls, workers, verbose=True):
        """Parse all files
        chunksize is setable from the philoload script and can be helpful when loading
        many small files"""
        if len(cls.filequeue) == 0:
            print(
                "\n\n"
                + r"¯\_(ツ)_/¯"
                + "\nThe path you provided for your source texts contains no parsable files. Exiting...\n"
            )
            sys.exit(1)
        os.chdir(cls.workdir)
        cls.parsed_line_counts = None
        if verbose is True:
            print("\n\n### Parsing files ###")
            print("%s: parsing %d files." % (time.ctime(), len(cls.filequeue)))
        line_counts = [0, 0]  # lines of the words and lemmas files of the files parsed so far
        with tqdm(total=len(cls.filequeue), smoothing=0, leave=False, desc="Parsing files") as pbar:
            if cls.nlp is None:
                # Parse the largest files first so a big file started last doesn't leave all but one worker idle.
                # Files are parsed independently (each has its own id and outputs): the order doesn't change the output.
                file_positions = sorted(
                    range(len(cls.data_dicts)), key=lambda file_pos: cls.filequeue[file_pos]["size"], reverse=True
                )
                with process_pool(workers, init_parse_worker, (*pickle_loader_state(cls), workers)) as pool:
                    parsed_files = [
                        pool.submit(parse_in_worker, cls.filequeue[pos], cls.data_dicts[pos]) for pos in file_positions
                    ]
                    for parsed_file in as_completed(parsed_files):
                        word_lines, lemma_lines = parsed_file.result()
                        line_counts[0] += word_lines
                        line_counts[1] += lemma_lines
                        pbar.update()
            else:  # the spaCy model runs on the GPU: files are tagged in this process only
                for file_pos in range(len(cls.data_dicts)):
                    cls.parse_file(file_pos)
                    line_counts[0] += cls.filequeue[file_pos]["word_lines"]
                    line_counts[1] += cls.filequeue[file_pos]["lemma_lines"]
                    pbar.update()
        cls.parsed_line_counts = line_counts
        if verbose is True:
            print("%s: done parsing" % time.ctime())

    @classmethod
    def parse_file(cls, file_pos):
        """Parse a single file"""
        return cls.parse_text(cls.filequeue[file_pos], cls.data_dicts[file_pos])

    @classmethod
    def parse_text(cls, text, metadata):
        """Parse a single file, given its filequeue entry and metadata"""
        options = text["options"]
        if "options" in metadata:  # cleanup, should do above.
            del metadata["options"]

        if "parser_factory" not in options:
            options["parser_factory"] = cls.parser_config["parser_factory"]
        parser_factory = options["parser_factory"]
        del options["parser_factory"]

        if "load_filters" not in options:
            options["load_filters"] = cls.parser_config["load_filters"]
        filters = options["load_filters"]
        del options["load_filters"]

        for option in [
            "token_regex",
            "suppress_tags",
            "break_apost",
            "chars_not_to_index",
            "break_sent_in_line_group",
            "tag_exceptions",
            "join_hyphen_in_words",
            "abbrev_expand",
            "long_word_limit",
            "flatten_ligatures",
            "sentence_breakers",
            "metadata_sql_types",
            "lowercase_index",
        ]:
            try:
                options[option] = cls.parser_config[option]
            except KeyError:  # option hasn't been set
                pass

        with open(text["raw"], "w", encoding="utf8") as raw_file:
            parser = parser_factory(
                raw_file,
                text["id"],
                text["size"],
                known_metadata=metadata,
                tag_to_obj_map=cls.parser_config["tag_to_obj_map"],
                metadata_to_parse=cls.parser_config["metadata_to_parse"],
                words_to_index=cls.words_to_index,
                file_type=cls.parser_config["file_type"],
                lemmas=cls.lemmas,
                **options,
            )
            with open(text["newpath"], "r", newline="", encoding="utf8") as input_file:
                try:
                    parser.parse(input_file)
                except RuntimeError as error:
                    raise ParserError(f"{text['name']} failed to parse") from error
        for f in filters:
            try:
                f(cls, text)
            except Exception:
                raise ParserError(f"{text['name']} has caused parser to die.")

        # Lines of its words and lemmas files, merged into those count_words would otherwise count
        text["word_lines"] = count_newlines(text["words"])
        lemma_file = text["raw"] + ".lemma.lz4"
        text["lemma_lines"] = count_newlines(lemma_file, lz4_compressed=True) if os.path.exists(lemma_file) else 0
        run_shell("lz4 --rm -c -q -3 %s > %s" % (text["words"], text["words"] + ".lz4"))
        if cls.debug is False:
            os.remove(text["raw"])
        return text["results"]

    def merge_objects(self):
        """Merge all parsed objects"""
        print("\n### Merge parser output ###")
        # With this few files, each merge is a single sort writing its own file (see merge_files), so they can
        # run at the same time. With more files, each merge already runs several sorts in parallel.
        if len(self.filequeue) <= (250 if sys.platform == "darwin" else 1000):
            print(f"{time.ctime()}: sorting words, lemmas and objects", flush=True)
            with thread_pool(3) as executor:
                merges = [executor.submit(self.merge_files, file_type) for file_type in ("words", "lemmas", "toms")]
                for merge in merges:
                    merge.result()
        else:
            print(f"{time.ctime()}: sorting words")
            self.merge_files("words")

            print(f"{time.ctime()}: sorting lemmas")
            self.merge_files("lemmas")

            print(f"{time.ctime()}: sorting objects", flush=True)
            self.merge_files("toms")
        if self.debug is False:
            for toms_file in iglob(self.workdir + "/*toms.sorted"):
                os.remove(toms_file)

        for object_type, extension in [
            ("pages", "pages"),
            ("references", "refs"),
            ("graphics", "graphics"),
            ("lines", "lines"),
        ]:
            print(f"{time.ctime()}: joining {object_type}", flush=True)
            # Concatenate in find's order (which determines row order in the SQL tables), without a cat and rm per file
            found_files = subprocess.run(
                ["find", self.workdir, "-type", "f", "-name", f"*{extension}"], capture_output=True, check=False
            ).stdout.split()
            if found_files:
                with open(f"{self.workdir}/all_{extension}", "ab") as joined_file:
                    for found_file in found_files:
                        with open(found_file, "rb") as object_file:
                            shutil.copyfileobj(object_file, joined_file)
                        if self.debug is False:
                            os.remove(found_file)

    def merge_files(self, file_type, file_num=1000, verbose=True):
        """This function runs a multi-stage merge sort on words
        Since PhiloLogic can potentially merge thousands of files, we need to split
        the sorting stage into multiple steps to avoid running out of file descriptors
        """
        if sys.platform == "darwin":
            file_num = 250
        lists_of_files = []
        files = []
        if file_type == "words":
            suffix = "/*words.sorted.lz4"
            if self.debug is False:
                open_file_command = "lz4cat --rm"
            else:
                open_file_command = "lz4cat"
            sort_command = f"LANG=C sort -S 7% -m -T {self.workdir} {self.sort_by_word} {self.sort_by_id} "
            final_sort_command = f"LANG=C sort -S 25% --parallel=4 -m -T {self.workdir} {self.sort_by_word} {self.sort_by_id} "
        elif file_type == "lemmas":
            suffix = "/*raw.lemma.lz4"
            if self.debug is False:
                open_file_command = "lz4cat --rm"
            else:
                open_file_command = "lz4cat"
            sort_command = f"LANG=C sort -S 7% -T {self.workdir} {self.sort_by_word} {self.sort_by_id} "
            final_sort_command = f"LANG=C sort -S 25% --parallel=4 -m -T {self.workdir} {self.sort_by_word} {self.sort_by_id} "
        else:  # sorting for toms
            suffix = "/*.toms.sorted"
            open_file_command = "cat"
            sort_command = f"LANG=C sort -S 7% -m -T {self.workdir} {self.sort_by_id} "
            final_sort_command = f"LANG=C sort -S 25% --parallel=4 -m -T {self.workdir} {self.sort_by_id} "

        # First we split the sort workload into chunks of 1000 (default defined in the file_num keyword)
        for f in iglob(self.workdir + suffix):
            f = os.path.basename(f)
            files.append((f"<({open_file_command} {f})", self.workdir + "/" + f))
            if len(files) == file_num:
                lists_of_files.append(files)
                files = []
        if files:
            lists_of_files.append(files)

        total_files = sum(len(files) for files in lists_of_files)
        # Then we run the merge sort on each chunk of 500 files and compress the result
        if verbose is True:
            print(
                f"{time.ctime()}: Merging {file_type} in batches of {file_num}...",
                flush=True,
            )
        else:
            print(f"Merging {file_type} in batches of {file_num}...", flush=True)
        os.system(f"touch {self.workdir}/sorted.init")

        if len(lists_of_files) == 1:
            # A single batch: write its output directly to the final file. Merging it again with sort -m
            # (the second stage below) would only copy a single sorted input.
            command_list = " ".join([i[0] for i in lists_of_files[0]])
            if file_type == "words":
                output_file = os.path.join(self.workdir, "all_words_sorted.lz4")
                command = f"{sort_command}{command_list} | lz4 -q > {output_file}"
            elif file_type == "lemmas":
                output_file = os.path.join(self.workdir, "all_lemmas_sorted.lz4")
                command = f"{sort_command}{command_list} | lz4 -q > {output_file}"
            else:
                output_file = os.path.join(self.workdir, "all_toms_sorted")
                command = f"{sort_command}{command_list} > {output_file}"
            run_shell(command, description=f"{file_type} sorting")
            return
        with tqdm(total=total_files, leave=False) as pbar:

            def run_batch(pos, object_list):
                command_list = " ".join([i[0] for i in object_list])
                output = os.path.join(self.workdir, f"sorted.{pos}.split")
                args = sort_command + command_list
                run_shell(f"{args} | lz4 -3 -q >{output}", description=f"{file_type} sorting")
                return len(object_list)

            with thread_pool(4) as executor:
                futures = [executor.submit(run_batch, pos, obj_list) for pos, obj_list in enumerate(lists_of_files)]
                for future in as_completed(futures):
                    pbar.update(future.result())

        # WARNING: we are technically limited by the file descriptor limit (1024), which should be equivalent to 1,024,000 files.
        sorted_files = " ".join([f"<(lz4cat -q --rm {i})" for i in iglob(f"{self.workdir}/*.split")])
        if file_type == "words":
            output_file = os.path.join(self.workdir, "all_words_sorted.lz4")
            command = f"{final_sort_command} -b --compress-program=lz4 {sorted_files} | lz4 -q > {output_file}"
        elif file_type == "lemmas":
            output_file = os.path.join(self.workdir, "all_lemmas_sorted.lz4")
            command = f"{final_sort_command} -b --compress-program=lz4 {sorted_files} | lz4 -q > {output_file}"
        else:
            output_file = os.path.join(self.workdir, "all_toms_sorted")
            command = f"{final_sort_command} {sorted_files} > {output_file}"
        if verbose is True:
            print(
                f"{time.ctime()}: Merging all merged sorted files (this may take a while)...",
                flush=True,
                end=" ",
            )

        run_shell(command, description=f"{file_type} sorting")
        print("done.", flush=True)

        for sorted_file in os.scandir(self.workdir):
            if sorted_file.name.endswith(".split"):
                os.unlink(sorted_file.path)

    @classmethod
    def count_words(cls):
        """Count words in all files"""
        print("\n### Counting total words ###", flush=True)
        print(f"{time.ctime()}: counting words in all files...", flush=True)
        if cls.parsed_line_counts is not None:  # counted file by file by parse_files
            print(f"{time.ctime()}: counting lemmas in all files...", flush=True)
            cls.word_count, cls.lemma_count = cls.parsed_line_counts
            return
        with ThreadPoolExecutor(max_workers=2) as executor:  # both counts run in parallel subprocesses
            word_count = executor.submit(count_lines, f"{cls.workdir}/all_words_sorted.lz4", lz4=True)
            print(f"{time.ctime()}: counting lemmas in all files...", flush=True)
            lemma_count = executor.submit(count_lines, f"{cls.workdir}/all_lemmas_sorted.lz4", lz4=True)
            cls.word_count = word_count.result()
            cls.lemma_count = lemma_count.result()

    @classmethod
    def build_inverted_index(cls, commit_interval=5000):
        """Create inverted index. Each part of it (words, lemmas and their attributes) is built from a sorted file by its
        own process, in its own database, and the parts are then merged into words.lmdb. The parts for attributes only
        have entries if words have attributes, which the words part finds out, so they are built at the same time."""
        print("\n### Create inverted index ###", flush=True)
        overflow_dir = f"{cls.destination}/overflow_words"
        os.mkdir(overflow_dir)
        cls.all_word_attribute_names = None
        words_file = f"{cls.workdir}/all_words_sorted.lz4"
        lemmas_file = f"{cls.workdir}/all_lemmas_sorted.lz4"
        part_paths = {
            part: f"{cls.destination}/temp_index_{part}.lmdb"
            for part in ("words", "lemmas", "word_attributes", "lemma_attributes")
        }
        progress = shared_value("q", 0)
        with process_pool(8 if cls.lemma_count > 0 else 3, init_index_worker, (progress,)) as pool:
            # The lemma and word attribute frequency files are also written from the sorted files alone: write them
            # alongside, for PostFilters.lemma_and_attribute_frequencies to use
            frequency_jobs = [
                (
                    write_unique_word_attributes,
                    (words_file, f"{cls.workdir}/word_attributes", "", cls.attributes_to_skip),
                )
            ]
            if cls.lemma_count > 0:
                frequency_jobs.append((write_lemma_frequencies, (lemmas_file, f"{cls.workdir}/lemmas")))
                frequency_jobs.append(
                    (
                        write_unique_word_attributes,
                        (lemmas_file, f"{cls.workdir}/lemma_word_attributes", "lemma:", cls.attributes_to_skip),
                    )
                )
            frequency_files = [
                (frequency_file_key(function, args), args[1], pool.submit(function, *args))
                for function, args in frequency_jobs
            ]
            lemma_lookup = None
            if cls.lemma_count > 0:  # separate database built from the lemmas file only
                lemma_lookup = pool.submit(build_lemma_lookup_index, cls.workdir, cls.destination, cls.lemma_count)
            print(f"{time.ctime()}: Creating word index...", flush=True)
            parts = {
                "words": pool.submit(
                    index_words,
                    words_file,
                    part_paths["words"],
                    overflow_dir,
                    cls.has_attributes,
                    cls.attributes_to_skip,
                    commit_interval,
                ),
                "word_attributes": pool.submit(
                    index_word_attributes,
                    words_file,
                    part_paths["word_attributes"],
                    overflow_dir,
                    "",
                    cls.attributes_to_skip,
                    commit_interval,
                    True,
                ),
            }
            if cls.lemma_count > 0:
                print(f"{time.ctime()}: Creating lemma index...", flush=True)
                parts["lemmas"] = pool.submit(
                    index_lemmas, lemmas_file, part_paths["lemmas"], overflow_dir, commit_interval
                )
                parts["lemma_attributes"] = pool.submit(
                    index_word_attributes,
                    lemmas_file,
                    part_paths["lemma_attributes"],
                    overflow_dir,
                    "lemma:",
                    cls.attributes_to_skip,
                    commit_interval,
                    False,
                )
            total = 2 * cls.word_count + (2 * cls.lemma_count if cls.lemma_count > 0 else 0)
            with tqdm(total=total, desc="Storing words, lemmas and their attributes", leave=False) as pbar:
                jobs = [*parts.values(), *(job for _, _, job in frequency_files)]
                if lemma_lookup is not None:
                    jobs.append(lemma_lookup)
                while not all(part.done() for part in parts.values()):
                    done, _ = wait(jobs, timeout=0.5, return_when=FIRST_EXCEPTION)
                    pbar.update(progress.value - pbar.n)
                    for job in done:
                        job.result()  # raises the error of a job which failed

            count, overflow_keys, has_attributes = parts["words"].result()
            cls.overflow_words.update(overflow_keys)
            if has_attributes:
                cls.has_attributes = True
            print(f"{time.ctime()}: Stored {cls.word_count} words in {count} entries.", flush=True)
            if cls.lemma_count > 0:
                count, overflow_keys = parts["lemmas"].result()
                cls.overflow_words.update(overflow_keys)
                print(f"{time.ctime()}: Stored {cls.lemma_count} lemmas in {count} entries.", flush=True)
            merged_parts = ["words", "lemmas"] if cls.lemma_count > 0 else ["words"]
            if cls.has_attributes is True:
                count, overflow_keys, all_word_attribute_names = parts["word_attributes"].result()
                cls.overflow_words.update(overflow_keys)
                file_stat = os.stat(words_file)
                cls.all_word_attribute_names = (
                    words_file,
                    file_stat.st_size,
                    file_stat.st_mtime_ns,
                    all_word_attribute_names,
                )
                print(f"{time.ctime()}: Found word attributes: stored {count} word attributes.", flush=True)
                merged_parts.append("word_attributes")
                if cls.lemma_count > 0:
                    count, overflow_keys, _ = parts["lemma_attributes"].result()
                    cls.overflow_words.update(overflow_keys)
                    print(f"{time.ctime()}: Stored {count} lemma word attributes.", flush=True)
                    merged_parts.append("lemma_attributes")

            if len(merged_parts) > 1:  # merge the parts in the order they used to be built in, one after the other
                print(f"{time.ctime()}: Merging word index parts...", flush=True)
                merge_indexes([part_paths[part] for part in merged_parts], f"{cls.destination}/words.lmdb")
            else:
                print(f"{time.ctime()}: Optimizing word index for space...", flush=True)
                os.mkdir(f"{cls.destination}/words.lmdb")
                # Reopen env without writemap to compact the database
                src_env = lmdb.open(part_paths["words"], readonly=True)
                src_env.copy(f"{cls.destination}/words.lmdb", compact=True)
                src_env.close()
            for part in parts:
                shutil.rmtree(part_paths[part])

            if lemma_lookup is not None:
                lemma_lookup.result()
            cls.precomputed_files = {}
            for key, path, job in frequency_files:
                job.result()
                cls.precomputed_files[key] = path

    def setup_sql_load(self, verbose=True):
        """Setup SQLite DB creation"""
        for table in self.tables:
            if table == "pages":
                file_in = self.destination + "/WORK/all_pages"
                indices = [("philo_id",)]
                depth = 9
            elif table == "toms":
                file_in = self.destination + "/WORK/all_toms_sorted"
                indices = (
                    [("philo_type",), ("philo_id",), ("img",)]
                    + Loader.metadata_fields
                    + [(f"philo_{philo_type}_id",) for philo_type in ["doc", "div1", "div2", "div3", "para"]]
                )
                depth = 7
            elif table == "refs":
                file_in = self.destination + "/WORK/all_refs"
                indices = [("parent",), ("target",), ("type",)]
                depth = 9
            elif table == "graphics":
                file_in = self.destination + "/WORK/all_graphics"
                indices = [("parent",), ("philo_id",)]
                depth = 9
            elif table == "lines":
                file_in = self.destination + "/WORK/all_lines"
                indices = [("doc_id", "start_byte", "end_byte")]
                depth = 9
            # Only load if file is not empty:
            if os.path.getsize(file_in) > 0:
                post_filter = make_sql_table(table, file_in, indices=indices, depth=depth, verbose=verbose)
                self.post_filters.insert(0, post_filter)

    @classmethod
    def post_processing(cls, *extra_filters, verbose=True):
        """Run important post-parsing functions for frequencies and word normalization"""
        if verbose is True:
            print("\n### Storing in database ###")
        for f in cls.post_filters:
            if f.__name__ == "metadata_frequencies":
                cls.metadata_fields_not_found = f(cls)
            else:
                f(cls)

        # Set up sentences database
        # We need to find which word attributes were found in the collection
        attributes_to_skip = list(cls.attributes_to_skip)
        attributes_to_skip.remove("lemma")
        attributes_to_skip = set(attributes_to_skip)
        attributes_to_skip.update({"token", "position", "philo_type"})
        word_attributes = set()
        if cls.has_attributes is True:
            words_file = f"{cls.workdir}/all_words_sorted.lz4"
            file_stat = os.stat(words_file)
            if cls.all_word_attribute_names is not None and cls.all_word_attribute_names[:3] == (
                words_file,
                file_stat.st_size,
                file_stat.st_mtime_ns,
            ):  # already collected while building the word attributes index from this same file
                word_attributes.update(cls.all_word_attribute_names[3])
            else:
                with lz4.frame.open(words_file) as input_file:
                    for line in input_file:
                        line = line.decode("utf-8")
                        _, _, _, attributes = line.split("\t", 3)
                        word_attributes.update(loads(attributes).keys())
        if cls.lemma_count > 0:
            word_attributes.add("lemma")
        cls.word_attributes = list(word_attributes.difference(attributes_to_skip))

        colloc_destination = os.path.join(cls.destination, "collocations")
        make_collocation_database(cls, colloc_destination)

        if extra_filters:
            print("Running the following additional filters:")
            for f in extra_filters:
                print(f.__name__ + "...", end=" ")
                f(cls)

    def finish(self):
        """Write important runtime information to the database directory"""
        print("\n### Finishing up ###")
        os.mkdir(self.destination + "/hitlists/")
        os.chmod(self.destination + "/hitlists/", 0o777)
        os.chmod(os.path.join(self.destination, "TEXT"), 0o775)

        # Note: the lemmas / word_attributes / lemma_word_attributes frequency
        # files are now written by the lemma_and_attribute_frequencies post-filter
        # (in post_processing), so build_word_forms_lmdb can consume them.

        # Note: data/.htaccess ("deny from all") is no longer needed.
        # Under gunicorn, Apache/Nginx only proxies requests — it never
        # serves files from the database directory directly.

        # The web app build only depends on appConfig.json (not on the database), so it runs while we finish up
        with open(os.path.join(self.web_app_dir, "appConfig.json"), "w", encoding="utf8") as app_config:
            dump({"dbUrl": ""}, app_config)
        npm = "/var/lib/philologic5/bin/npm"
        web_app_build = subprocess.Popen(
            f"cd {self.web_app_dir}; {npm} install > {self.web_app_dir}/web_app_build.log 2>&1 && {npm} run build >> {self.web_app_dir}/web_app_build.log 2>&1",
            shell=True,
        )

        self.write_db_config()
        if self.predefined_web_config is False:
            self.write_web_config()
        if self.debug is False:
            os.system(f"rm -rf {self.workdir}")

        print("Building Web Client Application...", end=" ", flush=True)
        os.chdir(self.web_app_dir)
        if web_app_build.wait() != 0:
            raise RuntimeError(f"Building the web client application failed, see {self.web_app_dir}web_app_build.log")
        print("done.")

    def write_db_config(self):
        """Write local variables used by libphilo"""
        filename = self.destination + "/db.locals.py"
        metadata = [i for i in Loader.metadata_fields if i not in Loader.metadata_fields_not_found]
        metadata_sql_types = {
            "philo_type": "text",
            "philo_id": "text",
            "philo_name": "text",
            "philo_seq": "text",
            "year": "int",
            **{
                field: self.parser_config["metadata_sql_types"].get(field, "text")
                for field in metadata
                if field != "year"
            },
        }
        db_values = {
            "metadata_fields": metadata,
            "metadata_hierarchy": Loader.metadata_hierarchy,
            "metadata_types": Loader.metadata_types,
            "normalized_fields": self.normalized_fields,
            "debug": self.debug,
            "ascii_conversion": Loader.ascii_conversion,
            "metadata_sql_types": metadata_sql_types,
        }
        db_values["token_regex"] = self.token_regex
        db_values["default_object_level"] = self.default_object_level
        db_values["word_attributes"] = self.word_attributes
        db_values["overflow_words"] = self.overflow_words

        db_config = MakeDBConfig(filename, **db_values)
        with open(filename, "w", encoding="utf8") as db_file:
            try:
                print(format_str(str(db_config), mode=FileMode()), file=db_file)
            except:
                print(str(db_config))
                raise
        print("wrote database info to %s." % (filename))

    def write_web_config(self):
        """Write configuration variables for the Web application"""
        dbname = os.path.basename(os.path.dirname(self.destination.rstrip("/")))
        metadata = [
            i
            for i in Loader.metadata_fields
            if i not in self.metadata_fields_not_found and not i.startswith("philo_") and i != "filename"
        ]
        config_values = {
            "dbname": dbname,
            "metadata": metadata,
            "facets": metadata,
        }

        # Fetch search examples:
        search_examples = {}
        conn = sqlite3.connect(self.destination + "/toms.db")
        conn.text_factory = str
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        for field in metadata:
            object_type = Loader.metadata_types[field]
            try:
                if object_type != "div":
                    cursor.execute(
                        f'select {field} from toms where philo_type="{object_type}" and {field} !="" limit 1'
                    )
                else:
                    cursor.execute(
                        f'select {field} from toms where philo_type="div1" or philo_type="div2" or philo_type="div3" and {field} !="" limit 1'
                    )
            except sqlite3.OperationalError:
                continue
            try:
                search_examples[field] = cursor.fetchone()[0]
            except (TypeError, AttributeError):
                continue
        config_values["search_examples"] = search_examples

        config_values["metadata_input_style"] = {}
        for field in metadata:
            if field == "year":
                config_values["metadata_input_style"][field] = "int"
            elif field not in self.parser_config["metadata_sql_types"]:
                config_values["metadata_input_style"][field] = "text"
            elif self.parser_config["metadata_sql_types"][field] == "int":
                config_values["metadata_input_style"][field] = "int"
            elif self.parser_config["metadata_sql_types"][field] == "date":
                config_values["metadata_input_style"][field] = "date"

        # Populate kwic metadata sorting and kwic biblio fields variables with metadata
        # Check if title and author are empty, if so, default to filename
        config_values["kwic_metadata_sorting_fields"] = []
        config_values["kwic_bibliography_fields"] = []
        if "author" in config_values["search_examples"]:
            config_values["kwic_metadata_sorting_fields"].append("author")
            config_values["kwic_bibliography_fields"].append("author")
        if "title" in config_values["search_examples"]:
            config_values["kwic_metadata_sorting_fields"].append("title")
            config_values["kwic_bibliography_fields"].append("title")
        if not config_values["kwic_metadata_sorting_fields"]:
            config_values["kwic_metadata_sorting_fields"] = ["filename"]
            config_values["kwic_bibliography_fields"] = ["filename"]

        if "author" in config_values["search_examples"] and "title" in config_values["search_examples"]:
            config_values["concordance_biblio_sorting"] = [
                ("author", "title"),
                ("title", "author"),
            ]

        # Find default start and end dates for times series
        try:
            cursor.execute("SELECT min(year), max(year) FROM toms")
            min_year, max_year = cursor.fetchone()
            try:
                start_date = int(min_year)
            except TypeError:
                start_date = 0
            try:
                end_date = int(max_year)
            except TypeError:
                end_date = 2100
            config_values["time_series_start_end_date"] = {
                "start_date": start_date,
                "end_date": end_date,
            }
        except sqlite3.OperationalError:  # no year field present
            config_values["time_series_start_end_date"] = {
                "start_date": "",
                "end_date": "",
            }

        words_facets = []
        if self.lemma_count > 0:  # Check if the lemmas file is empty
            words_facets.append("lemma")

        # Compile all possible word attributes with their types from the frequency file
        if self.has_attributes is True:
            if "words_facets" not in config_values:
                config_values["words_facets"] = []
            word_attributes = {}
            with open(f"{self.destination}/frequencies/word_attributes", "r", encoding="utf8") as freq_file:
                for line in freq_file:
                    line = line.strip()
                    _, attribute, attribute_value = line.split(":")
                    if attribute not in word_attributes:
                        word_attributes[attribute] = set()
                    word_attributes[attribute].add(attribute_value)
            config_values["word_attributes"] = {k: list(v) for k, v in word_attributes.items()}
            words_facets.extend((word_attributes.keys()))
            config_values["words_facets"] = words_facets
        config_values["ascii_conversion"] = Loader.ascii_conversion

        filename = self.destination + "/web_config.cfg"
        web_config = MakeWebConfig(filename, **config_values)
        with open(os.path.join(filename), "w", encoding="utf8") as output_file:
            print(format_str(str(web_config), mode=FileMode()), file=output_file)
        print(f"wrote Web application info to {filename}")


def shellquote(s):
    """Quote shell commands"""
    return "'" + s.replace("'", "'\\''") + "'"


def setup_db_dir(db_destination, force_delete=False):
    """Setup database directory"""
    try:
        os.mkdir(db_destination)
    except OSError:
        if force_delete is True:  # useful to run db loads with nohup
            os.system("rm -rf %s" % db_destination)
            os.mkdir(db_destination)
        else:
            print("The database folder could not be created at %s" % db_destination)
            print("Do you want to delete this database? Yes/No")
            choice = input().lower()
            if choice.startswith("y"):
                os.system("rm -rf %s" % db_destination)
                os.mkdir(db_destination)
            else:
                sys.exit()

    # Only copy files needed per-database. The central Gunicorn dispatcher
    # handles reports, scripts, and routing for all databases.
    web_app_src = "/var/lib/philologic5/web_app"
    os.system(f"cp -R {web_app_src}/app {db_destination}/")
    os.system(f"cp {web_app_src}/favicon.ico {db_destination}/")
    os.system("mkdir -p %s/custom_functions" % db_destination)
    os.system("touch %s/custom_functions/__init__.py" % db_destination)
