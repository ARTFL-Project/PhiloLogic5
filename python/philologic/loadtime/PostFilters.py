#!/var/lib/philologic5/philologic_env/bin/python3


import array
import hashlib
import os
import sqlite3
import struct as _struct
import time
from collections import defaultdict

import lmdb
import lz4.frame
import multiprocess as mp
import numpy as np
import pandas as pd
import regex as re
from orjson import loads
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from unidecode import unidecode

from philologic.utils import count_lines


def make_sql_table(table, file_in, db_file="toms.db", indices=None, depth=7, verbose=True):
    """SQL Loader function"""

    def inner_make_sql_table(loader_obj):
        if verbose is True:
            print(f"{time.ctime()}: Loading the {table} SQLite table...")
        else:
            print(f"Loading the {table} SQLite table...")
        db_destination = os.path.join(loader_obj.destination, db_file)
        line_count = count_lines(file_in)
        conn = sqlite3.connect(db_destination, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.text_factory = str
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if table == "toms":
            query = f"create table if not exists {table} (philo_type text, philo_name text, philo_id text, philo_seq text, year int)"
        else:
            query = f"create table if not exists {table} (philo_type, philo_name, philo_id, philo_seq)"
        cursor.execute(query)
        with tqdm(total=line_count, leave=False) as pbar:
            with open(file_in, encoding="utf8") as input_file:
                for sequence, line in enumerate(input_file):
                    philo_type, philo_name, philo_id, attrib = line.split("\t", 3)
                    fields = philo_id.split(None, 8)
                    if len(fields) == 9:
                        row = loads(attrib)
                        row["philo_type"] = philo_type
                        row["philo_name"] = philo_name
                        row["philo_id"] = " ".join(fields[:depth])
                        row["philo_seq"] = sequence
                        insert = f"INSERT INTO {table} ({','.join(list(row.keys()))}) values ({','.join('?' for i in range(len(row)))});"
                        try:
                            cursor.execute(insert, list(row.values()))
                        except sqlite3.OperationalError:
                            cursor.execute(f"PRAGMA table_info({table})")
                            column_list = [i[1] for i in cursor]
                            for column in row:
                                if column not in column_list:
                                    if column not in loader_obj.parser_config["metadata_sql_types"]:
                                        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} text;")
                                    else:
                                        cursor.execute(
                                            f"ALTER TABLE {table} ADD COLUMN {column} {loader_obj.parser_config['metadata_sql_types'][column]};"
                                        )
                            cursor.execute(insert, list(row.values()))
                    pbar.update()
        conn.commit()

        if indices is not None:
            for index in indices:
                try:
                    if isinstance(index, str):
                        index = (index,)
                    index_name = f'{table}_{"_".join(index)}_index'
                    index = ",".join(index)
                    cursor.execute(f"create index if not exists {index_name} on {table} ({index})")
                    if table == "toms":
                        index_null_name = f"{index}_null_index"  # this is for hitlist stats queries which require indexing philo_id with null metadata values
                        cursor.execute(
                            f"CREATE UNIQUE INDEX IF NOT EXISTS {index_null_name} ON toms(philo_id, {index}) WHERE {index} IS NULL"
                        )
                except sqlite3.OperationalError:
                    pass
        conn.commit()
        conn.close()

        if not loader_obj.debug:
            os.system(f"rm {file_in}")

    return inner_make_sql_table



def make_collocation_database(loader_obj, db_destination):
    """Build flat columnar arrays for vectorized collocation computation.

    Creates numpy arrays storing token IDs, sentence offsets, and sentence keys
    that enable fully vectorized collocation counting at runtime.
    """
    print(f"{time.ctime()}: Building collocation columnar database...")

    vocab = {}  # token string -> uint32 ID
    vocab_counter = 0
    token_ids = array.array("I")  # uint32 per word, memory-efficient
    sent_keys_flat = array.array("I")  # 6 uint32s per sentence, flattened
    sent_offsets = array.array("Q")  # uint64, cumulative word count
    sent_offsets.append(0)

    # Per-attribute arrays (only if word_attributes exist)
    attr_vocabs = {}
    attr_ids = {}
    for attr_name in loader_obj.word_attributes:
        attr_vocabs[attr_name] = {}
        attr_ids[attr_name] = array.array("I")

    word_count = 0
    current_sentence = None

    with tqdm(total=getattr(loader_obj, "word_count", None), leave=False, desc="Building collocation arrays") as pbar:
        for raw_words in sorted(
            (e for e in os.scandir(f"{loader_obj.destination}/words_and_philo_ids") if e.name.endswith(".lz4")),
            key=lambda x: int(x.name.split(".")[0]),
        ):
            with lz4.frame.open(raw_words.path) as input_file:
                for line in input_file:
                    word_obj = loads(line.decode("utf8"))
                    if word_obj["philo_type"] != "word":
                        continue

                    # Track sentence boundaries
                    position_parts = word_obj["position"].split()
                    sent_key = tuple(int(x) for x in position_parts[:6])
                    if sent_key != current_sentence:
                        if current_sentence is not None:
                            sent_offsets.append(word_count)
                        current_sentence = sent_key
                        sent_keys_flat.extend(sent_key)

                    # Token -> vocab ID
                    token = word_obj["token"]
                    if token not in vocab:
                        vocab[token] = vocab_counter
                        vocab_counter += 1
                    token_ids.append(vocab[token])

                    # Per-attribute IDs
                    for attr_name in loader_obj.word_attributes:
                        val = word_obj.get(attr_name, "")
                        if attr_name == "lemma":
                            val = f"lemma:{val}"
                        attr_vocab = attr_vocabs[attr_name]
                        if val not in attr_vocab:
                            attr_vocab[val] = len(attr_vocab)
                        attr_ids[attr_name].append(attr_vocab[val])

                    word_count += 1
                    pbar.update()

        # Final sentence offset
        sent_offsets.append(word_count)

    pbar.close()

    # Convert to numpy arrays
    token_ids_arr = np.frombuffer(token_ids, dtype=np.uint32).copy()
    sent_offsets_arr = np.frombuffer(sent_offsets, dtype=np.uint64).copy()
    n_sents = len(sent_offsets_arr) - 1
    sent_keys_arr = np.frombuffer(sent_keys_flat, dtype=np.uint32).reshape(n_sents, 6)

    # Sort sentences by key (required for searchsorted at query time)
    sent_keys_be = sent_keys_arr.astype(">u4")
    print(f"{time.ctime()}: Sorting {n_sents:,} sentences by key...")
    sort_order = np.lexsort(sent_keys_arr.T[::-1])
    is_sorted = np.array_equal(sort_order, np.arange(n_sents, dtype=sort_order.dtype))

    if not is_sorted:
        sent_keys_be = sent_keys_be[sort_order]

        # Reorder token_ids and attribute arrays by sentence order
        old_offsets = sent_offsets_arr
        new_lengths = np.diff(old_offsets)[sort_order]
        new_sent_offsets = np.empty(n_sents + 1, dtype=np.uint64)
        new_sent_offsets[0] = 0
        np.cumsum(new_lengths, out=new_sent_offsets[1:])

        new_token_ids = np.empty(word_count, dtype=np.uint32)
        new_attr_arrays = {name: np.empty(word_count, dtype=np.uint32) for name in attr_ids}
        attr_np = {name: np.frombuffer(attr_ids[name], dtype=np.uint32) for name in attr_ids}

        for i, old_idx in enumerate(sort_order):
            old_s = old_offsets[old_idx]
            old_e = old_offsets[old_idx + 1]
            new_s = new_sent_offsets[i]
            n = old_e - old_s
            new_token_ids[new_s : new_s + n] = token_ids_arr[old_s:old_e]
            for name in new_attr_arrays:
                new_attr_arrays[name][new_s : new_s + n] = attr_np[name][old_s:old_e]

        token_ids_arr = new_token_ids
        sent_offsets_arr = new_sent_offsets
        for name in new_attr_arrays:
            attr_ids[name] = new_attr_arrays[name]
    else:
        # Convert attribute arrays to numpy
        for name in attr_ids:
            attr_ids[name] = np.frombuffer(attr_ids[name], dtype=np.uint32).copy()

    # Build reverse vocab: token_id -> string
    vocab_reverse = np.empty(len(vocab), dtype=object)
    for token, idx in vocab.items():
        vocab_reverse[idx] = token

    # Save all arrays
    os.makedirs(db_destination, exist_ok=True)
    print(f"{time.ctime()}: Saving collocation arrays ({n_sents:,} sentences, {word_count:,} words, {len(vocab):,} unique tokens)...")
    np.save(os.path.join(db_destination, "token_ids.npy"), token_ids_arr)
    np.save(os.path.join(db_destination, "sent_offsets.npy"), sent_offsets_arr)
    np.save(os.path.join(db_destination, "sent_keys_be.npy"), sent_keys_be)
    # Also save native byte-order keys for numba parallel searchsorted
    sent_keys_native = sent_keys_be.byteswap().view(np.uint32).reshape(-1, 6)
    np.save(os.path.join(db_destination, "sent_keys_native.npy"), sent_keys_native)
    # Save keys as S24 byte strings for fast numpy searchsorted (avoids runtime copy)
    np.save(os.path.join(db_destination, "sent_keys_s24.npy"), sent_keys_be.view("S24").ravel())
    np.save(os.path.join(db_destination, "vocab.npy"), vocab_reverse)

    # Save vocab hashes (deterministic md5-based) for fast numba filter scanning
    vocab_hashes = np.empty(len(vocab), dtype=np.uint64)
    for token, idx in vocab.items():
        vocab_hashes[idx] = _struct.unpack("<Q", hashlib.md5(token.encode("utf-8")).digest()[:8])[0]
    np.save(os.path.join(db_destination, "vocab_hashes.npy"), vocab_hashes)

    # Save vocab as flat bytes + offsets for lazy string decoding at query time
    encoded = [vocab_reverse[i].encode("utf-8") for i in range(len(vocab_reverse))]
    vocab_byte_offsets = np.empty(len(vocab_reverse) + 1, dtype=np.uint64)
    vocab_byte_offsets[0] = 0
    for i, b in enumerate(encoded):
        vocab_byte_offsets[i + 1] = vocab_byte_offsets[i] + len(b)
    total_bytes = int(vocab_byte_offsets[-1])
    vocab_data = bytearray(total_bytes)
    pos = 0
    for b in encoded:
        vocab_data[pos : pos + len(b)] = b
        pos += len(b)
    np.save(os.path.join(db_destination, "vocab_offsets.npy"), vocab_byte_offsets)
    with open(os.path.join(db_destination, "vocab_strings.bin"), "wb") as f:
        f.write(vocab_data)

    # Save precomputed has-number flag per vocab entry (replaces per-word regex at query time)
    _NUMBER = re.compile(r"\d")
    vocab_has_number = np.array(
        [bool(_NUMBER.search(vocab_reverse[i])) for i in range(len(vocab_reverse))],
        dtype=np.bool_,
    )
    np.save(os.path.join(db_destination, "vocab_has_number.npy"), vocab_has_number)

    # Save alphabetical sort rank per vocab entry (for fast numpy KWIC sorting)
    max_word_len = 48
    n_v = len(vocab_reverse)
    padded = np.zeros((n_v, max_word_len), dtype=np.uint8)
    for i in range(n_v):
        b = vocab_reverse[i].encode("utf-8")[:max_word_len]
        padded[i, : len(b)] = list(b)
    keys = padded.view(f"S{max_word_len}").ravel()
    order = np.argsort(keys, kind="stable")
    # Assign equal ranks to tied entries (identical padded strings)
    sorted_keys = keys[order]
    differs = np.ones(n_v, dtype=np.bool_)
    if n_v > 1:
        differs[1:] = sorted_keys[1:] != sorted_keys[:-1]
    ranks_sorted = np.cumsum(differs).astype(np.uint32) - 1
    vocab_sort_rank = np.empty(n_v, dtype=np.uint32)
    vocab_sort_rank[order] = ranks_sorted
    np.save(os.path.join(db_destination, "vocab_sort_rank.npy"), vocab_sort_rank)

    # Save pre-unidecoded vocab strings (replaces per-word unidecode() at query time)
    if getattr(loader_obj, "ascii_conversion", True):
        ascii_encoded = [unidecode(vocab_reverse[i]).encode("utf-8") for i in range(len(vocab_reverse))]
        ascii_offsets = np.empty(len(vocab_reverse) + 1, dtype=np.uint64)
        ascii_offsets[0] = 0
        for i, b in enumerate(ascii_encoded):
            ascii_offsets[i + 1] = ascii_offsets[i] + len(b)
        ascii_flat = bytearray(int(ascii_offsets[-1]))
        p = 0
        for b in ascii_encoded:
            ascii_flat[p : p + len(b)] = b
            p += len(b)
        np.save(os.path.join(db_destination, "vocab_ascii_offsets.npy"), ascii_offsets)
        with open(os.path.join(db_destination, "vocab_ascii_strings.bin"), "wb") as f:
            f.write(ascii_flat)

        # Also compute sort rank for ascii-converted vocab
        padded_a = np.zeros((n_v, max_word_len), dtype=np.uint8)
        for i, b in enumerate(ascii_encoded):
            L = min(len(b), max_word_len)
            padded_a[i, :L] = list(b[:L])
        keys_a = padded_a.view(f"S{max_word_len}").ravel()
        order_a = np.argsort(keys_a, kind="stable")
        sorted_keys_a = keys_a[order_a]
        differs_a = np.ones(n_v, dtype=np.bool_)
        if n_v > 1:
            differs_a[1:] = sorted_keys_a[1:] != sorted_keys_a[:-1]
        ranks_sorted_a = np.cumsum(differs_a).astype(np.uint32) - 1
        vocab_sort_rank_ascii = np.empty(n_v, dtype=np.uint32)
        vocab_sort_rank_ascii[order_a] = ranks_sorted_a
        np.save(os.path.join(db_destination, "vocab_sort_rank_ascii.npy"), vocab_sort_rank_ascii)

    # Save per-attribute arrays
    for attr_name in loader_obj.word_attributes:
        attr_arr = attr_ids[attr_name] if isinstance(attr_ids[attr_name], np.ndarray) else np.frombuffer(attr_ids[attr_name], dtype=np.uint32).copy()
        np.save(os.path.join(db_destination, f"attr_{attr_name}_ids.npy"), attr_arr)
        attr_vocab_reverse = np.empty(len(attr_vocabs[attr_name]), dtype=object)
        for val, idx in attr_vocabs[attr_name].items():
            attr_vocab_reverse[idx] = val
        np.save(os.path.join(db_destination, f"attr_{attr_name}_vocab.npy"), attr_vocab_reverse)

        # For lemma: save hash + flat-string files for numba fast path
        if attr_name == "lemma":
            attr_hashes = np.empty(len(attr_vocabs[attr_name]), dtype=np.uint64)
            for val, idx in attr_vocabs[attr_name].items():
                attr_hashes[idx] = _struct.unpack("<Q", hashlib.md5(val.encode("utf-8")).digest()[:8])[0]
            np.save(os.path.join(db_destination, "attr_lemma_vocab_hashes.npy"), attr_hashes)

            attr_encoded = [attr_vocab_reverse[i].encode("utf-8") for i in range(len(attr_vocab_reverse))]
            attr_offsets = np.empty(len(attr_vocab_reverse) + 1, dtype=np.uint64)
            attr_offsets[0] = 0
            for i, b in enumerate(attr_encoded):
                attr_offsets[i + 1] = attr_offsets[i] + len(b)
            attr_flat = bytearray(int(attr_offsets[-1]))
            p = 0
            for b in attr_encoded:
                attr_flat[p : p + len(b)] = b
                p += len(b)
            np.save(os.path.join(db_destination, "attr_lemma_vocab_offsets.npy"), attr_offsets)
            with open(os.path.join(db_destination, "attr_lemma_vocab_strings.bin"), "wb") as f:
                f.write(attr_flat)

    print(f"{time.ctime()}: Collocation database built successfully.")


def word_frequencies(loader_obj):
    """Generate word frequencies"""
    print("%s: Generating word frequencies..." % time.ctime())
    # Generate frequency table
    os.system(
        f'/bin/bash -c "cut -f 2 <(lz4cat {loader_obj.workdir}/all_words_sorted.lz4) | uniq -c | LANG=C sort -S 25% -rn -k 1,1> {loader_obj.workdir}/all_frequencies"'
    )
    frequencies = loader_obj.destination + "/frequencies"
    os.system("mkdir %s" % frequencies)
    with open(frequencies + "/word_frequencies", "w", encoding="utf8") as output, open(
        loader_obj.workdir + "/all_frequencies", encoding="utf8"
    ) as input:
        for line in input:
            count, word = tuple(line.split())
            print(word + "\t" + count, file=output)
    output.close()


def normalized_word_frequencies(loader_obj):
    """Generate normalized word frequencies"""
    print("%s: Generating normalized word frequencies..." % time.ctime())
    frequencies = loader_obj.destination + "/frequencies"
    with open(frequencies + "/normalized_word_frequencies", "w", encoding="utf8") as output, open(
        frequencies + "/word_frequencies", encoding="utf8"
    ) as input:
        for line in input:
            word, _ = line.split("\t")
            norm_word = word.lower()
            if loader_obj.ascii_conversion is True:
                norm_word = unidecode(norm_word)
            norm_word = "".join(norm_word)
            print(norm_word + "\t" + word, file=output)
    output.close()


def build_norm_word_lmdb(loader_obj):
    """Build LMDB index from normalized_word_frequencies for fast query expansion.

    Maps each normalized form (key) to a NUL-delimited list of original word
    forms (value).
    """
    freq_file = loader_obj.destination + "/frequencies/normalized_word_frequencies"
    lmdb_path = freq_file + ".lmdb"
    tmp_path   = freq_file + ".lmdb.tmp"

    print("%s: Building norm_word LMDB index..." % time.ctime(), flush=True)

    mapping = defaultdict(list)
    with open(freq_file, "rb") as f:
        for line in f:
            tab = line.find(b"\t")
            if tab < 0:
                continue
            norm = line[:tab]
            orig = line[tab + 1:].rstrip(b"\n")
            if norm:
                mapping[norm].append(orig)

    # Write to a temp LMDB (large map_size for writemap), then compact-copy to
    # the final location so disk usage reflects actual data size, not map_size.
    tmp_env = lmdb.open(tmp_path, map_size=2 * 1024 * 1024 * 1024,
                        writemap=True, sync=False, metasync=False)
    with tmp_env.begin(write=True) as txn:
        for norm, originals in mapping.items():
            txn.put(norm, b"\x00".join(originals))
    tmp_env.sync(True)

    os.makedirs(lmdb_path, exist_ok=True)
    tmp_env.copy(lmdb_path, compact=True)
    tmp_env.close()
    os.system(f"rm -rf {tmp_path}")
    print("%s: norm_word LMDB index built (%d keys)." % (time.ctime(), len(mapping)), flush=True)


def build_word_forms_lmdb(loader_obj):
    """Build key-only word_forms.lmdb from lemma/attr flat files.

    Combines lemmas, word_attributes, and lemma_word_attributes into a single
    sorted LMDB with empty values. Used by expand_query_not and expand_autocomplete
    for fast prefix/regex scanning without touching the large words.lmdb.
    Skipped if none of the flat files exist (plain word-only corpus).
    """
    import lmdb

    freq_dir = loader_obj.destination + "/frequencies"
    flat_files = ["lemmas", "word_attributes", "lemma_word_attributes"]
    present = [f for f in flat_files if os.path.exists(os.path.join(freq_dir, f))]
    if not present:
        return

    lmdb_path = freq_dir + "/word_forms.lmdb"
    tmp_path = lmdb_path + ".tmp"
    print("%s: Building word_forms LMDB index..." % time.ctime(), flush=True)

    count = 0
    tmp_env = lmdb.open(tmp_path, map_size=512 * 1024 * 1024,
                        writemap=True, sync=False, metasync=False)
    with tmp_env.begin(write=True) as txn:
        for fname in present:
            with open(os.path.join(freq_dir, fname), "rb") as f:
                for line in f:
                    key = line.rstrip(b"\n")
                    if key:
                        txn.put(key, b"")
                        count += 1
    tmp_env.sync(True)
    os.makedirs(lmdb_path, exist_ok=True)
    tmp_env.copy(lmdb_path, compact=True)
    tmp_env.close()
    os.system(f"rm -rf {tmp_path}")
    print("%s: word_forms LMDB index built (%d keys)." % (time.ctime(), count), flush=True)


def build_metadata_word_index(loader_obj):
    """Build inverted word index LMDB for metadata fields.

    Maps each word found in normalized metadata values back to original values.
    """

    from philologic.runtime.term_expansion import build_metadata_word_index as _build

    print("%s: Building metadata word index LMDB..." % time.ctime(), flush=True)
    n_keys = _build(loader_obj.destination)
    print("%s: Metadata word index built (%d keys)." % (time.ctime(), n_keys), flush=True)


def metadata_frequencies(loader_obj):
    """ "Generate metadata frequencies"""
    print("%s: Generating metadata frequencies..." % time.ctime())
    frequencies = loader_obj.destination + "/frequencies"
    conn = sqlite3.connect(loader_obj.destination + "/toms.db")
    cursor = conn.cursor()
    for field in loader_obj.metadata_fields:
        query = "select %s, count(*) from toms group by %s order by count(%s) desc" % (field, field, field)
        try:
            cursor.execute(query)
            with open(frequencies + "/%s_frequencies" % field, "w", encoding="utf8") as output:
                for result in cursor:
                    if result[0] is not None:
                        val = result[0]
                        try:
                            clean_val = val.replace("\n", " ").replace("\t", "")
                        except AttributeError:  # type is not a string
                            clean_val = f"{val}"
                        print(clean_val + "\t" + str(result[1]), file=output)
        except sqlite3.OperationalError:
            loader_obj.metadata_fields_not_found.append(field)
            if os.path.exists(f"{frequencies}/{field}_frequencies"):
                os.remove(f"{frequencies}/{field}_frequencies")
    if loader_obj.metadata_fields_not_found and loader_obj.debug is True:
        print(
            f"""The following fields were not found in the input corpus {", ".join(loader_obj.metadata_fields_not_found)}"""
        )
    conn.close()
    return loader_obj.metadata_fields_not_found


def normalized_metadata_frequencies(loader_obj):
    """Generate normalized metadata frequencies"""
    print("%s: Generating normalized metadata frequencies..." % time.ctime())
    frequencies = loader_obj.destination + "/frequencies"
    for field in loader_obj.metadata_fields:
        try:
            output = open(frequencies + "/normalized_" + field + "_frequencies", "w", encoding="utf8")
            for line in open(frequencies + "/" + field + "_frequencies", encoding="utf8"):
                word, _ = line.split("\t")
                norm_word = word.lower()
                if loader_obj.ascii_conversion is True:
                    norm_word = unidecode(norm_word)
                norm_word = "".join(norm_word)
                print(norm_word + "\t" + word, file=output)
            output.close()
        except:
            if os.path.exists(f"{frequencies}/normalized_{field}_frequencies"):
                os.remove(f"{frequencies}/normalized_{field}_frequencies")
            pass


def tfidf_per_doc(loader_obj):
    """Get the TF-IDF vectors for each doc"""
    path = os.path.join(loader_obj.destination, "words_and_philo_ids")
    text_object_levels = {"doc": 1, "div1": 2, "div2": 3, "div3": 4, "para": 5}
    text_object_level_int = text_object_levels[loader_obj.default_object_level]

    def get_text(text):
        text_path, token_type = text
        with lz4.frame.open(text_path) as input_file:
            words = []
            current_text_section = None
            text_sections = []
            for line in input_file:
                word_obj = loads(line.decode("utf8"))  # type: ignore
                if word_obj["philo_type"] == "word":
                    text_section_id = list(map(int, word_obj["position"].split()[:text_object_level_int]))
                    if text_section_id != current_text_section:
                        if current_text_section is not None:
                            text_sections.append("#TOK#".join(words))
                            words = []
                        current_text_section = text_section_id
                    if token_type == "word":
                        words.append(word_obj["token"])
                    elif token_type == "lemma":
                        try:
                            words.append(f'lemma_{word_obj["lemma"]}')
                        except KeyError:
                            continue
            if text_section_id:
                text_sections.append("#TOK#".join(words))
        return text_sections

    def get_text_sections(paths):
        pool = mp.Pool(loader_obj.cores)
        total_texts = sum(1 for _ in os.scandir(path))
        with tqdm(total=total_texts, leave=False, desc="Gathering texts") as pbar:
            for text_sections in pool.imap_unordered(get_text, paths):
                for text_section in text_sections:
                    yield text_section
                pbar.update()
        pool.close()

    def get_philo_ids(text_path):
        with lz4.frame.open(text_path) as input_file:
            philo_ids = []
            current_text_section = None
            for line in input_file:
                word_obj = loads(line.decode("utf8"))
                if word_obj["philo_type"] == "word":
                    text_section_id = " ".join(word_obj["position"].split()[:text_object_level_int])
                    if text_section_id != current_text_section:
                        if current_text_section is not None:
                            philo_ids.append(current_text_section)
                        current_text_section = text_section_id
            philo_ids.append(current_text_section)
        return philo_ids

    def get_text_philo_ids(paths):
        pool = mp.Pool(loader_obj.cores)
        total_texts = sum(1 for _ in os.scandir(path))
        with tqdm(total=total_texts, leave=False, desc="Gathering philo_ids") as pbar:
            for philo_ids in pool.imap_unordered(get_philo_ids, [p[0] for p in paths]):
                for philo_id in philo_ids:
                    yield philo_id
                pbar.update()
        pool.close()

    token_types = ["word"]
    if loader_obj.lemma_count > 0:
        token_types.append("lemma")

    os.mkdir(f"{loader_obj.destination}/tfidf")
    for token_type in token_types:
        print(f"{time.ctime()}: Computing IDF score of all {token_type}s in corpus...")
        paths = [(f.path, token_type) for f in os.scandir(path)]
        vectorizer = TfidfVectorizer(sublinear_tf=True, lowercase=False, token_pattern=r"[^#TOK#]+")
        dt_matrix = vectorizer.fit_transform(get_text_sections(paths))
        for philo_id, vector in zip(get_text_philo_ids(paths), dt_matrix):
            document_vector = pd.Series(vector.toarray().flatten(), index=vectorizer.get_feature_names_out())
            document_vector.to_pickle(f"{loader_obj.destination}/tfidf/{philo_id}_{token_type}_idf.pickle")


DefaultPostFilters = [
    word_frequencies,
    normalized_word_frequencies,
    build_norm_word_lmdb,
    build_word_forms_lmdb,
    metadata_frequencies,
    normalized_metadata_frequencies,
    build_metadata_word_index,
    # tfidf_per_doc,
]


def set_default_postfilters(postfilters=DefaultPostFilters):
    """Setting default post filters"""
    filters = []
    for postfilter in postfilters:
        filters.append(postfilter)
    return filters
