#!/usr/bin/env python3


import os
import sqlite3
import struct
import time
from unidecode import unidecode
from pickle import dump

import multiprocess as mp
import lmdb
import lz4.frame
import msgpack
from orjson import loads
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


def make_sql_table(table, file_in, db_file="toms.db", indices=None, depth=7, verbose=True):
    """SQL Loader function"""

    def inner_make_sql_table(loader_obj):
        if verbose is True:
            print(f"{time.ctime()}: Loading the {table} SQLite table...")
        else:
            print(f"Loading the {table} SQLite table...")
        db_destination = os.path.join(loader_obj.destination, db_file)
        line_count = sum(1 for _ in open(file_in, "rbU"))
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


def make_sentences_database(loader_obj, db_destination):
    """Generate an LMDB database where keys are sentence IDs and values the associated sentence containing all the words in it"""
    print(f"{time.ctime()}: Loading the sentences LMDB database...")
    attributes_to_skip = list(loader_obj.attributes_to_skip)
    attributes_to_skip.remove("lemma")
    attributes_to_skip = set(attributes_to_skip)
    attributes_to_skip.update({"token", "position", "philo_type"})
    with tqdm(total=loader_obj.word_count, leave=False) as pbar:
        env = lmdb.open(db_destination, map_size=2 * 1024 * 1024 * 1024 * 1024, writemap=True)  # 2TB
        count = 0
        with env.begin(write=True) as txn:
            for raw_words in os.scandir(f"{loader_obj.destination}/words_and_philo_ids"):
                sentence_id = ""
                with lz4.frame.open(raw_words.path) as input_file:
                    current_sentence = None
                    words = []
                    for line in input_file:
                        word_obj = loads(line.decode("utf8"))  # type: ignore
                        if word_obj["philo_type"] == "word":
                            sentence_id = struct.pack("6I", *map(int, word_obj["position"].split()[:6]))
                            if sentence_id != current_sentence:
                                if current_sentence is not None:
                                    txn.put(current_sentence, msgpack.dumps(words))
                                    words = []
                                    count += 1
                                current_sentence = sentence_id
                            attribs = {k: v for k, v in word_obj.items() if k not in attributes_to_skip}
                            if "lemma" in attribs:
                                attribs["lemma"] = f'lemma:{attribs["lemma"]}'
                            words.append(
                                (
                                    word_obj["token"],
                                    word_obj["start_byte"],
                                    int(word_obj["position"].split()[6]),
                                    attribs,
                                )
                            )
                            pbar.update()
                    if sentence_id:
                        txn.put(sentence_id, msgpack.dumps(words))
        env.close()


def word_frequencies(loader_obj):
    """Generate word frequencies"""
    print("%s: Generating word frequencies..." % time.ctime())
    # Generate frequency table
    os.system(
        f'/bin/bash -c "cut -f 2 <(lz4cat {loader_obj.workdir}/all_words_sorted.lz4) | uniq -c | LANG=C sort -S 25% -rn -k 1,1> {loader_obj.workdir}/all_frequencies"'
    )
    frequencies = loader_obj.destination + "/frequencies"
    os.system("mkdir %s" % frequencies)
    output = open(frequencies + "/word_frequencies", "w", encoding="utf8")
    for line in open(loader_obj.destination + "/WORK/all_frequencies"):
        count, word = tuple(line.split())
        print(word + "\t" + count, file=output)
    output.close()


def normalized_word_frequencies(loader_obj):
    """Generate normalized word frequencies"""
    print("%s: Generating normalized word frequencies..." % time.ctime())
    frequencies = loader_obj.destination + "/frequencies"
    output = open(frequencies + "/normalized_word_frequencies", "w", encoding="utf8")
    for line in open(frequencies + "/word_frequencies"):
        word, _ = line.split("\t")
        norm_word = word.lower()
        if loader_obj.ascii_conversion is True:
            norm_word = unidecode(norm_word)
        norm_word = "".join(norm_word)
        print(norm_word + "\t" + word, file=output)
    output.close()


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
            with open(frequencies + "/%s_frequencies" % field, "w") as output:
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
            output = open(frequencies + "/normalized_" + field + "_frequencies", "w")
            for line in open(frequencies + "/" + field + "_frequencies"):
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


def idf_per_word(loader_obj):
    """Get the IDF weighting of every word in corpus"""
    path = os.path.join(loader_obj.destination, "words_and_philo_ids")

    def get_text(text):
        text_path, token_type = text
        with lz4.frame.open(text_path) as input_file:
            words = []
            current_sentence = None
            sentences = []
            for line in input_file:
                word_obj = loads(line.decode("utf8"))  # type: ignore
                if word_obj["philo_type"] == "word":
                    sentence_id = struct.pack("6I", *map(int, word_obj["position"].split()[:6]))
                    if sentence_id != current_sentence:
                        if current_sentence is not None:
                            sentences.append(" ".join(words))
                            words = []
                        current_sentence = sentence_id
                    if token_type == "word":
                        words.append(word_obj["token"])
                    elif token_type == "lemma":
                        words.append(word_obj["lemma"])
            if sentence_id:
                sentences.append(" ".join(words))
        return sentences

    def get_sentences(token_type="word"):
        pool = mp.Pool(loader_obj.cores)
        total_texts = sum(1 for _ in os.scandir(path))
        with tqdm(total=total_texts, leave=False) as pbar:
            for sentences in pool.imap_unordered(get_text, [(f.path, token_type) for f in os.scandir(path)]):
                for sentence in sentences:
                    yield sentence
                pbar.update()
        pool.close()

    for token_type in ["word", "lemma"]:
        print(f"{time.ctime()}: Computing IDF score of all {token_type}s in corpus...")
        vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r"\w+")
        vectorizer.fit(get_sentences())
        with open(os.path.join(loader_obj.destination, f"frequencies/{token_type}_idf.pickle"), "wb") as idf_output:
            idf = {word: vectorizer.idf_[i] for word, i in vectorizer.vocabulary_.items()}
            dump(idf, idf_output)


DefaultPostFilters = [
    word_frequencies,
    normalized_word_frequencies,
    metadata_frequencies,
    normalized_metadata_frequencies,
    idf_per_word,
]


def set_default_postfilters(postfilters=DefaultPostFilters):
    """Setting default post filters"""
    filters = []
    for postfilter in postfilters:
        filters.append(postfilter)
    return filters
