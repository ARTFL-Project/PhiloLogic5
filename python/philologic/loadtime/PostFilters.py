#!/var/lib/philologic5/philologic_env/bin/python3


import os
import sqlite3
import struct
import time

import lmdb
import lz4.frame
import msgspec
import multiprocess as mp
import pandas as pd
from orjson import loads
from philologic.utils import count_lines
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from unidecode import unidecode


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


def make_sentences_database(loader_obj, db_destination):
    """Generate an LMDB database where keys are sentence IDs and values the associated sentence containing all the words in it"""
    print(f"{time.ctime()}: Loading the sentences LMDB database...")
    temp_destination = f"{db_destination}_temp"

    msgpack = msgspec.msgpack.Encoder()
    Word = msgspec.defstruct(
        "Word", [("word", str), ("position", int)] + [(k, str) for k in loader_obj.word_attributes], array_like=True
    )
    Sentence = msgspec.defstruct("Sentence", [("words", list[Word])], array_like=True)

    with tqdm(total=loader_obj.word_count, leave=False) as pbar:
        env = lmdb.open(temp_destination, map_size=2 * 1024 * 1024 * 1024 * 1024, writemap=True, sync=False)  # 2TB
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
                                    txn.put(current_sentence, msgpack.encode(Sentence(words=words)))
                                    words = []
                                    count += 1
                                current_sentence = sentence_id
                            if "lemma" in word_obj:
                                word_obj["lemma"] = f'lemma:{word_obj["lemma"]}'
                            word_args = [word_obj["token"], int(word_obj["position"].split()[6])] + [
                                word_obj.get(k, "") for k in loader_obj.word_attributes
                            ]
                            words.append(Word(*word_args))
                            pbar.update()
                    if sentence_id:
                        txn.put(sentence_id, msgpack.encode(Sentence(words=words)))
        pbar.close()  # Make sure to clear the tqdm bar
        print(f"{time.ctime()}: Optimizing the sentences index for space...")
        os.mkdir(db_destination)
        env.sync(True) # Ensure all data is written to disk before compacting database
        env.copy(db_destination, compact=True)
        env.close()
        os.system(f"rm -r {temp_destination}")


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
    metadata_frequencies,
    normalized_metadata_frequencies,
    # tfidf_per_doc,
]


def set_default_postfilters(postfilters=DefaultPostFilters):
    """Setting default post filters"""
    filters = []
    for postfilter in postfilters:
        filters.append(postfilter)
    return filters
