#!/var/lib/philologic5/philologic_env/bin/python3
"""Load Filters used in Loader"""

import gc
import os
from collections import Counter

import lz4.frame
from orjson import dumps, loads
from philologic.loadtime.OHCOVector import Record
from spacy.tokens import Doc as SpacyDoc


# Default filters
def get_word_counts(_, text):
    """Count words"""
    attrib_set = set()
    with open(text["raw"] + ".tmp", "w", encoding="utf8") as tmp_file:
        object_types = ["doc", "div1", "div2", "div3", "para", "sent", "word"]
        counts = [0 for i in range(5)]
        with open(text["raw"], encoding="utf8") as fh:
            for line in fh:
                philo_type, word, philo_id, attrib = line.split("\t")
                philo_id = philo_id.split()
                record = Record(philo_type, word, philo_id)
                record.attrib = loads(attrib)
                for d, _ in enumerate(counts):
                    if philo_type == "word":
                        counts[d] += 1
                    elif philo_type == object_types[d]:
                        record.attrib["word_count"] = counts[d]
                        counts[d] = 0
                print(record, file=tmp_file)
                attrib_set.update(record.attrib.keys())
    os.remove(text["raw"])
    os.rename(text["raw"] + ".tmp", text["raw"])
    return attrib_set


def generate_words_sorted(loader_obj, text):
    """Generate sorted words for storing in index"""
    # -a in grep to avoid issues with NULL chars in file
    wordcommand = 'cat %s | rg -a "^word" | LANG=C sort %s %s > %s' % (
        text["raw"],
        loader_obj.sort_by_word,
        loader_obj.sort_by_id,
        text["words"],
    )
    os.system(wordcommand)


def spacy_tagger(loader_obj, text):
    """Tag words with Spacy"""

    def process_file():
        with open(text["raw"], encoding="utf8") as fh:
            sentence_records = []
            current_sent_id = None
            for line in fh:
                philo_type, word, philo_id, attrib = line.split("\t")
                if philo_type in ("word", "sent", "punct"):
                    sent_id = " ".join(philo_id.split()[:6])
                    record = Record(philo_type, word, philo_id.split())
                    record.attrib = loads(attrib)
                    if current_sent_id is not None and sent_id != current_sent_id:
                        spacy_sentence = SpacyDoc(loader_obj.nlp.vocab, [r.name for r in sentence_records])
                        yield spacy_sentence, sentence_records
                        sentence_records = []
                    sentence_records.append(record)
                    current_sent_id = sent_id
            if sentence_records:
                spacy_sentence = SpacyDoc(loader_obj.nlp.vocab, [r.name for r in sentence_records])
                yield spacy_sentence, sentence_records

    with open(text["raw"] + ".tmp", "w", encoding="utf8") as tmp_file:
        for spacy_sentence, sentence_records in loader_obj.nlp.pipe(process_file(), as_tuples=True, batch_size=128):
            for record, parsed_word in zip(sentence_records, spacy_sentence):
                record.attrib["pos"] = parsed_word.pos_
                record.attrib["tag"] = parsed_word.tag_
                record.attrib["ent_type"] = parsed_word.ent_type_
                record.attrib["lemma"] = parsed_word.lemma_
                print(record, file=tmp_file)

            spacy_sentence.tensor = None
            del spacy_sentence
            del sentence_records
    gc.collect()

    os.remove(text["raw"])
    os.rename(text["raw"] + ".tmp", text["raw"])


def get_lemmas(_, text):
    """Get lemmas for each word"""
    with open(text["raw"] + ".lemma", "w", encoding="utf8") as lemma_file:
        with open(text["raw"], encoding="utf8") as fh:
            for line in fh:
                philo_type, _, philo_id, attribs = line.split("\t")
                if philo_type != "word":
                    continue
                loaded_attribs = loads(attribs)
                if "lemma" in loaded_attribs:
                    print(f"lemma\t{loaded_attribs['lemma']}\t{philo_id}\t{attribs.strip()}", file=lemma_file)
    os.system(f"lz4 -z -q {text['raw']}.lemma {text['raw']}.lemma.lz4 && rm {text['raw']}.lemma")


def make_sorted_toms(*philo_types):
    """Sort metadata before insertion"""

    def sorted_toms(loader_obj, text):
        philo_type_pattern = "|".join("^%s" % t for t in philo_types)
        tomscommand = 'cat %s | rg "%s" | LANG=C sort %s > %s' % (
            text["raw"],
            philo_type_pattern,
            loader_obj.sort_by_id,
            text["sortedtoms"],
        )
        os.system(tomscommand)

    return sorted_toms


def prev_next_obj(*philo_types):
    """Outer function"""
    philo_types = list(philo_types)

    def inner_prev_next_obj(loader_obj, text):
        """Store the previous and next object for every object passed to this function
        By default, this is doc, div1, div2, div3."""
        record_dict = {}
        temp_file = text["raw"] + ".tmp"
        output_file = open(temp_file, "w", encoding="utf8")
        attrib_set = set()
        with open(text["sortedtoms"], encoding="utf8") as filehandle:
            for line in filehandle:
                philo_type, word, philo_id, attrib = line.split("\t")
                philo_id = philo_id.split()
                record = Record(philo_type, word, philo_id)
                record.attrib = loads(attrib)
                if philo_type in record_dict:
                    record_dict[philo_type].attrib["next"] = " ".join(philo_id)
                    if philo_type in philo_types:
                        print(record_dict[philo_type], file=output_file)
                    else:
                        del record_dict[philo_type].attrib["next"]
                        del record_dict[philo_type].attrib["prev"]
                        print(record_dict[philo_type], file=output_file)
                    record.attrib["prev"] = " ".join(record_dict[philo_type].id)
                    record_dict[philo_type] = record
                else:
                    record.attrib["prev"] = ""
                    record_dict[philo_type] = record
                attrib_set.update(record.attrib.keys())

        philo_types.reverse()
        for obj in philo_types:
            try:
                record_dict[obj].attrib["next"] = ""
                print(record_dict[obj], file=output_file)
            except KeyError:
                pass
        output_file.close()
        os.remove(text["sortedtoms"])
        philo_type_pattern = "|".join("^%s" % t for t in loader_obj.types)
        tomscommand = 'cat %s | rg "%s" | LANG=C sort %s > %s' % (
            temp_file,
            philo_type_pattern,
            loader_obj.sort_by_id,
            text["sortedtoms"],
        )
        os.system(tomscommand)
        os.remove(temp_file)
        return attrib_set

    return inner_prev_next_obj


def generate_pages(_, text):
    """Generate separate page file"""
    pagescommand = 'cat %s | rg "^page" > %s' % (text["raw"], text["pages"])
    os.system(pagescommand)


def prev_next_page(_, text):
    """Generate previous and next page"""

    def load_record(line):
        philo_type, word, philo_id, attrib = line.split("\t")
        philo_id = philo_id.split()
        record = Record(philo_type, word, philo_id)
        record.attrib = loads(attrib)
        record.attrib["prev"] = ""
        record.attrib["next"] = ""
        return record

    temp_file = text["pages"] + ".tmp"
    output_file = open(temp_file, "w", encoding="utf8")
    prev_record = None
    next_record = None
    record = None
    with open(text["pages"], encoding="utf8") as filehandle:
        whole_file = filehandle.readlines()
        last_pos = len(whole_file) - 1
        for pos, line in enumerate(whole_file):
            if not record:
                record = load_record(line)
            if prev_record:
                record.attrib["prev"] = " ".join(prev_record.id)
            if pos != last_pos:
                next_record = load_record(whole_file[pos + 1])
                record.attrib["next"] = " ".join(next_record.id)
            print(record, file=output_file)
            prev_record = record
            record = next_record
    output_file.close()
    os.remove(text["pages"])
    os.rename(temp_file, text["pages"])


def generate_refs(_, text):
    """Generate ref file"""
    refscommand = 'cat %s | rg "^ref" > %s' % (text["raw"], text["refs"])
    os.system(refscommand)


def generate_graphics(_, text):
    """Generate graphics file"""
    refscommand = 'cat %s | rg "^graphic" > %s' % (text["raw"], text["graphics"])
    os.system(refscommand)


def generate_lines(_, text):
    """Generate lines file"""
    lines_command = 'cat %s | rg "^line" > %s' % (text["raw"], text["lines"])
    os.system(lines_command)


def suppress_word_attributes(loader_obj, text):
    """Suppress word attributes"""
    with open(text["raw"] + ".tmp", "w", encoding="utf8") as tmp_file:
        with open(text["raw"], encoding="utf8") as fh:
            for line in fh:
                philo_type, word, philo_id, attrib = line.split("\t")
                philo_id = philo_id.split()
                record = Record(philo_type, word, philo_id)
                record.attrib = loads(attrib)
                if philo_type == "word":
                    attrib = loads(attrib)
                    record.attrib = {k: v for k, v in attrib.items() if k not in loader_obj.suppress_word_attributes}
                print(record, file=tmp_file)
    os.remove(text["raw"])
    os.rename(text["raw"] + ".tmp", text["raw"])


def index_word_transformation(transform_function):
    """This function is used to transform words before indexing. It could be anything from transliteration to modernization.
    Takes a function as an argument that transforms the word"""

    def inner_index_word_transformation(_, text):
        with open(text["raw"] + ".tmp", "w", encoding="utf8") as tmp_file:
            with open(text["raw"], encoding="utf8") as fh:
                for line in fh:
                    philo_type, word, philo_id, attrib = line.split("\t")
                    philo_id = philo_id.split()
                    attrib = loads(attrib)
                    if philo_type == "word":
                        word = transform_function(word)
                    record = Record(philo_type, word, philo_id)
                    record.attrib = attrib
                    print(record, file=tmp_file)
        os.remove(text["raw"])
        os.rename(text["raw"] + ".tmp", text["raw"])

    return inner_index_word_transformation


def store_words_and_philo_ids(loader_obj, text):
    """Store words and philo ids file for data-mining"""
    files_path = loader_obj.destination + "/words_and_philo_ids/"
    attributes_to_skip = list(loader_obj.attributes_to_skip)  # make a copy
    attributes_to_skip.remove("lemma")
    attributes_to_skip = set(attributes_to_skip)
    try:
        os.mkdir(files_path)
    except OSError:
        # Path was already created
        pass
    filename = os.path.join(files_path, str(text["id"]))
    with open(filename, "w", encoding="utf8") as output:
        with open(text["raw"], encoding="utf8") as filehandle:
            for line in filehandle:
                philo_type, word, philo_id, attrib = line.split("\t")
                if word == "__philo_virtual":
                    continue
                attrib = loads(attrib)
                if philo_type in ("word", "sent", "punct"):
                    if philo_type == "sent":
                        attrib["start_byte"] = attrib["end_byte"] - len(
                            word.encode("utf8")
                        )  # Parser uses beginning of sent as start_byte
                    word_obj = {
                        "token": word,
                        "position": philo_id,
                        "start_byte": attrib["start_byte"],
                        "end_byte": attrib["end_byte"],
                        "philo_type": philo_type,
                    }
                    word_obj.update({k: v for k, v in attrib.items() if k not in attributes_to_skip})
                    word_obj = dumps(word_obj).decode("utf-8")
                    print(word_obj, file=output)
    with open(f"{filename}.lz4", "wb") as compressed_file:
        with open(filename, "rb") as input_file:
            compressed_file.write(lz4.frame.compress(input_file.read(), compression_level=4))
        os.remove(filename)


def generate_word_frequencies(loader_obj, text):
    """Generate word frequencies for each text object"""

    def inner_generate_word_frequencies():
        word_frequencies_path = os.path.join(loader_obj.destination, "frequencies/text_object_word_frequencies")
        os.makedirs(word_frequencies_path, exist_ok=True)
        para_frequencies = Counter()
        div3_frequencies = Counter()
        div2_frequencies = Counter()
        div1_frequencies = Counter()
        doc_frequencies = Counter()
        with open(text["words"], encoding="utf8") as fh:
            current_para_id = None
            current_div3_id = None
            current_div2_id = None
            current_div1_id = None
            doc_id = None
            for line in fh:
                _, word, philo_id, _ = line.split("\t")
                split_id = philo_id.split()
                para_id = " ".join(split_id[:5])
                div3_id = " ".join(split_id[:4])
                div2_id = " ".join(split_id[:3])
                div1_id = " ".join(split_id[:2])
                doc_id = split_id[0]
                if current_para_id is not None and para_id != current_para_id:
                    para_freq_path = os.path.join(word_frequencies_path, f"{current_para_id}.json")
                    with open(para_freq_path, "wb") as output:
                        output.write(dumps(para_freq_path))
                if current_div3_id is not None and div3_id != current_div3_id:
                    div3_freq_path = os.path.join(word_frequencies_path, f"{current_div3_id}.json")
                    with open(div3_freq_path, "wb") as output:
                        output.write(dumps(div3_freq_path))
                if current_div2_id is not None and div2_id != current_div2_id:
                    div2_freq_path = os.path.join(word_frequencies_path, f"{current_div2_id}.json")
                    with open(div2_freq_path, "wb") as output:
                        output.write(dumps(div2_freq_path))
                if current_div1_id is not None and div1_id != current_div1_id:
                    div1_freq_path = os.path.join(word_frequencies_path, f"{current_div1_id}.json")
                    with open(div1_freq_path, "wb") as output:
                        output.write(dumps(div1_freq_path))
                para_frequencies[word] += 1
                div3_frequencies[word] += 1
                div2_frequencies[word] += 1
                div1_frequencies[word] += 1
                doc_frequencies[word] += 1
                current_para_id = para_id
                current_div3_id = div3_id
                current_div2_id = div2_id
                current_div1_id = div1_id

            doc_freq_path = os.path.join(word_frequencies_path, f"{doc_id}.json")
            with open(doc_freq_path, "wb") as output:
                output.write(dumps(doc_freq_path))

    return inner_generate_word_frequencies


DefaultNavigableObjects = ("doc", "div1", "div2", "div3", "para")
DefaultLoadFilters = [
    get_word_counts,
    make_sorted_toms,
    prev_next_obj,
    generate_pages,
    prev_next_page,
    generate_refs,
    generate_graphics,
    generate_lines,
    get_lemmas,
    generate_words_sorted,
    store_words_and_philo_ids,
]


def set_load_filters(load_filters=DefaultLoadFilters, navigable_objects=DefaultNavigableObjects):
    """Set default filters to run"""
    filters = []
    for load_filter in load_filters:
        if load_filter.__name__ in (
            "make_object_ancestors",
            "make_sorted_toms",
            "prev_next_obj",
            "store_in_plain_text",
        ):
            filters.append(load_filter(*navigable_objects))
        else:
            filters.append(load_filter)
    return filters


def update_navigable_objects(filters, navigable_objects):
    """Update navigable objects"""
    updated_filters = []
    for filter in filters:
        if filter.__name__ in (
            "make_object_ancestors",
            "make_sorted_toms",
            "prev_next_obj",
            "store_in_plain_text",
        ):
            updated_filters.append(filter(*navigable_objects))
        else:
            updated_filters.append(filter)
    return updated_filters
