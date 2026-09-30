"""Previews of a load on a sample of its files, with the options of the load pages: the document metadata found by
doc_xpaths in the TEI headers (with the loader's own functions), the tags of the texts and how the load would treat
them, the words and sentences the parser finds, and the order of the documents."""

import io
import multiprocessing
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

import lxml.etree
import orjson

from philologic.loadtime import Parser
from philologic.loadtime.Loader import Loader, tei_header, tei_header_metadata
from philologic.utils import sort_list
from philologic.webui_loader import load_schema

MAX_PREVIEW_FILE = 50 * 1024**2  # larger files aren't previewed
PREVIEW_TIMEOUT = 120
MAX_HEADER_TEXT = 20000
START_TAG = re.compile(r"<([A-Za-z_][\w.:-]*)((?:\s+[^>]*?)?)\s*/?>")
ATTRIBUTE = re.compile(r"([\w.:-]+)\s*=")
TEXT_START = re.compile(r"</teiheader\s*>", re.I)


class PreviewTimeout(Exception):
    """A preview which took too long"""


def run_isolated(function, *args, timeout=PREVIEW_TIMEOUT):
    """Run a preview in a process of its own, killed if it takes longer than timeout seconds (a regular expression of
    the options can take forever), so that the server isn't held up"""
    context = multiprocessing.get_context("forkserver")
    executor = ProcessPoolExecutor(max_workers=1, mp_context=context)
    try:
        return executor.submit(function, *args).result(timeout=timeout)
    except FuturesTimeout as error:
        for process in list(executor._processes.values()):
            process.kill()
        raise PreviewTimeout(f"the preview took more than {timeout} seconds (a slow regular expression?)") from error
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def sample(paths, count):
    """count files spread over the list"""
    if len(paths) <= count:
        return list(paths)
    step = len(paths) / count
    return [paths[int(index * step)] for index in range(count)]


def read_text(path, newline=None):
    if os.path.getsize(path) > MAX_PREVIEW_FILE:
        raise ValueError(f"larger than {MAX_PREVIEW_FILE // 1024**2} MB")
    with open(path, encoding="utf8", newline=newline) as text_file:
        return text_file.read()


def option(options, key):
    return options[key] if key in options else load_schema.default(key)


def header_preview(paths, options, count=10):
    """For each sampled file: the metadata found, the xpath which found each field, the year, and its header"""
    doc_xpaths = option(options, "doc_xpaths")
    sql_types = option(options, "metadata_sql_types")
    rows = []
    for path in sample(paths, count):
        row = {"file": os.path.basename(path), "metadata": {}, "xpaths": {}, "error": None, "header": None}
        try:
            text = read_text(path)
        except (OSError, UnicodeDecodeError, ValueError) as error:
            row["error"] = f"can't be read: {error}"
            rows.append(row)
            continue
        header = tei_header(text)
        if header is None:
            row["error"] = "no TEI header: the file would be left out"
            rows.append(row)
            continue
        row["header"] = header[:MAX_HEADER_TEXT]
        matched = {}
        try:
            metadata = tei_header_metadata(header, doc_xpaths, sql_types, matched)
        except lxml.etree.XMLSyntaxError:
            row["error"] = "invalid XML in the header: the file would be left out"
            rows.append(row)
            continue
        metadata["filename"] = os.path.basename(path)
        metadata = Loader.create_year_field(metadata)
        row["metadata"] = {key: str(value) if value is not None else None for key, value in metadata.items()}
        row["xpaths"] = matched
        rows.append(row)
    fields = list(doc_xpaths)
    found = Counter(field for row in rows for field in row["metadata"] if field in doc_xpaths)
    sort_order = option(options, "sort_order")
    sortable = [row for row in rows if not row["error"]]
    if sort_order:
        # sort_list needs every field, as the loader's metadata has
        keys = [{field: "" for field in sort_order} | row["metadata"] | {"_file": row["file"]} for row in sortable]
        order = [key["_file"] for key in sort_list(keys, sort_order)]
    else:
        order = [row["file"] for row in sortable]
    return {"rows": rows, "fields": fields, "found": dict(found), "sorted": order}


def tag_census(paths, options, count=10):
    """The tags of the texts of sampled files (after their TEI header), with their count and attributes, and how the
    load would treat them: the object type they map to, suppressed, or not breaking words (tag_exceptions)"""
    tag_map = option(options, "tag_to_obj_map")
    suppressed = set(option(options, "suppress_tags"))
    exceptions = [
        re.compile(pattern, re.I)
        for pattern in option(options, "tag_exceptions")
        if load_schema.regex_error(pattern) is None
    ]
    counts = Counter()
    attributes = defaultdict(Counter)
    examples = {}
    errors = []
    for path in sample(paths, count):
        try:
            text = read_text(path)
        except (OSError, UnicodeDecodeError, ValueError) as error:
            errors.append({"file": os.path.basename(path), "error": str(error)})
            continue
        end_of_header = TEXT_START.search(text)
        body = text[end_of_header.end() :] if end_of_header else text
        for match in START_TAG.finditer(body):
            name = match.group(1)
            counts[name] += 1
            for attribute in ATTRIBUTE.findall(match.group(2)):
                attributes[name][attribute] += 1
            if name not in examples:
                examples[name] = match.group(0)[:200]
    tags = []
    for name, number in counts.most_common():
        example = examples[name]
        tags.append(
            {
                "tag": name,
                "count": number,
                "attributes": dict(attributes[name].most_common(10)),
                "mapped_to": tag_map.get(name) or tag_map.get(name.lower()),
                "suppressed": name in suppressed or name == "gap",
                "exception": any(pattern.search(example) for pattern in exceptions),
                "example": example,
            }
        )
    return {"tags": tags, "errors": errors, "files": min(count, len(paths))}


def parser_options(options):
    """Options of the parser, as the loader gives them to it (Loader.parse_files)"""
    keys = (
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
    )
    return {key: load_schema.to_config_value(key, option(options, key)) for key in keys}


def tokens_preview(path, options, max_words=400):
    """The first words of a file, by sentence, as the parser finds them with the options (the default XML parser,
    without load filters such as a spaCy model), with the number of words, sentences and punctuation marks"""
    text = read_text(path, newline="")  # as the loader reads it: byte offsets are those of the file
    output = io.StringIO()
    parser = Parser.XMLParser(
        output,
        1,
        os.path.getsize(path),
        known_metadata={"filename": os.path.basename(path)},
        tag_to_obj_map=option(options, "tag_to_obj_map"),
        metadata_to_parse=option(options, "metadata_to_parse"),
        words_to_index=set(),
        file_type="xml",
        lemmas=None,
        **parser_options(options),
    )
    parser.parse(io.StringIO(text, newline=""))
    sentences = []
    current_sentence = None
    counts = Counter()
    words_shown = 0
    for line in output.getvalue().splitlines():
        kind, _, rest = line.partition("\t")
        if kind not in ("word", "punct", "sent"):
            continue
        counts[kind] += 1
        if words_shown >= max_words or kind == "sent":
            continue
        token, philo_id, attributes = rest.split("\t", 2)
        sentence_id = " ".join(philo_id.split()[:6])
        # A sentence starts with a word: punctuation after the end of a sentence goes with it
        if sentence_id != current_sentence and (kind == "word" or not sentences):
            current_sentence = sentence_id
            sentences.append({"id": sentence_id, "tokens": []})
        entry = {"text": token, "kind": kind}
        if kind == "word":
            words_shown += 1
            entry["start"] = orjson.loads(attributes).get("start_byte")
        sentences[-1]["tokens"].append(entry)
    return {
        "file": os.path.basename(path),
        "sentences": sentences,
        "words": counts["word"],
        "sentence_count": counts["sent"],
        "punctuation": counts["punct"],
    }
