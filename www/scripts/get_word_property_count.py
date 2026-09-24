import hashlib
import io
import os
import sys

from philologic.runtime.DB import DB
from philologic.runtime.lmdb_env import lmdb_env
from philologic.runtime.Query import filter_philo_ids, get_word_array, split_terms
from philologic.runtime.QuerySyntax import group_terms, parse_query
from philologic.runtime.term_expansion import expand_query_not

OBJECT_LEVEL = {"doc": 6, "div1": 5, "div2": 4, "div3": 3, "para": 2, "sent": 1}
OBJ_DICT = {"doc": 1, "div1": 2, "div2": 3, "div3": 4, "para": 5, "sent": 6, "word": 7}

HIT_SIZE = 9 * 4  # a hit is 9 uint32s, so a raw hit buffer's length gives the hit count

# Term kinds that can carry a word property, and the kind they become once they do
PROPERTY_KINDS = {"TERM": "ATTR", "QUOTE": "ATTR", "LEMMA": "LEMMA_ATTR"}


def with_property(group, word_property, value):
    """Rewrite a single-word term group so that each of its terms carries word_property=value.

    `love | light` becomes `love:pos:NOUN | light:pos:NOUN`, `lemma:love` becomes `lemma:love:pos:NOUN`
    and a quoted word, which stands for that exact form, becomes `love:pos:NOUN`. Returns None if the
    group can't be broken down by a property, e.g. because it already specifies one.
    """
    rewritten = []
    for kind, token in group:
        if kind in ("OR", "NOT"):
            rewritten.append((kind, token))
        elif kind in PROPERTY_KINDS:
            word = token.strip('"')
            rewritten.append((PROPERTY_KINDS[kind], f"{word}:{word_property}:{value}"))
        else:
            return None
    return rewritten


def index_keys(db, group):
    """Return the words.lmdb keys a single-word term group expands to.

    Such a group resolves to a set of index keys — one per matching form — whose stored hit
    buffers we can count directly, so no search is needed.
    """
    # Reuse the search's own expansion so the keys — and therefore the counts — are identical.
    expanded = io.StringIO()
    expand_query_not(
        [group],
        f"{db.path}/frequencies/normalized_word_frequencies",
        expanded,
        db.locals.ascii_conversion,
        db.locals["lowercase_index"],
    )
    return [key for key in expanded.getvalue().split("\n") if key]


def count_hits(txn, keys, overflow_words, db_path, corpus_file):
    """Count the hits stored under `keys`, restricted to `corpus_file` when there is one.

    Each key holds the hits for one word form, so the forms are disjoint and the counts add up
    the same way the search's merge of those arrays would.
    """
    count = 0
    for key in keys:
        if corpus_file is None:
            # No metadata filter: the hit count is just the size of the stored hit buffer,
            # so we never have to materialize the array.
            if key in overflow_words:
                path = os.path.join(
                    db_path, "overflow_words", f"{hashlib.sha256(key.encode('utf8')).hexdigest()}.bin"
                )
                try:
                    count += os.path.getsize(path) // HIT_SIZE
                except OSError:
                    pass
            else:
                buffer = txn.get(key.encode("utf8"))
                if buffer is not None:
                    count += len(buffer) // HIT_SIZE
            continue
        word_array = get_word_array(txn, key, overflow_words, db_path)
        if len(word_array):
            count += len(filter_philo_ids(corpus_file, word_array))
    return count


def has_metadata(metadata):
    """Mirror DB.query's test for whether any metadata field is actually set."""
    for value in metadata.values():
        if isinstance(value, str):
            if not value:
                continue
            value = [value]
        if any(v for v in value):
            return True
    return False


def get_corpus_file(db, request):
    """Resolve the metadata filter to a corpus hitlist once, shared by every property value.

    Returns (corpus_file, empty): `corpus_file` is None when no metadata is set, and
    `empty` is True when the metadata matches nothing, in which case every count is 0.
    """
    if not has_metadata(request.metadata):
        return None, False
    # Querying with an empty query string returns the metadata corpus itself. DB.query hashes
    # the metadata before the query string, so this is the same cached corpus file that the
    # per-property queries would each have rebuilt.
    corpus = db.query("", request["method"], request["arg"], raw_results=True, **request.metadata)
    corpus.finish()
    if len(corpus) == 0 or not getattr(corpus, "filename", None):
        return None, True
    return corpus.filename, False


def get_word_property_count(request, config):
    """Get word property count"""
    db = DB(config.db_path + "/data/")
    results = {"query": dict([i for i in request]), "results": []}

    # Word properties describe single words: the words of a multi-word query each have their own
    split = split_terms(group_terms(parse_query(request.q, query_patterns=db.locals.query_patterns)))
    if len(split) != 1:
        return results

    word_property_count = []
    if request.word_property != "lemma":
        # Get all word properties from config
        possible_word_properties = config.word_attributes[request.word_property]
        groups = {value: with_property(split[0], request.word_property, value) for value in possible_word_properties}

        # Counting hits per property value needs no search: the count falls out of the size of each
        # key's stored hit buffer, optionally filtered against the metadata corpus.
        corpus_file, empty_corpus = get_corpus_file(db, request)
        if not empty_corpus and None not in groups.values():
            keys = {value: index_keys(db, group) for value, group in groups.items()}
            overflow_words = db.locals.overflow_words
            with lmdb_env(f"{db.path}/words.lmdb") as env, env.begin(buffers=True) as txn:
                for value, group in groups.items():
                    query = " ".join(token for _, token in group)
                    try:
                        count = count_hits(txn, keys[value], overflow_words, db.path, corpus_file)
                    except Exception as e:
                        print(f"Exception occurred during processing {query}: {e}", file=sys.stderr)
                        continue
                    if count > 0:
                        word_property_count.append({"label": value, "count": count, "q": query})
    else:
        # Get all lemmas
        hits = db.query(
            request.q,
            request["method"],
            request["arg"],
            raw_results=True,
            raw_bytes=True,
            **request.metadata,
        )
        lemma_count = {}
        total_count_per_lemma = {}
        with lmdb_env(f"{config.db_path}/data/lemmas.lmdb") as lemma_db_env, lemma_db_env.begin() as txn:
            for hit in hits:
                lemma = txn.get(hit)
                if lemma is not None:  # some hits may not have corresponding lemmas
                    lemma = lemma.decode("utf8")
                    if lemma in lemma_count:
                        lemma_count[lemma] += 1
                    else:
                        lemma_count[lemma] = 1
        word_property_count = [{"label": k.replace("lemma:", ""), "count": v, "q": k} for k, v in lemma_count.items()]

    word_property_count.sort(key=lambda x: x["count"], reverse=True)
    results["results"] = word_property_count
    return results
