import hashlib
import io
import os
import sys

import lmdb
from philologic.runtime.DB import DB
from philologic.runtime.Query import filter_philo_ids, get_word_array, split_terms
from philologic.runtime.QuerySyntax import group_terms, parse_query
from philologic.runtime.term_expansion import expand_query_not

OBJECT_LEVEL = {"doc": 6, "div1": 5, "div2": 4, "div3": 3, "para": 2, "sent": 1}
OBJ_DICT = {"doc": 1, "div1": 2, "div2": 3, "div3": 4, "para": 5, "sent": 6, "word": 7}

HIT_SIZE = 9 * 4  # a hit is 9 uint32s, so a raw hit buffer's length gives the hit count


def index_keys(db, query):
    """Return the words.lmdb keys `query` expands to, or None if it needs a real search.

    A single-word query resolves to a set of index keys — one per matching form — whose
    stored hit buffers we can count directly, so no search is needed. Multi-word queries
    are the exception: their hits depend on where the words fall relative to each other,
    which only the phrase/proximity/cooccurrence kernels can work out.
    """
    split = split_terms(group_terms(parse_query(query, query_patterns=db.locals.query_patterns)))
    if len(split) != 1:
        return None
    # Reuse the search's own expansion so the keys — and therefore the counts — are identical.
    expanded = io.StringIO()
    expand_query_not(
        split,
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


def query_word_property(db, query, request):
    """Slow path: run a real search for queries that don't reduce to a single index key."""
    hits = db.query(
        query,
        request["method"],
        request["arg"],
        raw_results=True,
        raw_bytes=True,
        **request.metadata,
    )
    hits.finish()
    result = {"label": query.split(":")[-1], "count": len(hits), "q": query}
    for suffix in ("", ".done", ".terms"):  # we don't want to clog up the server with hitlist files
        try:
            os.remove(hits.filename + suffix)
        except OSError:
            pass
    return result


def get_word_property_count(request, config):
    """Get word property count"""
    db = DB(config.db_path + "/data/")

    word_property_count = []
    if request.word_property != "lemma":
        # Get all word properties from config
        possible_word_properties = config.word_attributes[request.word_property]
        queries = [f"{request.q}:{request.word_property}:{p}" for p in possible_word_properties]

        # Counting hits per property value needs no search unless the query is multi-word: the
        # count falls out of the size of each key's stored hit buffer, optionally filtered
        # against the metadata corpus. Doing this in one thread with one open environment also
        # keeps us from opening words.lmdb concurrently, which LMDB forbids within a process.
        corpus_file, empty_corpus = get_corpus_file(db, request)
        if not empty_corpus:
            keys = {query: index_keys(db, query) for query in queries}

            counts = {}
            direct = [q for q in queries if keys[q] is not None]
            if direct:
                overflow_words = db.locals.overflow_words
                env = lmdb.open(f"{db.path}/words.lmdb", readonly=True, lock=False, readahead=False)
                try:
                    with env.begin(buffers=True) as txn:
                        for query in direct:
                            try:
                                counts[query] = count_hits(txn, keys[query], overflow_words, db.path, corpus_file)
                            except Exception as e:
                                print(f"Exception occurred during processing {query}: {e}", file=sys.stderr)
                finally:
                    env.close()

            # Searches open words.lmdb themselves, so they have to run once ours is closed.
            for query in queries:
                if keys[query] is not None:
                    continue
                try:
                    counts[query] = query_word_property(db, query, request)["count"]
                except Exception as e:
                    print(f"Exception occurred during processing {query}: {e}", file=sys.stderr)

            word_property_count = [
                {"label": query.split(":")[-1], "count": counts[query], "q": query}
                for query in queries
                if counts.get(query, 0) > 0
            ]
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
        lemma_db_env = lmdb.open(f"{config.db_path}/data/lemmas.lmdb", readonly=True, lock=False)
        lemma_count = {}
        total_count_per_lemma = {}
        with lemma_db_env.begin() as txn:
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

    results = {"query": dict([i for i in request]), "results": word_property_count}
    return results
