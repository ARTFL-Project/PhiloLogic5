"""Integration tests for similar usage: the groups whose collocates, outside the part of the corpus a search's metadata
selects, over-use what that part's over-use (cosine of the positive fightin' words z-scores against the rest of the
corpus), and the shared collocates that explain it."""

import math
import os
import sqlite3
import sys
import urllib.parse
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "www"))

from philologic.runtime import WebConfig, WSGIHandler  # noqa: E402
from philologic.runtime.reports.collocation import (  # noqa: E402
    collocation_results,
    decode_group_collocates,
    load_map_field_cache,
    safe_pickle_load,
)
from scripts.get_similar_collocate_distributions import get_similar_collocate_distributions  # noqa: E402

pytestmark = pytest.mark.integration

COLLOCATES = {"q": "lemma:love", "colloc_filter_choice": "nofilter", "colloc_within": "sent", "report": "collocation"}


def call(fn, db, **params):
    root = os.path.dirname(os.path.normpath(db.path))
    config = WebConfig(root)
    return fn(WSGIHandler({"QUERY_STRING": urllib.parse.urlencode(params), "PHILOLOGIC_DBPATH": root}, config), config)


def similar_usage(db, field, **metadata):
    """The similar usage of the search with metadata by field, and the counts it is made from"""
    primary = call(collocation_results, db, **COLLOCATES, **metadata)["file_path"]
    inside = call(collocation_results, db, **COLLOCATES, **metadata, map_field=field)["file_path"]
    whole = call(collocation_results, db, **COLLOCATES, map_field=field)["file_path"]
    similar = call(
        get_similar_collocate_distributions, db, primary_file_path=primary, file_path=whole, inside_file_path=inside
    )["similar"]
    return similar, safe_pickle_load(primary), groups(db, inside), groups(db, whole)


def groups(db, file_path):
    tids, counts, bounds, names, count_lemmas = load_map_field_cache(file_path)
    root = os.path.dirname(os.path.normpath(db.path))
    return {name: decode_group_collocates(tids, counts, bounds, g, root, count_lemmas) for g, name in enumerate(names)}


def z_positive(y_a, y_b, n_a, n_b):
    """max(0, z) of the fightin' words log-odds with the informative prior of the pooled counts (Monroe et al. 2008)"""
    delta = math.log((2 * y_a + y_b) / (2 * n_a + n_b)) - math.log((y_a + 2 * y_b) / (n_a + 2 * n_b))
    return max(0.0, delta / math.sqrt(1 / (2 * y_a + y_b) + 1 / (y_a + 2 * y_b)))


def expected(reference, inside, whole):
    """Each group outside the reference with its similarity, and its z-score profile and the reference's"""
    outside = {name: whole[name] - inside.get(name, Counter()) for name in whole}
    outside = {name: counts for name, counts in outside.items() if counts}
    corpus = Counter(reference)
    for counts in outside.values():
        corpus.update(counts)
    total = sum(corpus.values())

    def profile(counts):
        n = sum(counts.values())
        return {word: z_positive(c, corpus[word] - c, n, total - n) for word, c in counts.items()}

    ref = profile(reference)
    ref_norm = math.sqrt(sum(z * z for z in ref.values()))
    result = {}
    for name, counts in outside.items():
        group = profile(counts)
        norm = math.sqrt(sum(z * z for z in group.values()))
        dot = sum(z * ref.get(word, 0.0) for word, z in group.items())
        result[name] = (dot / (norm * ref_norm) if dot > 0 else 0.0, group)
    return result, ref


def check(similar, reference, inside, whole):
    want, ref = expected(reference, inside, whole)
    alike = [s for s, _ in want.values() if s > 0]
    assert len(similar) == min(50, len(alike))
    similarities = [similarity for _, similarity, _ in similar]
    assert similarities == sorted(similarities, reverse=True)
    # none more alike left out
    assert not {name for name, (s, _) in want.items() if s > similarities[-1] + 1e-3} - {name for name, _, _ in similar}
    for name, similarity, words in similar:
        assert similarity == pytest.approx(want[name][0], abs=1e-3)
        group = want[name][1]
        parts = [group[f"lemma:{word}"] * ref[f"lemma:{word}"] for word in words]
        assert words and all(part > 0 for part in parts) and parts == sorted(parts, reverse=True)
    return want


def test_own_group_left_out(eltec_db):
    """An author's search compared by author: the others, the author's own group being all inside the search"""
    con = sqlite3.connect(os.path.join(eltec_db.path, "toms.db"))
    author = con.execute("select author from toms where philo_type='doc' and author like 'Dickens%'").fetchone()[0]
    similar, reference, inside, whole = similar_usage(eltec_db, "author", author=f'"{author}"')
    assert similar and author not in {name for name, _, _ in similar}
    check(similar, reference, inside, whole)


def test_groups_partly_inside(eltec_db):
    """A period compared by author: authors all inside it are left out, the others are taken outside it"""
    similar, reference, inside, whole = similar_usage(eltec_db, "author", year="1850-1870")
    want = check(similar, reference, inside, whole)
    all_inside = {name for name in inside if not whole[name] - inside[name]}
    partly_inside = {name for name in inside if whole[name] - inside[name]}
    assert all_inside and partly_inside
    assert not all_inside & set(want)
    assert partly_inside <= set(want)


def test_needs_a_part_of_the_corpus(eltec_db):
    """The whole corpus has no rest to compare it with"""
    similar, _, _, _ = similar_usage(eltec_db, "author")
    assert similar == []
