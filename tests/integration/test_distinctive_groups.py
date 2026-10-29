"""Integration tests for the distinctive groups: their collocates' work counts in the map_field caches, the explainers
(collocates of more than one work only), and the ranking by how far each group rises above the distinctiveness typical
of its size."""

import math
import os
import sys
import urllib.parse
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "www"))

from philologic.runtime import WebConfig, WSGIHandler  # noqa: E402
from philologic.runtime.reports.collocation import (  # noqa: E402
    collocation_results,
    decode_group_collocates,
    load_collocate_works,
    load_group_hits,
    load_map_field_cache,
)
from scripts.get_outlier_groups import get_outlier_groups, size_neutral_scores  # noqa: E402

pytestmark = pytest.mark.integration

COLLOCATES = {"q": "lemma:love", "colloc_filter_choice": "nofilter", "colloc_within": "sent", "report": "collocation"}


def call(fn, db, **params):
    root = os.path.dirname(os.path.normpath(db.path))
    config = WebConfig(root)
    return fn(WSGIHandler({"QUERY_STRING": urllib.parse.urlencode(params), "PHILOLOGIC_DBPATH": root}, config), config)


def groups(db, file_path):
    tids, counts, bounds, names, count_lemmas = load_map_field_cache(file_path)
    root = os.path.dirname(os.path.normpath(db.path))
    return {name: decode_group_collocates(tids, counts, bounds, g, root, count_lemmas) for g, name in enumerate(names)}


@pytest.fixture
def caches(eltec_db):
    """The collocates of the query by author, and by title: a title is one work in this corpus"""
    by_author = call(collocation_results, eltec_db, **COLLOCATES, map_field="author")["file_path"]
    by_title = call(collocation_results, eltec_db, **COLLOCATES, map_field="title")["file_path"]
    works = Counter(word for counts in groups(eltec_db, by_title).values() for word in counts)
    return by_author, by_title, works


def test_works_counted(eltec_db, caches):
    """Each entry's work count is the number of works its collocate is counted in, whatever the groups"""
    by_author, by_title, works = caches
    for file_path in (by_author, by_title):
        tids, _, bounds, names, _ = load_map_field_cache(file_path)
        stored = load_collocate_works(file_path)
        assert len(stored) == len(tids)
        decoded = groups(eltec_db, file_path)
        for g, name in enumerate(names):
            words = list(decoded[name])  # decoded in the order of the entries
            assert [works[word] for word in words] == stored[bounds[g] : bounds[g + 1]].tolist()


def test_ranking(eltec_db, caches):
    """Explainers are collocates of more than one work, by z-score; groups go by how much more distinctive than their
    size makes typical they are: the sum of their explainers' z-scores against the trend of the groups' sizes"""
    by_author, _, works = caches
    outliers = call(get_outlier_groups, eltec_db, file_path=by_author, min_hits=10, top_n=1000)["outliers"]
    assert len(outliers) >= 3
    hits = dict(zip(load_map_field_cache(by_author)[3], load_group_hits(by_author).tolist()))
    decoded = groups(eltec_db, by_author)
    for name, ratio, words, z_scores in outliers:
        assert hits[name] >= 10
        assert all(works[f"lemma:{word}"] >= 2 for word in words)
        assert z_scores == sorted(z_scores, reverse=True)
    ratios = [ratio for _, ratio, _, _ in outliers]
    assert ratios == sorted(ratios, reverse=True)
    raw = np.array([sum(z for z in z_scores) for _, _, _, z_scores in outliers])
    sizes = np.array([sum(decoded[name].values()) for name, _, _, _ in outliers], dtype=float)
    assert ratios == pytest.approx(np.exp(size_neutral_scores(raw, sizes)).tolist(), abs=2e-3)


def test_size_neutral_scores():
    """On the trend of the log scores by log size, 0; above it, more for a big group than for a small one as far"""
    sizes = np.geomspace(10, 10000, 40)
    noise = np.where(np.arange(40) % 2, 1, -1) * (0.1 + 1 / np.sqrt(sizes))
    scores = 3 * sizes**0.4 * np.exp(noise)
    probes = np.array([20.0, 5000.0])
    all_sizes = np.r_[sizes, probes]
    above = size_neutral_scores(np.r_[scores, 3 * probes**0.4 * math.e], all_sizes)
    assert above[-1] > above[-2] > 0
    # the same for scores all multiplied: a trend's height is no distinctiveness
    assert size_neutral_scores(7 * np.r_[scores, 3 * probes**0.4 * math.e], all_sizes) == pytest.approx(above)
    on_trend = size_neutral_scores(3 * sizes**0.4, sizes)
    assert on_trend == pytest.approx(np.zeros(40), abs=1e-9)
    assert size_neutral_scores(np.array([5.0, 9.0]), np.array([10.0, 50.0])).tolist() == [0.0, 0.0]
