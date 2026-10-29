"""Representative passages — passages within a group whose immediate context
contains the most of the group's distinctive-collocate signature.

Approach: a cooc query ("main query AND (sig1 | sig2 | ... | sigK)") run with
the same metadata filters plus the drill-down restriction. PhiloLogic's query
engine handles lemma expansion, ASCII folding and case for free. Each hit row
carries the byte offsets and word positions of every matched word (each word of
the main query, quoted phrases included, then the signature word); we merge
hits by sentence philo_id, recover the signature token id at each signature
position from the corpus per-position arrays, sum z-weights over distinct
matched signature tokens (formula A1), and re-sort by weighted score.

Highlighting uses format_concordance_distinctive, which emits two CSS classes
("highlight" for the main word, "colloc-explainer" for signature words).
"""

import hashlib
import os
import re
import struct

import numpy as np

from philologic.runtime.citations import citation_links, citations
from philologic.runtime.DB import DB, LEVEL_MAP, hit_to_string
from philologic.runtime.get_text import _get_mmap, adjust_bytes
from philologic.runtime.ObjectFormatter import format_concordance_distinctive
from philologic.runtime.Query import get_word_groups, resolve_method
from philologic.runtime.reports.collocation import collocate_filters, collocate_keys
from philologic.runtime.word_properties import property_names, split_properties


def get_representative_passages(request, config):
    db = DB(config.db_path + "/data/")

    top_n = _clamp_int(request, "top_n", default=25, low=1, high=100)
    offset = _clamp_int(request, "offset", default=0, low=0, high=10000)

    sig_words = _split(getattr(request, "signature_tokens", "") or "")
    sig_weights = _split_floats(getattr(request, "signature_weights", "") or "")
    if sig_weights and len(sig_weights) != len(sig_words):
        sig_weights = []
    if not sig_words:
        return _empty_response(request)

    rf = (getattr(request, "restrict_to_field", "") or "").strip()
    rv = (getattr(request, "restrict_to_value", "") or "").strip()
    if not rf or not rv:
        return _empty_response(request)

    main_q = (request["q"] or "").strip()
    if not main_q:
        return _empty_response(request)
    filters = collocate_filters(request, db)  # as the collocation report counted the signature's collocates
    count_lemmas = filters.count_lemmas

    # Resolve signature tokens to their corpus token ids.
    sig_keys = [(f"lemma:{w}" if count_lemmas else w) for w in sig_words]
    sig_tid_lookup = _build_sig_tid_lookup(config.db_path, sig_keys, count_lemmas)
    if not sig_tid_lookup:
        return _empty_response(request)

    # Per-tid weight (z-score). Tokens not in signature get implicit 0.
    sig_weight_by_tid = {}
    for i, key in enumerate(sig_keys):
        tid = sig_tid_lookup.get(key)
        if tid is None:
            continue
        sig_weight_by_tid[tid] = float(sig_weights[i]) if sig_weights else 1.0

    # Build cooc query: "main_q sig1|sig2|..." (lemma: prefix for both sides
    # when applicable). PhiloLogic parses space-separated tokens as a cooc
    # AND-group and `|` as an inline OR within each group. Each signature word is searched as it was counted: with
    # the kept values of the word properties of the filters.
    sig_or = "|".join(key for sig_key in sig_keys for key in collocate_keys(sig_key, filters))
    cooc_q = f"{main_q} {sig_or}"
    raw_method = "proxy" if (request.colloc_within or "") == "n" else "sentence"
    raw_arg = request.method_arg if raw_method == "proxy" else ""
    # resolve_method handles method normalization (e.g. "sentence" -> "sentence_unordered")
    # the way WSGIHandler does it for web requests.
    method, method_arg = resolve_method(cooc_q, raw_method, raw_arg, "no", db.locals.query_patterns)

    # Merge restrict_to_* into the metadata filters for this query.
    metadata = dict(request.metadata or {})
    metadata[rf] = rv

    hits = db.query(cooc_q, method, method_arg, **metadata)
    hits.finish()
    if len(hits) == 0:
        return _empty_response(request, total_in_group=0)

    # Build the set of token ids for the main query's words from the cooc query's
    # expansion (every group of the .terms file but the last, the signature's). This
    # handles regex/case-variants the same way the cooc engine does — ensures we can
    # tell main from sig at each hit position even when a main word is also in the
    # signature (e.g. main_q="lemma:citoyen" with "citoyen" present as a sig token).
    main_tids = _resolve_main_tids(config.db_path, hits.filename)

    # Bulk-read the multi-word hitlist as a uint32 matrix: for n words (the main
    # query's, then the signature's), length = 9 + 2*(n-1):
    #   [0..5]  sentence philo_id
    #   [6]     page
    #   [7]     word_pos of the first word, [8] its byte_offset
    #   [9]     word_pos of the next word, [10] its byte_offset, and so on
    with hits.open_raw() as f:
        raw = f.read()
    if not raw:
        return _empty_response(request, total_in_group=0)
    rows = np.frombuffer(raw, dtype=np.uint32).reshape(-1, hits.length)

    # Resolve sentence indices for each hit (for sig token-id lookup).
    colloc_dir = os.path.join(config.db_path, "data", "collocations")
    sent_keys_s24 = np.load(os.path.join(colloc_dir, "sent_keys_s24.npy"), mmap_mode="r")
    sent_offsets = np.load(os.path.join(colloc_dir, "sent_offsets.npy"), mmap_mode="r")
    word_ids = {
        lemmas: np.load(os.path.join(colloc_dir, "attr_lemma_ids.npy" if lemmas else "token_ids.npy"), mmap_mode="r")
        for lemmas in {count_lemmas, *main_tids}
    }
    per_word_ids = word_ids[count_lemmas]

    def is_main(word):
        """Whether the word at an index of the collocation arrays is a word of the main query, by its form or its lemma"""
        return any(int(word_ids[lemmas][word]) in tids for lemmas, tids in main_tids.items())

    sent_idx, matched = _np_searchsorted_keys(sent_keys_s24, rows[:, :6])
    if not matched.any():
        return _empty_response(request, total_in_group=0)

    # The cooc engine writes positions and byte offsets in DOCUMENT ORDER (by word
    # position in sentence), not query order. So any of the row's words may be the
    # signature word, depending on where each appears in the sentence. Classify each
    # via token-id lookup at the word position.
    word_positions = rows[:, 7::2]
    byte_offsets = rows[:, 8::2]

    # Group by sentence philo_id.
    groups = {}
    for i in range(len(rows)):
        if not matched[i]:
            continue
        sent_start = int(sent_offsets[int(sent_idx[i])])
        main_bytes = []
        sig_tid = sig_byte_i = None
        for word_pos, byte in zip(word_positions[i].tolist(), byte_offsets[i].tolist()):
            if word_pos == 0:
                continue
            word = sent_start + word_pos - 1
            if is_main(word):
                main_bytes.append(byte)
            else:
                # The signature candidate; only score it if its tid carries a sig weight.
                sig_tid = int(per_word_ids[word])
                sig_byte_i = byte
        if not main_bytes or sig_tid not in sig_weight_by_tid:
            continue

        key = tuple(int(v) for v in rows[i, :6])
        g = groups.get(key)
        if g is None:
            g = {
                "rep_idx": i,
                "main_offsets": set(),
                "sig_offsets": set(),
                "sig_tids": set(),
            }
            groups[key] = g
        g["main_offsets"].update(main_bytes)
        g["sig_offsets"].add(sig_byte_i)
        g["sig_tids"].add(sig_tid)

    if not groups:
        return _empty_response(request, total_in_group=0)

    # Score per merged hit: sum of weights over distinct matched signature tids.
    merged = []
    for g in groups.values():
        score = sum(sig_weight_by_tid[t] for t in g["sig_tids"])
        merged.append({
            "rep_idx": g["rep_idx"],
            "main_offsets": sorted(g["main_offsets"]),
            "sig_offsets": sorted(g["sig_offsets"]),
            "sig_tids": g["sig_tids"],
            "score": score,
        })
    merged.sort(key=lambda m: -m["score"])

    total_in_group = len(merged)
    page = merged[offset: offset + top_n]
    has_more = offset + top_n < total_in_group

    if not page:
        return _empty_response(request, total_in_group=total_in_group)

    # Prefetch toms rows for only the page's hits. They're scattered across the
    # score-sorted list, so a contiguous prefetch_hits(min..max) would pull in
    # almost the entire hitlist; build philo_id strings for just these rows.
    rep_indices = [m["rep_idx"] for m in page]
    _prefetch_page_rows(db, rows, rep_indices)
    tid_to_display = _build_tid_to_display(config.db_path, sig_weight_by_tid.keys(), count_lemmas)
    word_regex = db.locals["token_regex"]

    passages = []
    for m in page:
        hit = hits[m["rep_idx"]]
        passage = _render_passage(
            db, config, hit, m["main_offsets"], m["sig_offsets"],
            m["sig_tids"], tid_to_display, word_regex, m["score"],
        )
        passages.append(passage)

    return {
        "group": {"field": rf, "value": rv},
        "signature": [
            {"word": sig_words[i],
             "z": float(sig_weights[i]) if i < len(sig_weights) else None}
            for i in range(len(sig_words))
        ],
        "passages": passages,
        "total_in_group": total_in_group,
        "n_returned": len(passages),
        "offset": offset,
        "has_more": has_more,
    }


# ---------------- Helpers ----------------

def _clamp_int(request, attr, default, low, high):
    raw = getattr(request, attr, None)
    try:
        n = int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        n = default
    return max(low, min(high, n))


def _split(s):
    return [t.strip() for t in s.split(",") if t.strip()]


def _split_floats(s):
    out = []
    for t in s.split(","):
        t = t.strip()
        if not t:
            continue
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out


def _np_searchsorted_keys(sent_keys_s24, hit_keys_native):
    """Lexicographic match of each 6-uint32 hit key against sorted sentence keys.

    `sent_keys_s24` is the precomputed S24 view of big-endian sentence keys
    (already lex-sorted). We convert the native-order hit keys to big-endian
    S24 so np.searchsorted's bytewise compare matches integer order.
    """
    n_sents = len(sent_keys_s24)
    if hit_keys_native.shape[0] == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
    hit_be = np.ascontiguousarray(hit_keys_native.astype(">u4"))
    hit_arr = hit_be.view("S24").ravel()
    sent_idx = np.searchsorted(sent_keys_s24, hit_arr)
    matched = (sent_idx < n_sents) & (sent_keys_s24[np.clip(sent_idx, 0, n_sents - 1)] == hit_arr)
    return sent_idx.astype(np.int64), matched


def _resolve_main_tids(db_path, hitlist_filename):
    """Read the cooc query's `.terms` file and resolve the main query's groups (every
    group but the last, the signature's: the expanded forms of each of its words) to
    vocab token ids: those of its lemmas in the lemma vocab, of its word forms in the
    word vocab (a key with properties, aimer:pos:VERB, as its word).
    Returns {True: lemma ids, False: word ids}, without an empty one.

    The .terms file is written by `expand_query_not` during db.query(); groups
    are separated by blank lines. We use it directly so case/ASCII/regex
    expansion stays in sync with whatever the cooc engine matched.
    """
    terms_path = f"{hitlist_filename}.terms"
    groups = get_word_groups(terms_path) if os.path.exists(terms_path) else []
    if len(groups) < 2:
        return {}
    names = property_names(os.path.join(db_path, "data"))
    words = {split_properties(key, names)[0] for group in groups[:-1] for key in group}
    main = {}
    for lemmas in (False, True):
        needed = {word for word in words if word.startswith("lemma:") == lemmas}
        tids = frozenset(_build_sig_tid_lookup(db_path, needed, lemmas).values()) if needed else frozenset()
        if tids:
            main[lemmas] = tids
    return main


def _vocab_files(db_path, count_lemmas):
    """Paths to the (hashes, offsets, strings) arrays for the active vocab."""
    colloc_dir = os.path.join(db_path, "data", "collocations")
    if count_lemmas:
        prefix = "attr_lemma_vocab"
    else:
        prefix = "vocab"
    return (
        os.path.join(colloc_dir, f"{prefix}_hashes.npy"),
        os.path.join(colloc_dir, f"{prefix}_offsets.npy"),
        os.path.join(colloc_dir, f"{prefix}_strings.bin"),
    )


def _hash_key(key):
    """Deterministic 64-bit key hash — must match the index-time formula
    (see collocation.py's filter-hash construction)."""
    return struct.unpack("<Q", hashlib.md5(key.encode("utf-8")).digest()[:8])[0]


def _build_sig_tid_lookup(db_path, sig_keys, count_lemmas):
    """Map signature key strings to vocab token ids.

    Uses the precomputed md5-hash array (vectorized ``np.isin``) instead of a
    linear scan + UTF-8 decode of the whole vocab — the latter is O(vocab) and
    costs ~2s on a 4.7M-form corpus. The actual string at each hash match is
    verified to rule out the rare 64-bit collision.
    """
    needed = set(sig_keys)
    if not needed:
        return {}
    hashes_path, offsets_path, blob_path = _vocab_files(db_path, count_lemmas)

    want = np.fromiter((_hash_key(k) for k in needed), dtype=np.uint64, count=len(needed))
    hashes = np.asarray(np.load(hashes_path, mmap_mode="r"))
    cand_tids = np.nonzero(np.isin(hashes, want))[0]
    if len(cand_tids) == 0:
        return {}

    offsets = np.load(offsets_path, mmap_mode="r")
    out = {}
    with open(blob_path, "rb") as f:
        for tid in cand_tids.tolist():
            s = int(offsets[tid])
            f.seek(s)
            name = f.read(int(offsets[tid + 1]) - s).decode("utf-8")
            if name in needed:
                out[name] = tid
    return out


def _build_tid_to_display(db_path, tids, count_lemmas):
    if not tids:
        return {}
    _, offsets_path, blob_path = _vocab_files(db_path, count_lemmas)
    offsets = np.load(offsets_path, mmap_mode="r")
    out = {}
    with open(blob_path, "rb") as f:
        for tid in tids:
            s = int(offsets[tid])
            f.seek(s)
            name = f.read(int(offsets[tid + 1]) - s).decode("utf-8")
            if name.startswith("lemma:"):
                name = name[6:]
            out[int(tid)] = name
    return out


def _prefetch_page_rows(db, rows, rep_indices):
    """Batch-fetch toms rows for just the page's representative hits.

    Mirrors DB.prefetch_hits' level selection, but builds philo_id strings from
    the specific (scattered) rows we render rather than the whole min..max span.
    """
    needed_levels = {1}  # always need doc level
    for f_type in db.locals["metadata_types"].values():
        if f_type in LEVEL_MAP:
            needed_levels.add(LEVEL_MAP[f_type])
        elif f_type == "div":
            needed_levels.update((2, 3, 4))
    unique_ids = set()
    for idx in rep_indices:
        philo_id = [int(v) for v in rows[idx, :6]]
        for level in needed_levels:
            unique_ids.add(hit_to_string(philo_id[:level], db.width))
    db.prefetch_rows(list(unique_ids))


def _render_passage(db, config, hit, main_offsets, sig_offsets,
                    sig_tids, tid_to_display, word_regex, score):
    """Render one passage: citation, context (with two-class highlights),
    matched-word list, score. `hit` is a real HitWrapper from the cooc HitList."""
    citation_hrefs = citation_links(db, config, hit)
    citation = citations(hit, citation_hrefs, config, report="concordance")
    context = _get_context_html(db, config, hit, main_offsets, sig_offsets, word_regex)

    matched_words = [tid_to_display.get(int(t), "") for t in sig_tids]
    matched_words = sorted({w for w in matched_words if w})

    return {
        "philo_id": list(hit.philo_id),
        "citation": citation,
        "citation_links": citation_hrefs,
        "score": round(float(score), 3),
        "matched": matched_words,
        "context": context,
    }


def _get_context_html(db, config, hit, main_offsets, sig_offsets, word_regex):
    """Read source bytes around the hit's matched offsets, emit HTML with
    main/sig highlighting."""
    all_offsets = sorted(set(int(b) for b in (list(main_offsets) + list(sig_offsets))))
    if not all_offsets:
        return ""
    context_size = config.concordance_length
    byte_distance = all_offsets[-1] - all_offsets[0]
    length = context_size + byte_distance + context_size
    adj_offsets, start_byte = adjust_bytes(all_offsets, context_size)

    doc_id = int(hit.philo_id[0])
    cursor = db.dbh.cursor()
    cursor.execute(
        "SELECT filename FROM toms WHERE philo_type='doc' AND philo_id=? LIMIT 1",
        (f"{doc_id} 0 0 0 0 0 0",),
    )
    row = cursor.fetchone()
    if not row:
        return ""
    path = os.path.join(config.db_path, "data", "TEXT", row["filename"])
    if not os.path.exists(path):
        return ""

    mm = _get_mmap(path)
    text = bytes(mm[start_byte:start_byte + length])

    main_set = set(int(b) for b in main_offsets)
    main_adj = [a for a, b in zip(adj_offsets, all_offsets) if b in main_set]
    sig_adj = [a for a, b in zip(adj_offsets, all_offsets) if b not in main_set]
    return format_concordance_distinctive(text, word_regex, main_adj, sig_adj)


def _empty_response(request, total_in_group=0):
    return {
        "group": {
            "field": getattr(request, "restrict_to_field", "") or "",
            "value": getattr(request, "restrict_to_value", "") or "",
        },
        "signature": [],
        "passages": [],
        "total_in_group": total_in_group,
        "n_returned": 0,
        "offset": _clamp_int(request, "offset", default=0, low=0, high=10000),
        "has_more": False,
    }
