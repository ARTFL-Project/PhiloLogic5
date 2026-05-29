"""Representative passages — passages within a group whose immediate context
contains the most of the group's distinctive-collocate signature.

Approach: a cooc query ("main_word AND (sig1 | sig2 | ... | sigK)") run with
the same metadata filters plus the drill-down restriction. PhiloLogic's query
engine handles lemma expansion, ASCII folding and case for free. Each hit row
carries the byte offsets and word positions of every matched word; we merge
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
from philologic.runtime.Query import resolve_method


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
    count_lemmas = main_q.startswith("lemma:")

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
    # AND-group and `|` as an inline OR within each group.
    sig_or = "|".join(sig_keys)
    cooc_q = f"{main_q} {sig_or}"
    raw_method = "proxy" if (request.colloc_within or "") == "n" else "sentence"
    raw_arg = request.method_arg if raw_method == "proxy" else ""
    # resolve_method handles method normalization (e.g. "sentence" -> "sentence_unordered")
    # the way WSGIHandler does it for web requests.
    method, method_arg = resolve_method(cooc_q, raw_method, raw_arg, "no")

    # Merge restrict_to_* into the metadata filters for this query.
    metadata = dict(request.metadata or {})
    metadata[rf] = rv

    hits = db.query(cooc_q, method, method_arg, **metadata)
    hits.finish()
    if len(hits) == 0:
        return _empty_response(request, total_in_group=0)

    # Build the set of token ids for the main word from the cooc query's
    # expansion (group 0 in the .terms file). This handles regex/case-variants
    # the same way the cooc engine does — ensures we can tell main from sig
    # at each hit position even when the main word is also in the signature
    # (e.g. main_q="lemma:citoyen" with "citoyen" present as a sig token).
    main_tids = _resolve_main_tids(config.db_path, hits.filename, count_lemmas)

    # Bulk-read the multi-word hitlist as a uint32 matrix.
    # Layout for a 2-word query (length = 7 + 2*2 = 11):
    #   [0..5]  sentence philo_id
    #   [6]     page
    #   [7]     word_pos of word 1 (main)
    #   [8]     byte_offset of word 1 (main)
    #   [9]     word_pos of word 2 (sig)
    #   [10]    byte_offset of word 2 (sig)
    with open(hits.filename, "rb") as f:
        raw = f.read()
    if not raw:
        return _empty_response(request, total_in_group=0)
    rows = np.frombuffer(raw, dtype=np.uint32).reshape(-1, hits.length)

    # Resolve sentence indices for each hit (for sig token-id lookup).
    colloc_dir = os.path.join(config.db_path, "data", "collocations")
    sent_keys_s24 = np.load(os.path.join(colloc_dir, "sent_keys_s24.npy"), mmap_mode="r")
    sent_offsets = np.load(os.path.join(colloc_dir, "sent_offsets.npy"), mmap_mode="r")
    if count_lemmas:
        per_word_ids = np.load(os.path.join(colloc_dir, "attr_lemma_ids.npy"), mmap_mode="r")
    else:
        per_word_ids = np.load(os.path.join(colloc_dir, "token_ids.npy"), mmap_mode="r")

    sent_idx, matched = _np_searchsorted_keys(sent_keys_s24, rows[:, :6])
    if not matched.any():
        return _empty_response(request, total_in_group=0)

    # The cooc engine writes byte offsets in DOCUMENT ORDER (by word position
    # in sentence), not query order. So col 8 may be either the main word's
    # byte or the signature word's byte depending on which appears first in
    # the sentence. Classify each via token-id lookup at the word position.
    word_pos_a = rows[:, 7]
    byte_a = rows[:, 8]
    word_pos_b = rows[:, 9] if hits.length >= 11 else np.zeros(len(rows), dtype=np.uint32)
    byte_b = rows[:, 10] if hits.length >= 11 else np.zeros(len(rows), dtype=np.uint32)

    # Group by sentence philo_id.
    groups = {}
    for i in range(len(rows)):
        if not matched[i]:
            continue
        sent_start = int(sent_offsets[int(sent_idx[i])])
        wpa = int(word_pos_a[i])
        wpb = int(word_pos_b[i])
        if wpa == 0 or wpb == 0:
            continue
        tid_a = int(per_word_ids[sent_start + wpa - 1])
        tid_b = int(per_word_ids[sent_start + wpb - 1])

        # Classify by which position is the main word (from cooc-query expansion).
        # The other position is the signature candidate; only score it if its tid
        # carries a sig weight.
        a_is_main = tid_a in main_tids
        b_is_main = tid_b in main_tids
        if a_is_main and not b_is_main:
            main_byte_i = int(byte_a[i])
            sig_tid = tid_b
            sig_byte_i = int(byte_b[i])
        elif b_is_main and not a_is_main:
            main_byte_i = int(byte_b[i])
            sig_tid = tid_a
            sig_byte_i = int(byte_a[i])
        else:
            # Both positions are the main word, or neither is — skip.
            continue
        if sig_tid not in sig_weight_by_tid:
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
        g["main_offsets"].add(main_byte_i)
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


def _resolve_main_tids(db_path, hitlist_filename, count_lemmas):
    """Read the cooc query's `.terms` file and resolve group 0 (the main word's
    expanded forms) to vocab token ids. Returns a frozenset of tids.

    The .terms file is written by `expand_query_not` during db.query(); groups
    are separated by blank lines. We use it directly so case/ASCII/regex
    expansion stays in sync with whatever the cooc engine matched.
    """
    terms_path = f"{hitlist_filename}.terms"
    main_forms = []
    if os.path.exists(terms_path):
        with open(terms_path, "r", encoding="utf-8") as f:
            current = []
            groups = []
            for line in f:
                line = line.strip()
                if line:
                    current.append(line)
                elif current:
                    groups.append(current)
                    current = []
            if current:
                groups.append(current)
            if groups:
                main_forms = groups[0]
    if not main_forms:
        return frozenset()
    needed = set(main_forms)
    lookup = _build_sig_tid_lookup(db_path, needed, count_lemmas)
    return frozenset(lookup.values())


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
