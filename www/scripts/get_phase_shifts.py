"""Phase-shift detection for a query (collocation evolution).

Returns a phase decomposition of the query's collocate distribution over time,
plus the full ranked list of candidate boundaries so the front end can offer
a finer/coarser slider without re-running the detector.
"""

from philologic.runtime.DB import DB
from philologic.runtime.phase_shifts import detect_phases


def get_phase_shifts(request, config):
    db = DB(config.db_path + "/data/")

    q = (request.q or "").strip()
    if not q:
        return {"n_total_hits": 0, "phases": [], "candidates": []}
    count_lemmas = q.startswith("lemma:")

    attribute = None
    attribute_value = None
    if (request.colloc_filter_choice or "") == "attribute":
        attribute = (request.q_attribute or "").strip() or None
        attribute_value = (request.q_attribute_value or "").strip() or None
        if attribute is None or attribute_value is None:
            attribute = attribute_value = None

    metadata = dict(request.metadata or {})

    n_phases_raw = getattr(request, "n_phases", "")
    try:
        n_phases = int(n_phases_raw) if n_phases_raw not in (None, "") else None
    except (ValueError, TypeError):
        n_phases = None

    return detect_phases(
        db, config.db_path + "/data", q, count_lemmas, attribute, attribute_value,
        metadata, n_phases=n_phases,
    )
