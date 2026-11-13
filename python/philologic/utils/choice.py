#!/var/lib/philologic5/philologic_env/bin/python3
"""The readings of a TEI <choice>, its direct children orig, reg, sic, corr, abbr and expan: the parser indexes one of
them, the formatters show one, and neither has the others. The original spelling is shown, say, and the regularized
one indexed at its bytes. Elements of these names outside a <choice> are text like any other."""

READINGS = frozenset(("orig", "reg", "sic", "corr", "abbr", "expan"))
SHOWN = ("corr", "orig", "abbr", "sic")  # the first of these shown, else the first reading
INDEXED = ("reg", "expan", "corr")  # the first of these indexed, else the one shown


def choose_readings(names):
    """The index of the reading shown and that of the reading indexed, among the readings of a choice, named names in
    their order in it: of orig and reg, the orig is shown and the reg indexed; of sic and corr, the corr is both."""
    names = list(names)
    shown = next((names.index(name) for name in SHOWN if name in names), 0)
    indexed = next((names.index(name) for name in INDEXED if name in names), shown)
    return shown, indexed


def choice_readings_of(db_locals):
    """Whether the database of db_locals was loaded by a parser indexing the readings of choices as above, and so has
    its texts shown with the readings choose_readings picks: databases loaded before have every reading indexed, and
    shown."""
    try:
        return bool(db_locals["choice_readings"])
    except KeyError:
        return False
