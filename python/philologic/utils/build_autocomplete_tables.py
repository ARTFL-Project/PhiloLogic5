"""Build the autocomplete tables of databases loaded before them, from their frequency files: their autocomplete then
suggests the most frequent words and lemmas first, rather than the first in alphabetical order.

    python3 -m philologic.utils.build_autocomplete_tables /var/www/html/philologic5/mydb [...]
"""

import os
import sys
import time

from philologic.runtime.term_expansion import build_autocomplete_tables


def main(databases):
    for database in databases:
        db_path = os.path.join(database, "data") if os.path.isdir(os.path.join(database, "data")) else database
        if not os.path.isdir(os.path.join(db_path, "frequencies")):
            print(f"{database}: no PhiloLogic database (no frequencies directory)", file=sys.stderr)
            continue
        start = time.time()
        n_prefixes = build_autocomplete_tables(db_path)
        print(f"{database}: {n_prefixes} prefixes, in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
