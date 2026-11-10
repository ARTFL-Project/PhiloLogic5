"""Write the corpus counts of databases loaded before the loader wrote them: for each word (and lemma), the number of
sentences it is in and of times it occurs, which the collocation filter and the word senses weigh words by: those
reports fail on a database without them. Databases that have them all are left alone, unless --force is given. Run it
as a user who can write the databases' data/collocations directories.

    python3 -m philologic.utils.write_corpus_counts [--force] /var/www/html/philologic5/mydb [...]
"""

import os
import sys
import time

import numpy as np

from philologic.loadtime.PostFilters import _COUNTED, write_corpus_counts


def has_counts(colloc_dir):
    """Whether the corpus counts of every vocabulary of the collocation arrays are there, each of the right length."""
    for ids_file, voff_file, kind in _COUNTED.values():
        if not os.path.exists(os.path.join(colloc_dir, ids_file)):
            continue
        n_vocab = len(np.load(os.path.join(colloc_dir, voff_file), mmap_mode="r")) - 1
        for name in (f"sentence_df_{kind}.npy", f"token_counts_{kind}.npy"):
            try:
                if len(np.load(os.path.join(colloc_dir, name), mmap_mode="r")) != n_vocab:
                    return False
            except (OSError, ValueError):
                return False
    return True


def main(databases, force=False):
    for database in databases:
        db_path = os.path.join(database, "data") if os.path.isdir(os.path.join(database, "data")) else database
        colloc_dir = os.path.join(db_path, "collocations")
        if not os.path.exists(os.path.join(colloc_dir, "sent_offsets.npy")):
            print(f"{database}: no collocation arrays (data/collocations/sent_offsets.npy)", file=sys.stderr)
            continue
        if not force and has_counts(colloc_dir):
            print(f"{database}: already has its corpus counts", flush=True)
            continue
        start = time.time()
        written = write_corpus_counts(colloc_dir)
        print(f"{database}: {', '.join(written)}, in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    force = "--force" in args
    databases = [a for a in args if a != "--force"]
    if not databases:
        sys.exit(__doc__)
    main(databases, force=force)
