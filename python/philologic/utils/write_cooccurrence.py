"""Write the co-occurrence tables of databases loaded before the loader wrote them: for each word (and lemma), the
words it is in a sentence with and in how many sentences, which the word senses take the corpus's profiles of the
query's collocates from: that report fails on a database without them. Databases that have them are left alone, unless
--force is given. A table takes 8 bytes on disk for each pair of words seen in a sentence together, and while it is
written about 8 bytes of memory for each word of each sentence (each once). The corpus counts it is made from
(philologic.utils.write_corpus_counts) are written first if missing. Run it as a user who can write the databases'
data/collocations directories; --cores, the threads it counts in (4 by default, as philoload5).

    python3 -m philologic.utils.write_cooccurrence [--force] [--cores N] /var/www/html/philologic5/mydb [...]
"""

import os
import sys
import time

import numpy as np

from philologic.loadtime.PostFilters import _COUNTED, cooccurrence_paths, write_corpus_counts, write_cooccurrence
from philologic.utils.write_corpus_counts import has_counts


def has_table(colloc_dir):
    """Whether the co-occurrence table of every vocabulary of the collocation arrays is there, all of it."""
    for ids_file, voff_file, kind in _COUNTED.values():
        if not os.path.exists(os.path.join(colloc_dir, ids_file)):
            continue
        n_vocab = len(np.load(os.path.join(colloc_dir, voff_file), mmap_mode="r")) - 1
        paths = cooccurrence_paths(colloc_dir, kind)
        try:
            indptr = np.load(paths["indptr"], mmap_mode="r")
            lengths = {len(np.load(paths[part], mmap_mode="r")) for part in ("partners", "counts")}
        except (OSError, ValueError):
            return False
        if len(indptr) != n_vocab + 1 or lengths != {int(indptr[-1])}:
            return False
    return True


def main(databases, force=False, cores=4):
    for database in databases:
        db_path = os.path.join(database, "data") if os.path.isdir(os.path.join(database, "data")) else database
        colloc_dir = os.path.join(db_path, "collocations")
        if not os.path.exists(os.path.join(colloc_dir, "sent_offsets.npy")):
            print(f"{database}: no collocation arrays (data/collocations/sent_offsets.npy)", file=sys.stderr)
            continue
        if not force and has_table(colloc_dir):
            print(f"{database}: already has its co-occurrence tables", flush=True)
            continue
        start = time.time()
        if not has_counts(colloc_dir):
            print(f"{database}: {', '.join(write_corpus_counts(colloc_dir))} written first", flush=True)
        written = write_cooccurrence(colloc_dir, cores)
        size = sum(os.path.getsize(os.path.join(colloc_dir, name)) for name in written)
        print(f"{database}: {', '.join(written)} ({size / 2**30:.2f} GiB), in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    force = "--force" in args
    cores = 4
    if "--cores" in args:
        at = args.index("--cores")
        try:
            cores = int(args[at + 1])
        except (IndexError, ValueError):
            sys.exit(__doc__)
        del args[at : at + 2]
    databases = [a for a in args if a != "--force"]
    if not databases or cores < 1:
        sys.exit(__doc__)
    main(databases, force=force, cores=cores)
