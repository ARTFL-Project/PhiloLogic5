#!/var/lib/philologic5/philologic_env/bin/python3
"""Upgrade PhiloLogic5 databases after a code update.

Usage:
    python migrate.py /var/www/html/philologic5

Performs the following phases (ordered to minimize disruption):
  1. Migrate collocation data, build normalized_word_frequencies.lmdb, and clean stale KWIC caches
  2. Remove legacy CGI files (reports/, scripts/, dispatcher.py, webApp.py, .htaccess)
  3. Copy updated app/ source to each database (preserving each db's appConfig.json)
  4. Rebuild the frontend (npm run build) for each database

Prerequisites:
  Run install.sh first to update the Python library and central web app.
"""

import argparse
import os
import shutil
import subprocess
import sys

# ---------------------------------------------------------------------------
# Ensure PhiloLogic is importable (works when run from repo, not just venv)
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PYTHON_ROOT = os.path.join(REPO_ROOT, "python")
if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)

REPO_APP_DIR = os.path.join(REPO_ROOT, "app")
NPM = "/var/lib/philologic5/bin/npm"


# ---------------------------------------------------------------------------
# Database detection
# ---------------------------------------------------------------------------

def is_philologic_db(path):
    """Check if a directory is a PhiloLogic database."""
    data_dir = os.path.join(path, "data")
    return os.path.isdir(data_dir) and os.path.exists(os.path.join(data_dir, "toms.db"))


def _db_text_size(db_path):
    """Estimate database size from its TEXTS directory."""
    texts_dir = os.path.join(db_path, "data", "TEXTS")
    if not os.path.isdir(texts_dir):
        return 0
    return sum(e.stat().st_size for e in os.scandir(texts_dir) if e.is_file())


def find_databases(root_path):
    """Find all PhiloLogic databases under root_path."""
    databases = []
    for entry in sorted(os.scandir(root_path), key=lambda e: e.name):
        if entry.is_dir() and is_philologic_db(entry.path):
            databases.append(entry.path)
    return databases


# ---------------------------------------------------------------------------
# Phase 1: Collocation migration + KWIC cache cleanup
# ---------------------------------------------------------------------------

def ensure_norm_lmdb(db_path):
    """Build normalized_word_frequencies.lmdb if absent."""
    freq_file = os.path.join(db_path, "data", "frequencies", "normalized_word_frequencies")
    lmdb_path = freq_file + ".lmdb"
    if not os.path.exists(freq_file):
        print("  Skipping norm LMDB (no normalized_word_frequencies)")
        return
    if os.path.exists(lmdb_path):
        print("  Norm LMDB already exists — skipping")
        return
    print("  Building normalized_word_frequencies.lmdb...")
    from philologic.runtime.term_expansion import _build_norm_lmdb
    _build_norm_lmdb(freq_file, lmdb_path)
    size = sum(e.stat().st_size for e in os.scandir(lmdb_path) if e.is_file())
    print(f"  Norm LMDB built ({size / 1e6:.0f} MB)")


def ensure_forms_lmdb(db_path):
    """Build word_forms.lmdb from lemma/attr flat files if absent."""
    freq_dir = os.path.join(db_path, "data", "frequencies")
    flat_files = ["lemmas", "word_attributes", "lemma_word_attributes"]
    present = [f for f in flat_files if os.path.exists(os.path.join(freq_dir, f))]
    if not present:
        print("  Skipping word_forms LMDB (no lemma/attr flat files)")
        return False
    lmdb_path = os.path.join(freq_dir, "word_forms.lmdb")
    if os.path.exists(lmdb_path):
        print("  word_forms LMDB already exists — skipping")
        return False
    print(f"  Building word_forms.lmdb ({len(present)} flat file(s))...")
    from philologic.runtime.term_expansion import _build_forms_lmdb
    _build_forms_lmdb(os.path.join(db_path, "data"), lmdb_path)
    size = sum(e.stat().st_size for e in os.scandir(lmdb_path) if e.is_file())
    print(f"  word_forms LMDB built ({size / 1e6:.0f} MB)")
    return True


def ensure_metadata_word_index(db_path):
    """Build metadata_word_index.lmdb if absent."""
    data_path = os.path.join(db_path, "data")
    freq_dir = os.path.join(data_path, "frequencies")
    lmdb_path = os.path.join(freq_dir, "metadata_word_index.lmdb")
    if not os.path.isdir(freq_dir):
        print("  Skipping metadata word index (no frequencies/)")
        return
    if os.path.exists(lmdb_path):
        print("  Metadata word index already exists — skipping")
        return
    # Check for any normalized metadata freq files
    has_meta = any(
        f.startswith("normalized_") and f.endswith("_frequencies") and not f.endswith(".lmdb")
        and f != "normalized_word_frequencies"
        for f in os.listdir(freq_dir)
    )
    if not has_meta:
        print("  Skipping metadata word index (no normalized metadata freq files)")
        return
    print("  Building metadata_word_index.lmdb...")
    from philologic.runtime.term_expansion import build_metadata_word_index
    n_keys = build_metadata_word_index(data_path)
    size = sum(e.stat().st_size for e in os.scandir(lmdb_path) if e.is_file())
    print(f"  Metadata word index built ({n_keys} keys, {size / 1e6:.0f} MB)")


def migrate_collocation(db_path):
    """Migrate collocation data for a single database."""
    data_path = os.path.join(db_path, "data")
    words_dir = os.path.join(data_path, "words_and_philo_ids")
    colloc_dir = os.path.join(data_path, "collocations")

    # Clear all cached hitlists (stale formats, old search results)
    hitlists_dir = os.path.join(data_path, "hitlists")
    if os.path.isdir(hitlists_dir):
        removed = 0
        for entry in os.scandir(hitlists_dir):
            if entry.is_file():
                os.remove(entry.path)
                removed += 1
        if removed:
            print(f"  Cleared {removed} files from hitlists/")

    if not os.path.isdir(words_dir):
        print("  Skipping collocation migration (no words_and_philo_ids/)")
        return

    # Skip if collocation data already exists
    if os.path.isdir(colloc_dir) and any(os.scandir(colloc_dir)):
        print("  Collocation data already exists — skipping")
        return

    from philologic.runtime.DB import DB

    db = DB(data_path + "/")
    try:
        word_attrs = list(db.locals.word_attributes)
    except KeyError:
        word_attrs = []

    import lz4.frame
    import orjson

    print(f"  Counting words...")
    word_count = 0
    for entry in os.scandir(words_dir):
        if entry.name.endswith(".lz4"):
            with lz4.frame.open(entry.path) as f:
                for line in f:
                    if orjson.loads(line)["philo_type"] == "word":
                        word_count += 1
    print(f"  {word_count:,} words")

    class LoaderShim:
        def __init__(self):
            self.destination = data_path
            self.word_attributes = word_attrs
            self.ascii_conversion = db.locals.ascii_conversion
            self.word_count = word_count

    from philologic.loadtime.PostFilters import make_collocation_database

    make_collocation_database(LoaderShim(), colloc_dir)

    files = sorted(os.listdir(colloc_dir))
    total_size = sum(os.path.getsize(os.path.join(colloc_dir, f)) for f in files)
    print(f"  Collocation database: {len(files)} files, {total_size / 1e6:.0f} MB")


def _migrate_one(db_path):
    """Worker for parallel data migration. Returns (name, summary)."""
    import contextlib

    name = os.path.basename(db_path)
    data_path = os.path.join(db_path, "data")
    colloc_dir = os.path.join(data_path, "collocations")
    words_dir = os.path.join(data_path, "words_and_philo_ids")
    freq_file = os.path.join(data_path, "frequencies", "normalized_word_frequencies")
    lmdb_path = freq_file + ".lmdb"
    forms_lmdb_path = os.path.join(data_path, "frequencies", "word_forms.lmdb")
    meta_lmdb_path = os.path.join(data_path, "frequencies", "metadata_word_index.lmdb")

    # Check state before running so we can distinguish "built" from "skipped"
    has_words = os.path.isdir(words_dir)
    had_colloc = os.path.isdir(colloc_dir) and any(os.scandir(colloc_dir))
    had_norm_lmdb = os.path.exists(lmdb_path)
    had_forms_lmdb = os.path.exists(forms_lmdb_path)
    had_meta_lmdb = os.path.exists(meta_lmdb_path)

    # Redirect all output (including tqdm) to a log file
    log_path = os.path.join(data_path, "collocation_migration.log")
    with open(log_path, "w") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            migrate_collocation(db_path)
            ensure_norm_lmdb(db_path)
            ensure_forms_lmdb(db_path)
            ensure_metadata_word_index(db_path)

    # Build one-line summary
    parts = []
    if not has_words:
        parts.append("colloc: skipped (no words_and_philo_ids/)")
    elif had_colloc:
        parts.append("colloc: already exists")
    elif os.path.isdir(colloc_dir):
        files = os.listdir(colloc_dir)
        total_size = sum(os.path.getsize(os.path.join(colloc_dir, f)) for f in files)
        parts.append(f"colloc: done ({len(files)} files, {total_size / 1e6:.0f} MB)")
    else:
        parts.append("colloc: done")

    if had_norm_lmdb:
        parts.append("norm-lmdb: already exists")
    elif os.path.exists(lmdb_path):
        size = sum(e.stat().st_size for e in os.scandir(lmdb_path) if e.is_file())
        parts.append(f"norm-lmdb: built ({size / 1e6:.0f} MB)")
    else:
        parts.append("norm-lmdb: skipped (no freq file)")

    if had_forms_lmdb:
        parts.append("forms-lmdb: already exists")
    elif os.path.exists(forms_lmdb_path):
        size = sum(e.stat().st_size for e in os.scandir(forms_lmdb_path) if e.is_file())
        parts.append(f"forms-lmdb: built ({size / 1e6:.0f} MB)")
    else:
        parts.append("forms-lmdb: skipped (no flat files)")

    if had_meta_lmdb:
        parts.append("meta-lmdb: already exists")
    elif os.path.exists(meta_lmdb_path):
        size = sum(e.stat().st_size for e in os.scandir(meta_lmdb_path) if e.is_file())
        parts.append(f"meta-lmdb: built ({size / 1e6:.0f} MB)")
    else:
        parts.append("meta-lmdb: skipped (no metadata freq files)")

    return name, " | ".join(parts)


def _pipeline_one(db_path, skip_collocation=False):
    """Run the full migration pipeline for a single database. Returns (name, results_dict)."""
    import contextlib

    name = os.path.basename(db_path)
    results = {}

    # Phase 1: Data migration
    if not skip_collocation:
        _, data_summary = _migrate_one(db_path)
        results["data"] = data_summary
    else:
        results["data"] = "skipped (--skip-collocation)"

    # Phase 2: Remove legacy files
    try:
        log_path = os.path.join(db_path, "data", "legacy_cleanup.log")
        with open(log_path, "w") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                remove_legacy_files(db_path)
        results["legacy"] = "done"
    except Exception as e:
        results["legacy"] = f"FAILED: {e}"

    # Phase 3: Copy app source
    try:
        log_path = os.path.join(db_path, "data", "app_copy.log")
        with open(log_path, "w") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                copy_app(db_path)
        results["app_copy"] = "done"
    except Exception as e:
        results["app_copy"] = f"FAILED: {e}"

    # Phase 4: Build frontend
    _, build_summary = _build_one(db_path)
    results["frontend"] = build_summary

    return name, results


# ---------------------------------------------------------------------------
# Phase 2: Remove legacy CGI files
# ---------------------------------------------------------------------------

# These were per-database in CGI mode but are now served centrally by gunicorn.
# Databases with custom Vue apps that should not be overwritten by the default app.
_SKIP_APP_COPY = {"encyclopedie0226", "kafker", "bayle-0326", "montaigne1580", "montaigne1588", "montessaisvilley", "mvogeminitest", "tout-voltaire"}

_LEGACY_DIRS = ("reports", "scripts")
_LEGACY_FILES = ("dispatcher.py", "webApp.py", ".htaccess")


def remove_legacy_files(db_path):
    """Remove CGI-era files that are no longer needed under gunicorn."""
    removed = []
    for name in _LEGACY_DIRS:
        p = os.path.join(db_path, name)
        if os.path.isdir(p):
            shutil.rmtree(p)
            removed.append(f"{name}/")
    for name in _LEGACY_FILES:
        p = os.path.join(db_path, name)
        if os.path.isfile(p):
            os.remove(p)
            removed.append(name)
    # data/.htaccess ("deny from all") is also unnecessary with gunicorn
    data_htaccess = os.path.join(db_path, "data", ".htaccess")
    if os.path.isfile(data_htaccess):
        os.remove(data_htaccess)
        removed.append("data/.htaccess")
    if removed:
        print(f"  Removed: {', '.join(removed)}")
    else:
        print("  No legacy files found")


# ---------------------------------------------------------------------------
# Phase 3: Copy app/ source to each database
# ---------------------------------------------------------------------------

def copy_app(db_path):
    """Copy app/ source to database, preserving its appConfig.json."""
    name = os.path.basename(db_path)
    if name in _SKIP_APP_COPY:
        print(f"  Skipped (custom app)")
        return

    db_app_dir = os.path.join(db_path, "app")
    if not os.path.isdir(db_app_dir):
        print(f"  Warning: {db_app_dir} does not exist, skipping")
        return

    # Save the database's appConfig.json
    config_path = os.path.join(db_app_dir, "appConfig.json")
    config_backup = None
    if os.path.exists(config_path):
        with open(config_path) as f:
            config_backup = f.read()

    # Delete old app/ and copy fresh from repo
    shutil.rmtree(db_app_dir)

    def ignore(directory, contents):
        ignored = set()
        if "node_modules" in contents:
            ignored.add("node_modules")
        if "dist" in contents:
            ignored.add("dist")
        if os.path.relpath(directory, REPO_APP_DIR) == ".":
            ignored.add("appConfig.json")
        return ignored

    shutil.copytree(REPO_APP_DIR, db_app_dir, ignore=ignore)

    # Restore the database's appConfig.json
    if config_backup is not None:
        with open(config_path, "w") as f:
            f.write(config_backup)

    print(f"  App source updated (appConfig.json preserved)")


# ---------------------------------------------------------------------------
# Phase 4: Rebuild frontends
# ---------------------------------------------------------------------------

def build_frontend(db_path):
    """Run npm install + npm run build for a database."""
    name = os.path.basename(db_path)
    if name in _SKIP_APP_COPY:
        print(f"  Skipped (custom app)")
        return

    app_dir = os.path.join(db_path, "app")
    if not os.path.isdir(app_dir):
        print(f"  Warning: {app_dir} does not exist, skipping")
        return

    log_file = os.path.join(app_dir, "web_app_build.log")

    print(f"  npm install...")
    result = subprocess.run(
        [NPM, "install"], cwd=app_dir,
        capture_output=True, text=True,
    )
    with open(log_file, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write(result.stderr)
    if result.returncode != 0:
        print(f"  Error: npm install failed (see {log_file})")
        return

    print(f"  npm run build...")
    result = subprocess.run(
        [NPM, "run", "build"], cwd=app_dir,
        capture_output=True, text=True,
    )
    with open(log_file, "a") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write(result.stderr)
    if result.returncode != 0:
        print(f"  Error: npm run build failed (see {log_file})")
        return

    print(f"  Frontend built")


def _build_one(db_path):
    """Worker for parallel frontend builds. Returns (name, summary)."""
    name = os.path.basename(db_path)
    if name in _SKIP_APP_COPY:
        return name, "skipped (custom app)"
    app_dir = os.path.join(db_path, "app")
    if not os.path.isdir(app_dir):
        return name, "skipped (no app/)"

    log_file = os.path.join(app_dir, "web_app_build.log")

    result = subprocess.run(
        [NPM, "install"], cwd=app_dir,
        capture_output=True, text=True,
    )
    with open(log_file, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write(result.stderr)
    if result.returncode != 0:
        return name, f"FAILED npm install (see {log_file})"

    result = subprocess.run(
        [NPM, "run", "build"], cwd=app_dir,
        capture_output=True, text=True,
    )
    with open(log_file, "a") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write(result.stderr)
    if result.returncode != 0:
        return name, f"FAILED npm run build (see {log_file})"

    return name, "done"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Upgrade PhiloLogic5 databases after a code update.",
    )
    parser.add_argument(
        "db_root",
        help="Root directory containing PhiloLogic databases (e.g. /var/www/html/philologic5)",
    )
    parser.add_argument(
        "--skip-collocation", action="store_true",
        help="Skip collocation data migration (phase 1)",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=1,
        help="Number of databases to process in parallel (default: 1)",
    )
    args = parser.parse_args()

    # Validate paths
    if not os.path.isdir(REPO_APP_DIR):
        print(f"Error: cannot find app/ directory at {REPO_APP_DIR}")
        sys.exit(1)

    db_root = os.path.abspath(args.db_root)
    if not os.path.isdir(db_root):
        print(f"Error: {db_root} is not a directory")
        sys.exit(1)

    databases = find_databases(db_root)
    if not databases:
        print(f"No PhiloLogic databases found under {db_root}")
        sys.exit(1)

    print(f"Found {len(databases)} database(s) under {db_root}:")
    for db in databases:
        print(f"  {os.path.basename(db)}")

    failures = []  # list of (db_name, phase, error_message)

    if args.jobs > 1 and len(databases) > 1:
        # ---- Parallel mode: run the full pipeline per-database ----
        from concurrent.futures import ProcessPoolExecutor, as_completed

        n_jobs = min(args.jobs, len(databases))
        # Schedule largest databases first so small ones fill the tail
        by_size = sorted(databases, key=_db_text_size, reverse=True)
        print(f"\nRunning full pipeline for {n_jobs} databases in parallel (largest first)")
        print(f"{'=' * 60}")

        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {
                pool.submit(_pipeline_one, db, args.skip_collocation): db
                for db in by_size
            }
            for future in as_completed(futures):
                try:
                    name, results = future.result()
                    parts = [f"{k}: {v}" for k, v in results.items()]
                    print(f"  [{name}] {' | '.join(parts)}")
                    for phase, summary in results.items():
                        if isinstance(summary, str) and summary.startswith("FAILED"):
                            failures.append((name, phase, summary))
                except Exception as e:
                    db = futures[future]
                    name = os.path.basename(db)
                    print(f"  [{name}] Error: {e}")
                    failures.append((name, "pipeline", str(e)))
    else:
        # ---- Sequential mode: run phase-by-phase with visible output ----

        # Phase 1: Data migration (collocations + norm LMDB)
        if not args.skip_collocation:
            print(f"\n{'=' * 60}")
            print("Phase 1: Migrating data (collocations + norm LMDB)")
            print(f"{'=' * 60}")
            for db_path in databases:
                print(f"\n[{os.path.basename(db_path)}]")
                migrate_collocation(db_path)
                ensure_norm_lmdb(db_path)
                ensure_forms_lmdb(db_path)
                ensure_metadata_word_index(db_path)
        else:
            print("\nPhase 1: Skipped (--skip-collocation)")

        # Phase 2: Remove legacy CGI files
        print(f"\n{'=' * 60}")
        print("Phase 2: Removing legacy CGI files")
        print(f"{'=' * 60}")
        for db_path in databases:
            name = os.path.basename(db_path)
            print(f"\n[{name}]")
            try:
                remove_legacy_files(db_path)
            except Exception as e:
                print(f"  Error: {e}")
                failures.append((name, "legacy cleanup", str(e)))

        # Phase 3: Copy app source
        print(f"\n{'=' * 60}")
        print("Phase 3: Copying app source to databases")
        print(f"{'=' * 60}")
        for db_path in databases:
            name = os.path.basename(db_path)
            print(f"\n[{name}]")
            try:
                copy_app(db_path)
            except Exception as e:
                print(f"  Error: {e}")
                failures.append((name, "app copy", str(e)))

        # Phase 4: Rebuild frontends
        print(f"\n{'=' * 60}")
        print("Phase 4: Rebuilding frontends")
        print(f"{'=' * 60}")
        for db_path in databases:
            name = os.path.basename(db_path)
            print(f"\n[{name}]")
            try:
                build_frontend(db_path)
            except Exception as e:
                print(f"  Error: {e}")
                failures.append((name, "frontend build", str(e)))

    print(f"\n{'=' * 60}")
    if failures:
        print(f"Migration finished with {len(failures)} error(s):")
        for name, phase, err in failures:
            print(f"  [{name}] {phase}: {err}")
    else:
        print(f"Migration complete — {len(databases)} database(s) upgraded")
    print(f"{'=' * 60}")
    print("\nIf you haven't already, run install.sh to update the Python")
    print("library and central web app, then restart gunicorn:")
    print("  bash install.sh")
    print("  sudo systemctl restart philologic5-gunicorn")


if __name__ == "__main__":
    main()
