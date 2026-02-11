#!/var/lib/philologic5/philologic_env/bin/python3
"""Standalone search worker for WSGI deployment.

This script is spawned as a subprocess by Query.query() to perform the actual
search in a detached process. This replaces the double-fork pattern used in
CGI mode, allowing PhiloLogic to run under WSGI servers like Gunicorn.

Usage:
    _search_worker.py <db_path> <filename> <method> <method_arg> <overflow_words_json> <object_level> [corpus_file]

Arguments:
    db_path: Path to the PhiloLogic database (e.g., /var/lib/philologic5/databases/mydb/data)
    filename: Path to the hitlist file to write results to
    method: Search method (single_term, phrase_ordered, phrase_unordered,
            proxy_ordered, proxy_unordered, exact_cooc_ordered, exact_cooc_unordered,
            sentence_ordered, sentence_unordered)
    method_arg: Method-specific argument (e.g., distance for proximity search)
    overflow_words_json: JSON-encoded list of overflow words
    object_level: Text object level for sentence searches (sent, para, etc.)
    corpus_file: Optional path to corpus filter file
"""

import json
import os
import sys

# Fork to detach from parent immediately.
# In CGI mode, Apache waits for direct child processes before sending the response.
# This fork makes the subprocess.Popen child exit right away, while the grandchild
# (orphaned, reparented to init) runs the search in the background.
# In WSGI/Gunicorn mode this is harmless - Gunicorn never forks, only calls Popen.
_pid = os.fork()
if _pid > 0:
    os._exit(0)

# Grandchild: fully detach
os.setsid()
sys.stdin = open(os.devnull, 'r')
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

# Set umask before importing numba to ensure cache files are world-writable
os.umask(0o000)

# Now import the heavy modules
from philologic.runtime.Query import (
    search_word,
    search_phrase,
    search_within_word_span,
    search_within_text_object,
)


def main():
    """Execute the search based on command line arguments."""
    if len(sys.argv) < 7:
        # Can't print error since stdout is redirected, just exit
        sys.exit(1)

    db_path = sys.argv[1]
    filename = sys.argv[2]
    method = sys.argv[3]
    method_arg = int(sys.argv[4]) if sys.argv[4] and sys.argv[4] != "0" else 0
    overflow_words = set(json.loads(sys.argv[5]))
    object_level = sys.argv[6]  # For sentence searches (sent, para, etc.)
    corpus_file = sys.argv[7] if len(sys.argv) > 7 and sys.argv[7] else None

    # Execute the appropriate search method
    if method == "single_term":
        search_word(db_path, filename, overflow_words, corpus_file=corpus_file)
    elif method == "phrase_ordered":
        search_phrase(db_path, filename, overflow_words, corpus_file=corpus_file)
    elif method == "phrase_unordered":
        search_within_word_span(db_path, filename, overflow_words, method_arg or 1, False, False, corpus_file=corpus_file)
    elif method == "proxy_ordered":
        search_within_word_span(db_path, filename, overflow_words, method_arg or 1, True, False, corpus_file=corpus_file)
    elif method == "proxy_unordered":
        search_within_word_span(db_path, filename, overflow_words, method_arg or 1, False, False, corpus_file=corpus_file)
    elif method == "exact_cooc_ordered":
        search_within_word_span(db_path, filename, overflow_words, method_arg or 1, True, True, corpus_file=corpus_file)
    elif method == "exact_cooc_unordered":
        search_within_word_span(db_path, filename, overflow_words, method_arg or 1, False, True, corpus_file=corpus_file)
    elif method == "sentence_ordered":
        search_within_text_object(db_path, filename, overflow_words, object_level, True, corpus_file=corpus_file)
    elif method == "sentence_unordered":
        search_within_text_object(db_path, filename, overflow_words, object_level, False, corpus_file=corpus_file)
    else:
        # Unknown method - exit with error
        sys.exit(1)

    # Mark search as complete
    with open(filename + ".done", "w") as flag:
        flag.write(" ".join(sys.argv) + "\n")
        flag.flush()


if __name__ == "__main__":
    main()
