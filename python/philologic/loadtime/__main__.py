#!/var/lib/philologic5/philologic_env/bin/python3

import os
import sys
import time
from contextlib import contextmanager

from orjson import dumps

from philologic.utils import start_worker_server

# Start the server load workers are forked from now, so that its imports run while this process does its own
start_worker_server(["philologic.loadtime.Loader"])

from philologic.loadtime.Loader import Loader, setup_db_dir
from philologic.loadtime.LoadOptions import CONFIG_FILE, LoadOptions

os.environ["LC_ALL"] = "C"  # Exceedingly important to get uniform sort order.
os.environ["PYTHONIOENCODING"] = "utf-8"

# Where to record the start and end of each stage of the load, as JSON lines (used by philologic5-webui-loader)
PROGRESS_FILE = os.environ.get("PHILOLOGIC_PROGRESS_FILE")


@contextmanager
def stage(name):
    """Record the start and end of a stage of the load in PROGRESS_FILE, if set"""
    if PROGRESS_FILE:
        with open(PROGRESS_FILE, "ab") as progress_file:
            progress_file.write(dumps({"stage": name, "event": "start", "time": time.time()}) + b"\n")
    yield
    if PROGRESS_FILE:
        with open(PROGRESS_FILE, "ab") as progress_file:
            progress_file.write(dumps({"stage": name, "event": "end", "time": time.time()}) + b"\n")


if __name__ == "__main__":
    load_options = LoadOptions()
    load_options.parse(sys.argv)
    setup_db_dir(load_options["db_destination"], force_delete=load_options.force_delete)

    # Database load
    l = Loader.set_class_attributes(load_options.values)
    with stage("copy_files"):
        l.add_files(load_options.files)
    with stage("metadata"):
        if load_options.bibliography:
            load_metadata = l.parse_bibliography_file(load_options.bibliography, load_options.sort_order)
        else:
            load_metadata = l.parse_metadata(load_options.sort_order, header=load_options.header)
        l.set_file_data(load_metadata, l.textdir, l.workdir)
    with stage("parse"):
        l.parse_files(load_options.cores)
    with stage("merge"):
        l.merge_objects()
    with stage("count_words"):
        l.count_words()
    with stage("index"):
        l.build_inverted_index()
    with stage("sql"):
        l.setup_sql_load()
    with stage("post_filters"):
        l.post_processing()
    with stage("finish"):
        l.finish()
    if l.deleted_files:
        print(
            "The following files where not loaded due to invalid data in the header:\n{}".format(
                "\n".join(l.deleted_files)
            )
        )

    print(f"Application viewable at {os.path.join(CONFIG_FILE.url_root, load_options.dbname)}\n")
