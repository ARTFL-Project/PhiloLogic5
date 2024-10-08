#!/var/lib/philologic5/philologic_env/bin/python3

import os
import sys

from philologic.loadtime.Loader import Loader, setup_db_dir
from philologic.loadtime.LoadOptions import CONFIG_FILE, LoadOptions

os.environ["LC_ALL"] = "C"  # Exceedingly important to get uniform sort order.
os.environ["PYTHONIOENCODING"] = "utf-8"


if __name__ == "__main__":
    load_options = LoadOptions()
    load_options.parse(sys.argv)
    setup_db_dir(load_options["db_destination"], force_delete=load_options.force_delete)

    # Database load
    l = Loader.set_class_attributes(load_options.values)
    l.add_files(load_options.files)
    if load_options.bibliography:
        load_metadata = l.parse_bibliography_file(load_options.bibliography, load_options.sort_order)
    else:
        load_metadata = l.parse_metadata(load_options.sort_order, header=load_options.header)
    l.set_file_data(load_metadata, l.textdir, l.workdir)
    l.parse_files(load_options.cores)
    l.merge_objects()
    l.count_words()
    l.build_inverted_index()
    l.setup_sql_load()
    l.post_processing()
    l.finish()
    if l.deleted_files:
        print(
            "The following files where not loaded due to invalid data in the header:\n{}".format(
                "\n".join(l.deleted_files)
            )
        )

    print(f"Application viewable at {os.path.join(CONFIG_FILE.url_root, load_options.dbname)}\n")
