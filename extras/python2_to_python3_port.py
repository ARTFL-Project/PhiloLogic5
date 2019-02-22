#!/usr/bin/env python3
"""This script should be used to convert a PhiloLogic 4.5 loaded database to
a PhiloLogic 4.5 database. Essentially makes it run under Python3.
This code assums all the code in the PhiloLogic database cirectory is unmodified.
All custom code should be handled separately"""

import sys
import os

PHILOLOGIC_INSTALL = "/var/lib/philologic4/web_app/"
TWO_TO_THREE_EXEC = "2to3-2.7"


def convert_config(database_to_convert, config_file):
    """Convert config files"""
    os.system(f"{TWO_TO_THREE_EXEC} --no-diffs -w {database_to_convert}/data/{config_file} > /dev/null 2>&1")
    os.system(f"black -q -l 120 {database_to_convert}/data/{config_file} > /dev/null 2>&1")


def main():
    """Main Loop"""
    database_to_convert = sys.argv[1]
    convert_config(database_to_convert, "web_config.cfg")
    convert_config(database_to_convert, "db.locals.py")
    # convert_config(database_to_convert, "load_config.py")
    os.system(f"cp -Rf {PHILOLOGIC_INSTALL}/* {database_to_convert}")
    print(database_to_convert, "converted...")


if __name__ == "__main__":
    main()
