import os
import sys


def update_shebang(db_dir):
    """Update the python executable in the shebang line of all scripts in the Web app directory."""
    new_shebang = f"#!{sys.executable}\n"

    def rewrite_file(script):
        with open(script, "r") as input_file:
            file = input_file.readlines()
        with open(script, "w") as output_file:
            for line in file:
                if line.startswith("#!/usr/bin/env python3"):
                    output_file.write(new_shebang)
                else:
                    output_file.write(line)

    for script in os.scandir(os.path.join(db_dir, "reports")):
        if script.is_file() and script.name.endswith(".py"):
            rewrite_file(script.path)

    for script in os.scandir(os.path.join(db_dir, "scripts")):
        if script.is_file() and script.name.endswith(".py"):
            rewrite_file(script.path)

    rewrite_file(os.path.join(db_dir, "webApp.py"))
    rewrite_file(os.path.join(db_dir, "dispatcher.py"))
