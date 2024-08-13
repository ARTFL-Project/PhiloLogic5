#!/var/lib/philologic5/philologic_env/bin/python3

"""Count number of lines in a file using subprocess module."""

import subprocess


def count_lines(file_path, lz4=False):
    """Count number of lines in a file."""
    if lz4:
        cmd = f"lz4 -dc {file_path} | wc -l"
    else:
        cmd = f"wc -l {file_path} | cut -d ' ' -f 1"
    process = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    count = int(process.stdout.strip())
    return count
