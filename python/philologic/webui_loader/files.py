"""Files which can be loaded: directory listings for the file browser of the load pages, and the files of a load,
from a directory (and a pattern), a list of paths, or a file listing them (as philoload5 -F). As a service, only
files within the allowed roots (and the user's upload area) can be seen or loaded."""

import fnmatch
import os
from collections import Counter

MAX_LISTED = 2000  # entries of a directory listing
MAX_FILES = 1000000  # files of a load


class FilesError(Exception):
    """Files which can't be listed or loaded"""


def within(path, roots):
    """Whether a path (resolved, symbolic links followed) is inside one of the roots (all paths when there are none)"""
    if not roots:
        return True
    real = os.path.realpath(path)
    return any(real == root or real.startswith(root.rstrip("/") + "/") for root in roots)


def check_name(path):
    """File names with control characters (line breaks...) or spaces around them are refused: the files of a load are
    listed one per line for philoload5, which strips the lines"""
    if (
        not isinstance(path, str)
        or path != path.strip()
        or any(ord(char) < 32 or 127 <= ord(char) < 160 for char in path)
    ):
        raise FilesError(f"file names with control characters or spaces around them can't be loaded: {path!r}")


def check_allowed(path, roots):
    check_name(path)
    if not os.path.isabs(path):
        raise FilesError(f"{path} is not an absolute path")
    if not within(path, roots):
        raise FilesError(f"{path} is outside the directories whose files can be loaded")


def loaded_name(path):
    """Name of a file in the database (Loader.add_files)"""
    return os.path.basename(path).replace(" ", "_").replace("'", "_")


def list_directory(path, roots, pattern="*"):
    """Entries of a directory for the file browser: its subdirectories and files (the first MAX_LISTED), with the
    number of files matching the pattern and their total size"""
    check_allowed(path, roots)
    if not os.path.isdir(path):
        raise FilesError(f"{path} is not a directory")
    directories, files = [], []
    matching, matching_size = 0, 0
    try:
        entries = list(os.scandir(path))
    except PermissionError as error:
        raise FilesError(f"{path} can't be read") from error
    for entry in entries:
        if entry.name.startswith("."):
            continue
        try:
            if entry.is_dir():
                if within(entry.path, roots):
                    directories.append(entry.name)
            elif entry.is_file():
                size = entry.stat().st_size
                files.append((entry.name, size))
                if fnmatch.fnmatch(entry.name, pattern):
                    matching += 1
                    matching_size += size
        except OSError:
            continue
    directories.sort()
    files.sort()
    parent = os.path.dirname(path.rstrip("/")) or "/"
    return {
        "path": path,
        "parent": parent if parent != path and within(parent, roots) else None,
        "directories": directories[:MAX_LISTED],
        "files": [{"name": name, "size": size} for name, size in files[:MAX_LISTED]],
        "truncated": len(directories) > MAX_LISTED or len(files) > MAX_LISTED,
        "file_count": len(files),
        "matching": matching,
        "matching_size": matching_size,
    }


def resolve(spec, roots):
    """Paths of the files of a load, sorted, from its spec: {"directory": path, "pattern": "*.xml", "recursive":
    false}, {"paths": [...]} or {"file_list": path} (a file with one path per line)"""
    if "directory" in spec:
        directory = spec["directory"]
        check_allowed(directory, roots)
        if not os.path.isdir(directory):
            raise FilesError(f"{directory} is not a directory")
        pattern = spec.get("pattern") or "*"
        paths = []
        if spec.get("recursive"):
            for dirpath, dirnames, filenames in os.walk(directory):
                dirnames[:] = sorted(
                    d for d in dirnames if not d.startswith(".") and within(os.path.join(dirpath, d), roots)
                )
                paths.extend(os.path.join(dirpath, name) for name in filenames if fnmatch.fnmatch(name, pattern))
                if len(paths) > MAX_FILES:
                    break
        else:
            paths = [
                entry.path
                for entry in os.scandir(directory)
                if entry.is_file() and not entry.name.startswith(".") and fnmatch.fnmatch(entry.name, pattern)
            ]
    elif "paths" in spec:
        paths = list(spec["paths"])
    elif "file_list" in spec:
        check_allowed(spec["file_list"], roots)
        try:
            with open(spec["file_list"], encoding="utf8") as file_list:
                paths = [line.strip() for line in file_list if line.strip()]
        except OSError as error:
            raise FilesError(f"{spec['file_list']} can't be read") from error
    else:
        raise FilesError("no files given")
    if len(paths) > MAX_FILES:
        raise FilesError(f"more than {MAX_FILES} files")
    for path in paths:
        check_allowed(path, roots)
    return sorted(set(paths))


def summary(paths):
    """Count, total size, extensions and missing files of the files of a load, and the names which the loader would
    give to several of them (it copies them under their name, so all but one would be lost)"""
    size = 0
    missing = []
    extensions = Counter()
    names = Counter()
    for path in paths:
        try:
            info = os.stat(path)
        except OSError:
            missing.append(path)
            continue
        if not os.path.isfile(path) or not os.access(path, os.R_OK):
            missing.append(path)
            continue
        size += info.st_size
        extensions[os.path.splitext(path)[1].lower() or "(none)"] += 1
        names[loaded_name(path)] += 1
    return {
        "count": len(paths),
        "size": size,
        "extensions": dict(extensions.most_common()),
        "missing": missing[:100],
        "missing_count": len(missing),
        "duplicate_names": sorted(name for name, count in names.items() if count > 1)[:100],
        "sample": [os.path.basename(path) for path in paths[:10]],
    }
