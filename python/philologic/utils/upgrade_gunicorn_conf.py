"""Upgrade gunicorn.conf.py while preserving user customizations.

Compares the old installed gunicorn.conf.py against the old shipped defaults
(gunicorn.conf.defaults.py) to detect user customizations, then replaces
the corresponding lines in the new version in place.

Settings that the user never changed get the new defaults automatically.
Settings the user explicitly changed are preserved at their original location.
"""

import ast
import re


# Settings that can be safely merged across upgrades.
# Hooks, imports, and computed values are always taken from the new version.
MERGEABLE_SETTINGS = {
    "bind",
    "workers",
    "timeout",
    "max_requests",
    "max_requests_jitter",
    "preload_app",
    "proc_name",
    "accesslog",
    "errorlog",
    "loglevel",
    "capture_output",
}


def _load_conf_values(path):
    """Extract simple top-level assignments from a Python config file using AST.

    Only extracts assignments of literal values (strings, numbers, booleans, None)
    for settings in MERGEABLE_SETTINGS.  Ignores function calls, imports, and
    computed values — no code is executed.
    """
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)

    values = {}
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name not in MERGEABLE_SETTINGS:
            continue
        try:
            values[name] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            # Not a literal (e.g. min(cpu_count(), 4)) — skip, can't merge
            pass
    return values


def _load_conf_names(path):
    """Extract all top-level assignment names from a config file.

    Unlike _load_conf_values, this returns names even for non-literal values
    (e.g. min(cpu_count(), 4)), so we can detect which settings exist in the file.
    """
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    names = set()
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        names.add(node.targets[0].id)
    return names


def _replace_setting_in_file(filepath, name, value):
    """Replace a setting's value in-place in a Python config file.

    Finds the line matching `name = ...` and replaces it with the new value.
    """
    with open(filepath) as f:
        content = f.read()

    # Match the assignment line: `name = <anything>` (not inside a comment)
    pattern = re.compile(rf'^({re.escape(name)}\s*=\s*).*$', re.MULTILINE)
    replacement = rf'\g<1>{value!r}'
    new_content, count = pattern.subn(replacement, content)

    if count > 0:
        with open(filepath, "w") as f:
            f.write(new_content)
        return True
    return False


def upgrade_gunicorn_conf(old_conf, old_defaults, new_conf, new_defaults=None):
    """Upgrade gunicorn.conf.py preserving user customizations.

    The new conf and defaults files should already be in their final location
    (e.g. copied by install.sh). This function reads the OLD backups to detect
    customizations, then replaces the corresponding values in the new conf.

    Args:
        old_conf:      path to backup of the previously installed gunicorn.conf.py
        old_defaults:  path to backup of the previously installed gunicorn.conf.defaults.py
        new_conf:      path to the new gunicorn.conf.py (already in place)
        new_defaults:  path to the new gunicorn.conf.defaults.py (already in place)

    Returns:
        List of setting names that were preserved from the old config.
    """
    # Load values from the old files
    prev_defaults = _load_conf_values(old_defaults)
    prev_conf = _load_conf_values(old_conf)

    # Find settings the user explicitly changed from the original defaults
    user_customizations = {}
    for key in MERGEABLE_SETTINGS:
        if key in prev_conf and key in prev_defaults:
            if prev_conf[key] != prev_defaults[key]:
                user_customizations[key] = prev_conf[key]
        elif key in prev_conf and key not in prev_defaults:
            # User added a setting that wasn't in the defaults (e.g. user/group)
            user_customizations[key] = prev_conf[key]

    # Only preserve customizations for settings that exist in the new defaults.
    # Settings removed from the new defaults (e.g. worker_class, threads) are
    # intentionally dropped — even if the user had customized them.
    new_default_names = _load_conf_names(new_defaults) if new_defaults else set()
    preserved = []
    for key, value in user_customizations.items():
        if new_default_names and key not in new_default_names:
            continue
        if _replace_setting_in_file(new_conf, key, value):
            preserved.append(key)

    return preserved
