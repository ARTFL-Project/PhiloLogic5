"""Helpers for PhiloLogic5 web layer.

Provides:
- resolve(): per-database override resolution (custom_functions)
- BadRequest: exception for 400 responses
"""

import importlib.util
import os
import sys

_cache = {}


def _load_module(db_path):
    """Load custom_functions module from a database directory.

    Returns the module or None if not found. Results are cached per db_path.
    """
    if db_path in _cache:
        return _cache[db_path]

    custom_dir = os.path.join(db_path, "custom_functions")
    custom_path = os.path.join(custom_dir, "__init__.py")
    if os.path.exists(custom_path):
        module_name = f"custom_functions_{db_path.replace('/', '_')}"
        spec = importlib.util.spec_from_file_location(
            module_name, custom_path,
            submodule_search_locations=[custom_dir],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        _cache[db_path] = module
    else:
        _cache[db_path] = None

    return _cache[db_path]


def resolve(db_path, name, default):
    """Get a named attribute from the database's custom_functions, or return default.

    Args:
        db_path: Path to the database's web directory
        name: Name of the attribute to look up (e.g., "WebConfig", "concordance_results")
        default: Default value if custom_functions doesn't exist or doesn't define this name
    """
    module = _load_module(db_path)
    if module is not None:
        return getattr(module, name, default)
    return default


# Re-export from philologic package so www/ code keeps using the same import path
from philologic.runtime.exceptions import BadRequest  # noqa: F401
