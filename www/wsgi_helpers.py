"""Helpers for PhiloLogic5 web layer.

Provides:
- resolve(): per-database override resolution (custom_functions)
- BadRequest: exception for 400 responses
"""

import importlib.util
import os

_cache = {}


def _load_module(db_path):
    """Load custom_functions module from a database directory.

    Returns the module or None if not found. Results are cached per db_path.
    """
    if db_path in _cache:
        return _cache[db_path]

    custom_path = os.path.join(db_path, "custom_functions", "__init__.py")
    if os.path.exists(custom_path):
        module_name = f"custom_functions_{db_path.replace('/', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, custom_path)
        module = importlib.util.module_from_spec(spec)
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


class BadRequest(Exception):
    """Raise from an endpoint to return a 400 response with a plain-text message."""
