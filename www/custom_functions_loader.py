"""Dynamic loader for per-database custom_functions in central WSGI mode.

In CGI mode, each database has its own copy of the web files and
sys.path.append("..") loads custom_functions from the database directory.

In central WSGI mode, all databases share one set of web files, so we
use importlib to load each database's custom_functions directly by path,
bypassing sys.path entirely.

Usage in reports/scripts:
    from philologic.runtime import WebConfig, WSGIHandler
    from custom_functions_loader import get_custom

    def some_report(environ, start_response):
        db_path = environ.get("PHILOLOGIC_DBPATH", ...)
        _WebConfig = get_custom(db_path, "WebConfig", WebConfig)
        _WSGIHandler = get_custom(db_path, "WSGIHandler", WSGIHandler)
        config = _WebConfig(db_path)
        request = _WSGIHandler(environ, config)
        ...
"""

import importlib.util
import os
import threading

_cache = {}
_lock = threading.Lock()


def _load_module(db_path):
    """Load custom_functions module from a database directory.

    Returns the module or None if not found. Results are cached per db_path.
    """
    if db_path in _cache:
        return _cache[db_path]

    with _lock:
        # Double-check after acquiring lock
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


def get_custom(db_path, name, default):
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
