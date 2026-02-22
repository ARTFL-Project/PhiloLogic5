"""WSGI endpoint helpers for PhiloLogic5.

Provides:
- resolve(): per-database override resolution (custom_functions)
- json_endpoint(): decorator that eliminates WSGI boilerplate from standard endpoints

Usage:
    from wsgi_helpers import json_endpoint, resolve

    @json_endpoint
    def concordance(request, config):
        _concordance_results = resolve(config.db_path, "concordance_results", concordance_results)
        return _concordance_results(request, config)
"""

import functools
import importlib.util
import os

import orjson
from philologic.runtime import WebConfig, WSGIHandler

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


_JSON_HEADERS = [
    ("Content-type", "application/json; charset=UTF-8"),
    ("Access-Control-Allow-Origin", "*"),
]

_ERROR_HEADERS = [
    ("Content-type", "text/plain; charset=UTF-8"),
    ("Access-Control-Allow-Origin", "*"),
]


def json_endpoint(fn):
    """Wrap a (request, config) -> object function into a WSGI endpoint.

    The wrapped function receives:
        request: a (possibly custom) WSGIHandler instance
        config:  a (possibly custom) WebConfig instance

    It must return a JSON-serializable Python object.
    The decorator handles db_path extraction, per-database override resolution,
    config/request construction, JSON serialization, and response headers.

    Raise BadRequest(message) to return a 400 response instead.
    """

    @functools.wraps(fn)
    def wsgi_wrapper(environ, start_response):
        db_path = environ["PHILOLOGIC_DBPATH"]
        config = resolve(db_path, "WebConfig", WebConfig)(db_path)
        request = resolve(db_path, "WSGIHandler", WSGIHandler)(environ, config)
        try:
            result = fn(request, config)
        except BadRequest as e:
            start_response("400 Bad Request", _ERROR_HEADERS)
            yield str(e).encode("utf-8")
            return
        start_response("200 OK", _JSON_HEADERS)
        yield orjson.dumps(result)

    return wsgi_wrapper


_HTML_HEADERS = [
    ("Content-type", "text/html; charset=UTF-8"),
    ("Access-Control-Allow-Origin", "*"),
]


def html_endpoint(fn):
    """Wrap a (request, config) -> str function into a WSGI endpoint returning HTML."""

    @functools.wraps(fn)
    def wsgi_wrapper(environ, start_response):
        db_path = environ["PHILOLOGIC_DBPATH"]
        config = resolve(db_path, "WebConfig", WebConfig)(db_path)
        request = resolve(db_path, "WSGIHandler", WSGIHandler)(environ, config)
        result = fn(request, config)
        start_response("200 OK", _HTML_HEADERS)
        yield result.encode("utf-8")

    return wsgi_wrapper
