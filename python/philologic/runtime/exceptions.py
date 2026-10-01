"""Runtime exceptions for PhiloLogic5."""


class BadRequest(Exception):
    """Raise from a report/script handler to return a 400 response."""


class NotFound(LookupError):
    """Raise from a report/script handler to return a 404 response: what the request names does not exist (or no
    longer, as caches)."""
