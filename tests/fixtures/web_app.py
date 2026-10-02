"""The web app (www/), loaded in-process on a test's own database root, for falcon.testing."""

import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent


@contextmanager
def web_app_client(db_root):
    """A falcon.testing.TestClient of the web app serving the databases of db_root. The web app reads its database root
    when first imported, so its modules are imported again for each root."""
    import falcon.testing

    saved_env, saved_path = os.environ.get("PHILOLOGIC_DB_ROOT"), list(sys.path)
    os.environ["PHILOLOGIC_DB_ROOT"] = str(db_root)
    sys.path.insert(0, str(REPO_ROOT / "www"))
    try:
        for name in ("middleware", "resources.static", "resources.spa", "app"):
            if name in sys.modules:
                importlib.reload(sys.modules[name])
        app = importlib.import_module("app")
        yield falcon.testing.TestClient(app.create_app())
    finally:
        sys.path[:] = saved_path
        if saved_env is None:
            os.environ.pop("PHILOLOGIC_DB_ROOT", None)
        else:
            os.environ["PHILOLOGIC_DB_ROOT"] = saved_env
