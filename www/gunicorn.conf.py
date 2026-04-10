"""Gunicorn configuration for PhiloLogic5.

Usage:
    gunicorn --config gunicorn.conf.py app:application

Or via systemd (Linux) / launchd (macOS) service — see install.sh.
"""

import multiprocessing
import os

# Server socket
bind = "unix:/var/run/philologic/gunicorn.sock"

# Worker processes — sync workers (one request per process, no threads).
# This avoids LMDB "already open" errors that occur with threaded workers.
workers = min(multiprocessing.cpu_count(), 8)

# Timeout (seconds) — long to accommodate large corpus searches
timeout = 300

# Recycle workers periodically to bound memory growth
max_requests = 1000
max_requests_jitter = 50

# Preload application code before forking workers (saves memory via COW)
preload_app = True

# Process naming
proc_name = "philologic5"

# Logging — access logs handled by Apache/Nginx reverse proxy
accesslog = None
errorlog = "/var/log/philologic5/gunicorn-error.log"
loglevel = "info"
capture_output = True

# Numba JIT thread count — cap to avoid contention across workers
os.environ.setdefault("NUMBA_NUM_THREADS", "2")

# Numba cache directory — must be set BEFORE import numba (which happens
# during preload_app). Otherwise Numba writes to __pycache__ next to the
# source file, which the web server user cannot write to.
os.environ.setdefault("NUMBA_CACHE_DIR", "/var/lib/philologic5/numba_cache")

# Working directory
chdir = os.path.dirname(os.path.abspath(__file__))


def on_starting(server):
    """Called just before the master process is initialized."""
    if bind.startswith("unix:"):
        socket_dir = os.path.dirname(bind.split(":", 1)[1])
        if not os.path.exists(socket_dir):
            os.makedirs(socket_dir, mode=0o755)
    log_dir = os.path.dirname(errorlog)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, mode=0o755)
