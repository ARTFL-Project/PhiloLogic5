"""Gunicorn configuration for PhiloLogic5.

This configuration is used when running PhiloLogic under Gunicorn instead of
Apache CGI. It provides better performance and resource efficiency.

Usage:
    gunicorn --config gunicorn.conf.py app:application

Or via systemd (Linux) / launchd (macOS) service — see install.sh.
"""

import multiprocessing
import os

# Server socket
bind = "unix:/var/run/philologic/gunicorn.sock"

# Worker processes
workers = min(multiprocessing.cpu_count(), 4)

# Worker class
# gthread: each worker handles multiple requests via threads.
# Do NOT use 'gevent' or 'eventlet' - they break subprocess and numpy.
worker_class = "gthread"

# Threads per worker
# 4 workers × 4 threads = 16 concurrent request slots
threads = 4

# Timeout for worker processes (seconds)
# Long timeout to accommodate large searches
# Searches can take several minutes for large corpora
timeout = 300

# Request timeout (seconds)
# How long to wait for request data from client
graceful_timeout = 30

# Keep-alive connections
keepalive = 2

# Maximum requests per worker before recycling
# Helps prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Preload application
# Loads application code before forking workers
# Saves memory through copy-on-write
preload_app = True

# Process naming
proc_name = "philologic5"

# Logging
# Access logs handled by Apache/Nginx; errors to file for easy tailing
accesslog = None
errorlog = "/var/log/philologic5/gunicorn-error.log"
loglevel = "info"
capture_output = True

# Security
# Limit request sizes to prevent DoS
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# User/Group (uncomment if running as root)
# user = "www-data"
# group = "www-data"

# Numba JIT thread count for parallel collocation kernels
# Defaults to CPU count if unset; cap to avoid contention with gunicorn workers
os.environ.setdefault("NUMBA_NUM_THREADS", "2")

# Numba cache directory — must be set BEFORE import numba (which happens during
# preload_app). If set after, Numba ignores it and falls back to __pycache__
# next to the source file, which www-data cannot write to.
os.environ.setdefault("NUMBA_CACHE_DIR", "/var/lib/philologic5/numba_cache")

# Working directory
# For central dispatcher mode, this should be /var/lib/philologic5/web_app/
# For per-database mode, this is the database's www/ directory
chdir = os.path.dirname(os.path.abspath(__file__))


def on_starting(server):
    """Called just before the master process is initialized."""
    # Ensure socket directory exists (only needed when binding to a Unix socket)
    if bind.startswith("unix:"):
        socket_dir = os.path.dirname(bind.split(":", 1)[1])
        if not os.path.exists(socket_dir):
            os.makedirs(socket_dir, mode=0o755)
    # Ensure log directory exists
    log_dir = os.path.dirname(errorlog)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, mode=0o755)


def worker_exit(server, worker):
    """Called when a worker exits."""
    # Clean up any worker-specific resources if needed
    pass
