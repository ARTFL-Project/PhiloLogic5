"""Gunicorn configuration for PhiloLogic5.

This configuration is used when running PhiloLogic under Gunicorn instead of
Apache CGI. It provides better performance and resource efficiency.

Usage:
    gunicorn --config gunicorn.conf.py dispatcher:application

Or via systemd service (see install.sh for service file creation).
"""

import multiprocessing
import os

# Server socket
bind = "unix:/var/run/philologic/gunicorn.sock"

# Worker processes
# Formula: (2 x CPU cores) + 1 is a good starting point
# Adjust based on your workload and available memory
workers = min(multiprocessing.cpu_count() * 2 + 1, 8)

# Worker class
# 'sync' is the default and works well with subprocess spawning
# Do NOT use 'gevent' or 'eventlet' - they don't work well with subprocess
worker_class = "sync"

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

# Security
# Limit request sizes to prevent DoS
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# User/Group (uncomment if running as root)
# user = "www-data"
# group = "www-data"

# Working directory
# For central dispatcher mode, this should be /var/lib/philologic5/web_app/
# For per-database mode, this is the database's www/ directory
chdir = os.path.dirname(os.path.abspath(__file__))


def on_starting(server):
    """Called just before the master process is initialized."""
    # Ensure socket directory exists with correct permissions
    socket_dir = "/var/run/philologic"
    if not os.path.exists(socket_dir):
        os.makedirs(socket_dir, mode=0o755)


def worker_exit(server, worker):
    """Called when a worker exits."""
    # Clean up any worker-specific resources if needed
    pass
