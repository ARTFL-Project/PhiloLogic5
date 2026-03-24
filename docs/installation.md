---
title: Installation
---

## Overview

PhiloLogic5 runs as a Gunicorn WSGI application. On Linux it typically runs behind a reverse proxy (Apache or Nginx); on macOS it can bind directly to a TCP port. The installer handles Python, Node.js, and all Python dependencies automatically via [uv](https://docs.astral.sh/uv/).

Installation steps:

1. Install system dependencies
2. Run `install.sh`
3. Configure `/etc/philologic/philologic5.cfg`
4. Set up your web server as a reverse proxy (Linux) or configure a TCP port (macOS)
5. Enable and start the Gunicorn service

## System Requirements

- Linux (Ubuntu 22.04+, Debian 12+, RHEL 9+, or similar) or macOS (Apple Silicon or Intel)
- Python 3.11+ (the installer downloads its own Python via uv by default)
- Root/sudo access for installing to `/var/lib/philologic5/`

## System Dependencies

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y \
    libxml2-dev libxslt-dev zlib1g-dev \
    liblz4-tool ripgrep curl
```

### RHEL / CentOS / Fedora

```bash
sudo dnf install -y \
    libxml2-devel libxslt-devel zlib-devel \
    lz4 ripgrep curl
```

### macOS

```bash
brew install lz4 ripgrep
```

Xcode Command Line Tools are also required (for C compiler headers used by `lxml`):

```bash
xcode-select --install
```

**Notes:**
- `libxml2-dev`/`libxslt-dev`/`zlib1g-dev`: required for building `lxml` (XML parsing). On macOS these are provided by Xcode Command Line Tools.
- `liblz4-tool`/`lz4`: used at database load time for compressing word indexes
- `ripgrep`: used at database load time for filtering parser output
- `curl`: used by the installer to download [uv](https://docs.astral.sh/uv/) and [nvm](https://github.com/nvm-sh/nvm). Pre-installed on macOS.
- The installer downloads its own Python via uv, so system Python packages are not required

## Installing PhiloLogic

Download the [latest release](https://github.com/ARTFL-Project/PhiloLogic5/releases/latest) from GitHub, extract it, and run the install script:

```bash
cd PhiloLogic5-*/   # or PhiloLogic5/ if you cloned via git
./install.sh
```

### Installer Options

| Flag | Description |
|------|-------------|
| `-t` | Install transformer support (includes spacy-transformers with CUDA) |

Example:

```bash
# Install with transformer support
./install.sh -t
```

### What the Installer Does

The installer:

1. Installs [uv](https://docs.astral.sh/uv/) (if not already present)
2. Downloads Python version via uv
3. Creates a virtual environment at `/var/lib/philologic5/philologic_env/`
4. Installs [nvm](https://github.com/nvm-sh/nvm) and Node.js 22 (for building the web app)
5. Builds and installs the PhiloLogic Python package with all dependencies (numpy, numba, lmdb, spacy, etc.)
6. Installs Gunicorn and Falcon
7. Copies the web application to `/var/lib/philologic5/web_app/`
8. Installs the `philoload5` command to `/usr/local/bin/`
9. Creates the global config at `/etc/philologic/philologic5.cfg` (if it doesn't exist)
10. Installs a systemd service file for Gunicorn (Linux) or a launchd plist (macOS)

### Installation Layout

```
/var/lib/philologic5/
├── philologic_env/       # Python virtual environment
├── web_app/              # Web application (Falcon + JS frontend)
│   ├── app.py            # WSGI entry point
│   ├── gunicorn.conf.py  # Gunicorn configuration
│   └── scripts/          # API endpoint scripts
├── nvm/                  # Node.js (used at load time for building the frontend)
├── bin/
│   └── philoload5        # Database loading command
└── numba_cache/          # JIT compilation cache

/etc/philologic/
└── philologic5.cfg       # Global configuration

/usr/local/bin/
└── philoload5            # Symlink to loader script
```

## Global Configuration

Edit `/etc/philologic/philologic5.cfg` to set two required paths:

```python
# Filesystem path where databases will be stored
database_root = "/var/www/html/philologic5/"

# URL root matching the database_root location
url_root = "http://localhost/philologic5/"
```

On macOS, typical values would be:

```python
database_root = "/Library/WebServer/Documents/philologic/"
url_root = "http://localhost:8080/"
```

Make sure the `database_root` directory exists and is writable by your user:

```bash
sudo mkdir -p /var/www/html/philologic5    # Linux
# or
sudo mkdir -p /Library/WebServer/Documents/philologic  # macOS

sudo chown -R $USER:$USER <database_root>
```

## Web Server Configuration

### Linux

On Linux, PhiloLogic5 runs behind Gunicorn, which listens on a Unix socket. You need a reverse proxy (Apache or Nginx) to forward HTTP requests to Gunicorn.

#### Starting Gunicorn

```bash
sudo systemctl enable philologic5-gunicorn
sudo systemctl start philologic5-gunicorn
```

Check status:

```bash
sudo systemctl status philologic5-gunicorn
journalctl -u philologic5-gunicorn -f   # follow logs
```

### macOS

On macOS, the simplest setup is to bind Gunicorn directly to a TCP port (no reverse proxy needed). Edit `/var/lib/philologic5/web_app/gunicorn.conf.py` and change the `bind` setting:

```python
bind = "127.0.0.1:8080"
```

Make sure the port matches the one in your `url_root` in `/etc/philologic/philologic5.cfg`.

You can then start Gunicorn manually:

```bash
/var/lib/philologic5/philologic_env/bin/gunicorn \
    --config /var/lib/philologic5/web_app/gunicorn.conf.py \
    app:application
```

Or use the installed launchd plist for automatic startup:

```bash
sudo cp /var/lib/philologic5/com.philologic5.gunicorn.plist /Library/LaunchDaemons/
sudo launchctl bootstrap system /Library/LaunchDaemons/com.philologic5.gunicorn.plist
```

To stop or restart:

```bash
# Restart (launchd will relaunch automatically since KeepAlive is set)
sudo launchctl kill SIGTERM system/com.philologic5.gunicorn

# Stop completely
sudo launchctl bootout system/com.philologic5.gunicorn
```

### Apache (Linux)

Enable the required modules:

```bash
sudo a2enmod proxy proxy_http
sudo systemctl restart apache2
```

Add to your `<VirtualHost>` block:

```apache
ProxyTimeout 300
<Location "/philologic5">
    ProxyPass unix:/var/run/philologic/gunicorn.sock|http://localhost/philologic5 flushpackets=on
    ProxyPassReverse unix:/var/run/philologic/gunicorn.sock|http://localhost/philologic5
    SetEnv no-gzip 1
    SetEnv force-no-buffering 1
</Location>
```

### Nginx (Linux)

Add to your `server` block:

```nginx
location /philologic5/ {
    proxy_pass http://unix:/var/run/philologic/gunicorn.sock;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_read_timeout 300s;
    proxy_buffering off;
}
```

Adjust the URL prefix (`/philologic5`) to match your `url_root` setting in `/etc/philologic/philologic5.cfg`.

## Docker

A `Dockerfile` is included for containerized deployment:

```bash
docker build -t philologic5 .
docker run -p 8000:8000 -v /path/to/databases:/var/www/html/philologic philologic5
```

In the container, Gunicorn binds directly to port 8000 (no reverse proxy needed inside the container).

## Tuning Gunicorn

The default configuration is in `/var/lib/philologic5/web_app/gunicorn.conf.py`. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `workers` | `min(cpu_count, 4)` | Number of worker processes |
| `threads` | `4` | Threads per worker |
| `timeout` | `300` | Request timeout (seconds) |
| `max_requests` | `1000` | Requests before worker recycling |
| `preload_app` | `True` | Preload app for memory efficiency |

The installer preserves any customizations to `gunicorn.conf.py` across reinstalls.

## Upgrading

To upgrade an existing installation, download the [latest release](https://github.com/ARTFL-Project/PhiloLogic5/releases/latest) and rerun the installer:

```bash
cd PhiloLogic5-*/
sudo ./install.sh
```

The installer will remove and recreate `/var/lib/philologic5/` but preserves your `gunicorn.conf.py` customizations and `/etc/philologic/philologic5.cfg`. Existing databases are not affected (they live under `database_root`).

After upgrading, restart Gunicorn:

```bash
# Linux
sudo systemctl restart philologic5-gunicorn

# macOS (launchd restarts automatically via KeepAlive)
sudo launchctl kill SIGTERM system/com.philologic5.gunicorn
```
