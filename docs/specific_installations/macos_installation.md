---
title: Installing PhiloLogic5 on macOS
---

Tested on macOS Ventura and later (Apple Silicon and Intel).

### 1. Install System Dependencies

Install [Homebrew](https://brew.sh/) if you don't already have it, then:

```bash
brew install lz4 ripgrep
xcode-select --install   # if not already installed
```

### 2. Run the Installer

Download the [latest release](https://github.com/ARTFL-Project/PhiloLogic5/releases/latest) from GitHub, extract it, then:

```bash
cd PhiloLogic5-*/
./install.sh
```

The installer automatically detects macOS and adjusts user/group ownership accordingly.

### 3. Configure PhiloLogic

Edit `/etc/philologic/philologic5.cfg`:

```python
database_root = "/Library/WebServer/Documents/philologic/"
url_root = "http://localhost:8080/"
```

Create the database directory:

```bash
sudo mkdir -p /Library/WebServer/Documents/philologic
sudo chown -R $USER:staff /Library/WebServer/Documents/philologic
```

### 4. Configure Gunicorn for Direct Access

On macOS, the simplest setup is to bind Gunicorn directly to a TCP port instead of a Unix socket. Edit `/var/lib/philologic5/web_app/gunicorn.conf.py` and change:

```python
bind = "127.0.0.1:8080"
```

Make sure the port matches the one in your `url_root` above. This customization is preserved across reinstalls.

### 5. Start Gunicorn

You can run Gunicorn directly:

```bash
/var/lib/philologic5/philologic_env/bin/gunicorn \
    --config /var/lib/philologic5/web_app/gunicorn.conf.py \
    app:application
```

Or install the launchd service for automatic startup:

```bash
sudo cp /var/lib/philologic5/com.philologic5.gunicorn.plist /Library/LaunchDaemons/
sudo launchctl bootstrap system /Library/LaunchDaemons/com.philologic5.gunicorn.plist
```

### 6. Verify

PhiloLogic5 should now be accessible at `http://localhost:8080/`. Load a database with `philoload5` and navigate to it.

### Managing the Service

```bash
# Restart (launchd relaunches automatically via KeepAlive)
sudo launchctl kill SIGTERM system/com.philologic5.gunicorn

# Stop completely
sudo launchctl bootout system/com.philologic5.gunicorn

# Re-enable after stopping
sudo launchctl bootstrap system /Library/LaunchDaemons/com.philologic5.gunicorn.plist
```

### Differences from Linux

| | Linux | macOS |
|---|---|---|
| Service manager | systemd | launchd |
| Gunicorn bind | Unix socket (via reverse proxy) | TCP port (direct access) |
| Web server user | `www-data` | Current user |
| Typical `database_root` | `/var/www/html/philologic5/` | `/Library/WebServer/Documents/philologic/` |
| Reverse proxy | Required (Apache/Nginx) | Optional |
