---
title: Installing PhiloLogic5 on RedHat (and CentOS)
---

Tested on RHEL 9 and CentOS Stream 9.

### 1. Install System Dependencies

```bash
sudo dnf install -y \
    libxml2-devel libxslt-devel zlib-devel \
    lz4 ripgrep curl
```

### 2. Run the Installer

Download the [latest release](https://github.com/ARTFL-Project/PhiloLogic5/releases/latest) from GitHub, extract it, then:

```bash
cd PhiloLogic5-*/
./install.sh
```

### 3. Configure PhiloLogic

Edit `/etc/philologic/philologic5.cfg`:

```python
database_root = "/var/www/html/philologic5/"
url_root = "http://localhost/philologic5/"
```

Create the database directory:

```bash
sudo mkdir -p /var/www/html/philologic5
sudo chown -R $USER:$USER /var/www/html/philologic5
```

### 4. Start Gunicorn

```bash
sudo systemctl enable philologic5-gunicorn
sudo systemctl start philologic5-gunicorn
```

### 5. Configure Apache as Reverse Proxy

```bash
sudo dnf install -y httpd mod_proxy_html
```

Add to `/etc/httpd/conf.d/philologic5.conf`:

```apache
ProxyTimeout 300
<Location "/philologic5">
    ProxyPass unix:/var/run/philologic/gunicorn.sock|http://localhost/philologic5 flushpackets=on
    ProxyPassReverse unix:/var/run/philologic/gunicorn.sock|http://localhost/philologic5
    SetEnv no-gzip 1
    SetEnv force-no-buffering 1
</Location>
```

Restart Apache:

```bash
sudo systemctl restart httpd
```

Make sure the correct permissions are set on the database directory — the user building databases needs write access.
