Tested on Ubuntu 22.04 and 24.04.

### 1. Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
    libxml2-dev libxslt-dev zlib1g-dev \
    liblz4-tool ripgrep curl
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
sudo a2enmod proxy proxy_http
```

Add to your `/etc/apache2/sites-available/000-default.conf` (inside the `<VirtualHost>` block):

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
sudo systemctl restart apache2
```

PhiloLogic5 should now be accessible at `http://localhost/philologic5/`.
