#!/bin/bash
set -e

# Python version — change here to update across all installs
PYTHON_VERSION="3.12"
INSTALL_TRANSFORMERS=false
NODE_MAJOR_VERSION="22"

# Parse command line arguments
while getopts "t" opt; do
  case $opt in
    t) INSTALL_TRANSFORMERS=true
    ;;
    *) echo "Usage: $0 [-t]"
       echo "  -t: Install transformers support (includes CUDA)"
       exit 1
    ;;
  esac
done

echo "Using Python version: $PYTHON_VERSION"
if [ "$INSTALL_TRANSFORMERS" = true ]; then
    echo "Transformers support will be installed (with CUDA)"
fi

# Get the directory where install.sh lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect OS and set platform-specific defaults
PHILO_ROOT="/var/lib/philologic5"
if [ "$(uname)" = "Darwin" ]; then
    IS_MACOS=true
    WEB_USER="$(whoami)"
    WEB_GROUP="staff"
    echo "Detected macOS — using user $WEB_USER for service ownership"
else
    IS_MACOS=false
    WEB_USER="www-data"
    WEB_GROUP="www-data"
fi

# Set uv cache in the repo directory so it survives reinstalls
export UV_CACHE_DIR="$SCRIPT_DIR/.uv-cache"

# Install uv if not present
if ! command -v uv &> /dev/null
then
    UV_LOCAL_DIR="$SCRIPT_DIR/.uv-bin"
    mkdir -p "$UV_LOCAL_DIR"
    echo "uv could not be found. Installing to $UV_LOCAL_DIR..."
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_LOCAL_DIR" sh
    export PATH="$UV_LOCAL_DIR:$PATH"
fi

# Preserve gunicorn config files for upgrade merge
GUNICORN_CONF="/var/lib/philologic5/web_app/gunicorn.conf.py"
GUNICORN_DEFAULTS="/var/lib/philologic5/web_app/gunicorn.conf.defaults.py"
GUNICORN_CONF_BACKUP=""
GUNICORN_DEFAULTS_BACKUP=""
if [ -f "$GUNICORN_CONF" ]; then
    GUNICORN_CONF_BACKUP=$(mktemp)
    cp "$GUNICORN_CONF" "$GUNICORN_CONF_BACKUP"
fi
if [ -f "$GUNICORN_DEFAULTS" ]; then
    GUNICORN_DEFAULTS_BACKUP=$(mktemp)
    cp "$GUNICORN_DEFAULTS" "$GUNICORN_DEFAULTS_BACKUP"
fi

# Delete virtual environment if it already exists
if [ -d /var/lib/philologic5 ]; then
    echo "Deleting existing PhiloLogic5 installation..."
    sudo rm -rf /var/lib/philologic5
fi

# Create base directory
sudo mkdir -p /var/lib/philologic5
mkdir -p "$UV_CACHE_DIR"
# Make writable by current user
sudo chown -R "$(id -u):$(id -g)" /var/lib/philologic5

# Install nvm and Node.js to a shared location
echo -e "\n## INSTALLING NVM AND NODE.JS ##"
export NVM_DIR=/var/lib/philologic5/nvm
mkdir -p "$NVM_DIR"

# Install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash

# Source nvm and install Node.js (latest in the major version series)
source "$NVM_DIR/nvm.sh"
nvm install $NODE_MAJOR_VERSION
nvm alias default $NODE_MAJOR_VERSION
nvm use default

# Verify node installation
echo "Node.js version: $(node --version)"
echo "npm version: $(npm --version)"

# Create symlink to npm in a fixed location
mkdir -p /var/lib/philologic5/bin
ln -sf "$(which npm)" /var/lib/philologic5/bin/npm

# Make nvm directory accessible to all users
chmod -R 755 "$NVM_DIR"
chmod -R 755 /var/lib/philologic5

# Create Numba cache directory with full write permissions
# The sticky bit (1777) ensures new files/dirs inherit group write permissions
mkdir -p /var/lib/philologic5/numba_cache
chmod -R 1777 /var/lib/philologic5/numba_cache

# Create IP whitelist cache directory writable by the web server user
mkdir -p /var/lib/philologic5/ip_cache
sudo chown "$WEB_USER:$WEB_GROUP" /var/lib/philologic5/ip_cache
chmod 755 /var/lib/philologic5/ip_cache

# Configure uv to install Python in /var/lib/philologic5/python
# This location is accessible by all users
export UV_PYTHON_INSTALL_DIR=/var/lib/philologic5/python
mkdir -p "$UV_PYTHON_INSTALL_DIR"

# Download and use uv-managed Python (e.g., "3.12" will download cpython-3.12.x)
# --managed-python forces uv to download its own Python instead of using system Python
uv venv /var/lib/philologic5/philologic_env --python $PYTHON_VERSION --managed-python

# Activate virtual environment
source /var/lib/philologic5/philologic_env/bin/activate

# Install build tool with uv
uv pip install build

echo -e "\n## INSTALLING PYTHON LIBRARY ##"
cd python
rm -rf dist/ philologic.egg-info/
python3 -m build --sdist

# Get the actual package filename
PACKAGE_FILE=$(ls dist/*.tar.gz)

if [ "$INSTALL_TRANSFORMERS" = true ]; then
    # Install with transformers extra - this uses the optional dependency from pyproject.toml
    echo "Installing with transformers support (CUDA enabled)..."
    uv pip install "${PACKAGE_FILE}[transformers]"
else
    # Install without transformers
    echo "Installing without transformers support..."
    uv pip install "$PACKAGE_FILE" --quiet
fi

# Install Gunicorn WSGI server and Falcon web framework
# Falcon ships pre-built Cython wheels on PyPI — binary install is ~5x faster at runtime
echo "Installing Gunicorn and Falcon..."
uv pip install gunicorn falcon --quiet

# Install test dependencies (without reinstalling philologic as editable)
echo "Installing test dependencies..."
uv pip install pytest pytest-timeout pytest-benchmark --quiet

# Deactivate virtual environment
deactivate

cd ..

# Install philoload5 script
echo -e '#!/bin/bash\n/var/lib/philologic5/philologic_env/bin/python3 -m philologic.loadtime "$@"' > /var/lib/philologic5/bin/philoload5
chmod 775 /var/lib/philologic5/bin/philoload5
sudo cp /var/lib/philologic5/bin/philoload5 /usr/local/bin/philoload5
sudo chmod 775 /usr/local/bin/philoload5

sudo mkdir -p /etc/philologic/
sudo mkdir -p /var/log/philologic5/
sudo chown "$WEB_USER:$WEB_GROUP" /var/log/philologic5/
mkdir -p /var/lib/philologic5/web_app/
rm -rf /var/lib/philologic5/web_app/*

if [ -d app/node_modules ]; then
    rm -rf app/node_modules
fi

cp -R www/* /var/lib/philologic5/web_app/
cp -R app /var/lib/philologic5/web_app/
# Delete appConfig.json if it exists
if [ -f /var/lib/philologic5/web_app/app/appConfig.json ]; then
    rm /var/lib/philologic5/web_app/app/appConfig.json
fi

# Upgrade gunicorn.conf.py, preserving user customizations
# At this point the new conf/defaults are already in place (from cp -R www/*).
# The backups contain the OLD installed versions.
if [ -n "$GUNICORN_CONF_BACKUP" ] && [ -n "$GUNICORN_DEFAULTS_BACKUP" ]; then
    echo "Upgrading gunicorn.conf.py (preserving user customizations)..."
    if /var/lib/philologic5/philologic_env/bin/python3 -c "
from philologic.utils.upgrade_gunicorn_conf import upgrade_gunicorn_conf
import sys
old_conf, old_defaults = sys.argv[1], sys.argv[2]
customized = upgrade_gunicorn_conf(
    old_conf=old_conf,
    old_defaults=old_defaults,
    new_conf='$GUNICORN_CONF',
    new_defaults='$GUNICORN_DEFAULTS',
)
if customized:
    print(f'  Preserved user settings: {\", \".join(customized)}')
else:
    print('  No user customizations detected')
" "$GUNICORN_CONF_BACKUP" "$GUNICORN_DEFAULTS_BACKUP"; then
        rm -f "$GUNICORN_CONF_BACKUP" "$GUNICORN_DEFAULTS_BACKUP"
    else
        echo "  WARNING: upgrade failed, restoring previous gunicorn.conf.py"
        cp "$GUNICORN_CONF_BACKUP" "$GUNICORN_CONF"
        rm -f "$GUNICORN_CONF_BACKUP" "$GUNICORN_DEFAULTS_BACKUP"
    fi
elif [ -n "$GUNICORN_CONF_BACKUP" ]; then
    # Old install without defaults file — keep user's conf as-is
    echo "Restoring existing gunicorn.conf.py (no defaults file to diff against)"
    cp "$GUNICORN_CONF_BACKUP" "$GUNICORN_CONF"
    rm -f "$GUNICORN_CONF_BACKUP"
fi


if [ ! -f /etc/philologic/philologic5.cfg ]; then
    db_url="# Set the filesystem path to the root web directory for your PhiloLogic install.
    database_root = None
    # /var/www/html/philologic/ is conventional for linux,
    # /Library/WebServer/Documents/philologic for Mac OS.\n"
    echo -e "$db_url" | sed "s/^ *//g" | sudo tee /etc/philologic/philologic5.cfg > /dev/null

    url_root="# Set the URL path to the same root directory for your philologic install.
    url_root = None
    # http://localhost/philologic/ is appropriate if you don't have a DNS hostname.\n"
    echo -e "$url_root" | sed "s/^ *//g" | sudo tee -a /etc/philologic/philologic5.cfg > /dev/null
else
    echo -e "\n## WARNING ##"
    echo "/etc/philologic/philologic5.cfg already exists"
    echo "Please delete and rerun the install script to avoid incompatibilities\n"
fi

# Install service file for Gunicorn (platform-specific)
if [ "$IS_MACOS" = true ]; then
    # Install launchd plist for macOS
    PLIST_LABEL="com.philologic5.gunicorn"
    PLIST_SRC="$SCRIPT_DIR/${PLIST_LABEL}.plist"
    PLIST_DEST="$PHILO_ROOT/${PLIST_LABEL}.plist"

    # Fill in the current user in the plist template
    sed "s|__WEB_USER__|$WEB_USER|g" "$PLIST_SRC" > "$PLIST_DEST"

    # Restart if already running, otherwise print setup instructions
    if sudo launchctl print system/$PLIST_LABEL &> /dev/null; then
        sudo cp "$PLIST_DEST" /Library/LaunchDaemons/
        sudo launchctl kickstart -k system/$PLIST_LABEL
        echo -e "\n## GUNICORN SERVICE RESTARTED ##"
    else
        echo -e "\n## LAUNCHD SERVICE ##"
        echo "Plist installed to $PLIST_DEST"
        echo ""
        echo "To install and start the service:"
        echo "  sudo cp $PLIST_DEST /Library/LaunchDaemons/"
        echo "  sudo launchctl bootstrap system /Library/LaunchDaemons/${PLIST_LABEL}.plist"
        echo ""
        echo "To stop:"
        echo "  sudo launchctl bootout system/$PLIST_LABEL"
    fi

elif [ -d /etc/systemd/system ] && command -v systemctl &> /dev/null; then
    sudo cp "$SCRIPT_DIR/philologic5-gunicorn.service" /etc/systemd/system/
    sudo systemctl daemon-reload
    if systemctl is-active --quiet philologic5-gunicorn 2>/dev/null; then
        echo "Restarting philologic5-gunicorn service..."
        sudo systemctl restart philologic5-gunicorn
    fi
    echo -e "\n## GUNICORN SERVICE INSTALLED ##"
    echo "To enable and start the WSGI server:"
    echo "  sudo systemctl enable philologic5-gunicorn"
    echo "  sudo systemctl start philologic5-gunicorn"
    echo ""
    echo "Configure your web server as a reverse proxy to the Gunicorn socket."
    echo "A single rule covers ALL databases under the PhiloLogic URL root."
    echo ""
    echo "=== Apache ==="
    echo "  Requires: mod_proxy, mod_proxy_http"
    echo "  sudo a2enmod proxy proxy_http"
    echo ""
    echo "  Add to your <VirtualHost> block:"
    echo '    ProxyTimeout 300'
    echo '    <Location "/philologic5">'
    echo '        ProxyPass unix:/var/run/philologic/gunicorn.sock|http://localhost/philologic5 flushpackets=on'
    echo '        ProxyPassReverse unix:/var/run/philologic/gunicorn.sock|http://localhost/philologic5'
    echo '        SetEnv no-gzip 1'
    echo '        SetEnv force-no-buffering 1'
    echo '    </Location>'
    echo ""
    echo "=== Nginx ==="
    echo "  location /philologic5/ {"
    echo "      proxy_pass http://unix:/var/run/philologic/gunicorn.sock;"
    echo "      proxy_set_header Host \$host;"
    echo "      proxy_set_header X-Real-IP \$remote_addr;"
    echo "      proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;"
    echo "      proxy_set_header X-Forwarded-Proto \$scheme;"
    echo "      proxy_read_timeout 300s;"
    echo "      proxy_buffering off;"
    echo "  }"
    echo ""
    echo "  Adjust the URL prefix (/philologic5) to match your url_root setting"
    echo "  in /etc/philologic/philologic5.cfg"
fi

echo -e "\n## INSTALLATION COMPLETE ##"
echo "philoload5 installed to /usr/local/bin/philoload5"
