#!/bin/bash

# Default Python version
PYTHON_VERSION="3.12"
INSTALL_TRANSFORMERS=false
NODE_MAJOR_VERSION="22"

# Parse command line arguments
while getopts "p:t" opt; do
  case $opt in
    p) PYTHON_VERSION="$OPTARG"
    ;;
    t) INSTALL_TRANSFORMERS=true
    ;;
    *) echo "Usage: $0 [-p python_version] [-t]"
       echo "  -p: Python version (default: 3.12)"
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

# Set uv cache to a writable location
export UV_CACHE_DIR=/var/lib/philologic5/.uv-cache

# Install uv if not present
if ! command -v uv &> /dev/null
then
    UV_LOCAL_DIR="$SCRIPT_DIR/.uv-bin"
    mkdir -p "$UV_LOCAL_DIR"
    echo "uv could not be found. Installing to $UV_LOCAL_DIR..."
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_LOCAL_DIR" sh
    export PATH="$UV_LOCAL_DIR:$PATH"
fi

# Delete virtual environment if it already exists
if [ -d /var/lib/philologic5 ]; then
    echo "Deleting existing PhiloLogic5 installation..."
    sudo rm -rf /var/lib/philologic5
fi

# Create base directory
sudo mkdir -p /var/lib/philologic5
sudo mkdir -p "$UV_CACHE_DIR"
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
    uv pip install "${PACKAGE_FILE}[transformers]" --quiet
else
    # Install without transformers
    echo "Installing without transformers support..."
    uv pip install "$PACKAGE_FILE" --quiet
fi

# Install Gunicorn WSGI server
echo "Installing Gunicorn..."
uv pip install gunicorn --quiet

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
sudo chown www-data:www-data /var/log/philologic5/
mkdir -p /var/lib/philologic5/web_app/
rm -rf /var/lib/philologic5/web_app/*

if [ -d app/node_modules ]; then
    rm -rf app/node_modules
fi

cp -R www/* /var/lib/philologic5/web_app/
cp www/.htaccess /var/lib/philologic5/web_app/
cp -R app /var/lib/philologic5/web_app/


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

# Install systemd service file for Gunicorn (non-fatal if no systemd)
if [ -d /etc/systemd/system ]; then
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
    echo '    <Location "/philologic5">'
    echo '        ProxyPass unix:/var/run/philologic/gunicorn.sock|http://localhost/philologic5'
    echo '        ProxyPassReverse unix:/var/run/philologic/gunicorn.sock|http://localhost/philologic5'
    echo '        ProxyTimeout 300'
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
    echo "  }"
    echo ""
    echo "  Adjust the URL prefix (/philologic5) to match your url_root setting"
    echo "  in /etc/philologic/philologic5.cfg"
fi

echo -e "\n## INSTALLATION COMPLETE ##"
echo "philoload5 installed to /usr/local/bin/philoload5"
