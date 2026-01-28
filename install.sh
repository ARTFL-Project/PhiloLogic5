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

# Install uv system-wide if not present
if ! command -v uv &> /dev/null
then
    echo "uv could not be found. Installing system-wide..."
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh
fi

# Delete virtual environment if it already exists
if [ -d /var/lib/philologic5 ]; then
    echo "Deleting existing PhiloLogic5 installation..."
    sudo rm -rf /var/lib/philologic5
fi

# Install nvm and Node.js to a shared location
echo -e "\n## INSTALLING NVM AND NODE.JS ##"
export NVM_DIR=/var/lib/philologic5/nvm
sudo mkdir -p "$NVM_DIR"
sudo chown -R $USER:$USER "$NVM_DIR"

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
sudo mkdir -p /var/lib/philologic5/bin
sudo ln -sf "$(which npm)" /var/lib/philologic5/bin/npm

# Make nvm directory accessible to all users
sudo chmod -R 755 "$NVM_DIR"

# Create base directory with write permissions for current user
sudo mkdir -p /var/lib/philologic5
sudo chown -R $USER:$USER /var/lib/philologic5
sudo chmod -R 755 /var/lib/philologic5

# Create Numba cache directory with full write permissions
# The sticky bit (1777) ensures new files/dirs inherit group write permissions
sudo mkdir -p /var/lib/philologic5/numba_cache
sudo chmod -R 1777 /var/lib/philologic5/numba_cache

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
rm -rf dist/
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

# Deactivate virtual environment
deactivate

cd ..

# Install philoload5 script
echo -e '#!/bin/bash\n/var/lib/philologic5/philologic_env/bin/python3 -m philologic.loadtime "$@"' > philoload5
sudo mv philoload5 /usr/local/bin/
sudo chmod 775 /usr/local/bin/philoload5
sudo mkdir -p /etc/philologic/
sudo mkdir -p /var/lib/philologic5/web_app/
sudo rm -rf /var/lib/philologic5/web_app/*

if [ -d www/app/node_modules ]; then
    sudo rm -rf www/app/node_modules
fi

sudo cp -R www/* /var/lib/philologic5/web_app/
sudo cp www/.htaccess /var/lib/philologic5/web_app/

if [ ! -f /etc/philologic/philologic5.cfg ]; then
    db_url="# Set the filesystem path to the root web directory for your PhiloLogic install.
    database_root = None
    # /var/www/html/philologic/ is conventional for linux,
    # /Library/WebServer/Documents/philologic for Mac OS.\n"
    sudo echo -e "$db_url" | sed "s/^ *//g" | sudo tee /etc/philologic/philologic5.cfg > /dev/null

    url_root="# Set the URL path to the same root directory for your philologic install.
    url_root = None
    # http://localhost/philologic/ is appropriate if you don't have a DNS hostname.\n"
    sudo echo -e "$url_root" | sed "s/^ *//g" | sudo tee -a /etc/philologic/philologic5.cfg > /dev/null
else
    echo -e "\n## WARNING ##"
    echo "/etc/philologic/philologic5.cfg already exists"
    echo "Please delete and rerun the install script to avoid incompatibilities\n"
fi
