#!/bin/bash

# Default Python version
PYTHON_VERSION="python3"
INSTALL_TRANSFORMERS=false

# Parse command line arguments
while getopts "p:t" opt; do
  case $opt in
    p) PYTHON_VERSION="$OPTARG"
    ;;
    t) INSTALL_TRANSFORMERS=true
    ;;
    *) echo "Usage: $0 [-p python_version] [-t]"
       echo "  -p: Python version (default: python3)"
       echo "  -t: Install transformers support (includes CUDA)"
       exit 1
    ;;
  esac
done

echo "Using Python version: $PYTHON_VERSION"
if [ "$INSTALL_TRANSFORMERS" = true ]; then
    echo "Transformers support will be installed (with CUDA)"
fi

# Install uv if not present
if ! command -v uv &> /dev/null
then
    echo "uv could not be found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Delete virtual environment if it already exists
if [ -d /var/lib/philologic5 ]; then
    echo "Deleting existing PhiloLogic5 installation..."
    sudo rm -rf /var/lib/philologic5
fi

# Create virtual environment
sudo mkdir -p /var/lib/philologic5
sudo chown -R $USER:$USER /var/lib/philologic5
uv venv /var/lib/philologic5/philologic_env --python $PYTHON_VERSION

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
echo -e '#!/bin/bash\nsource /var/lib/philologic5/philologic_env/bin/activate\npython3 -m philologic.loadtime "$@"\ndeactivate' > philoload5 && sudo mv philoload5 /usr/local/bin/
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
