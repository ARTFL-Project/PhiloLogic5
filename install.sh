#!/bin/bash

PYTHON_INSTALL="\n## INSTALLING PYTHON LIBRARY ##"

# Default Python version
PYTHON_VERSION="python3"

# Parse command line arguments
while getopts "p:" opt; do
  case $opt in
    p) PYTHON_VERSION="$OPTARG"
    ;;
    *) echo "Usage: $0 [-p python_version]"
       exit 1
    ;;
  esac
done

echo "Using Python version: $PYTHON_VERSION"

# Check if virtualenv is installed
if ! command -v virtualenv &> /dev/null
then
    echo "virtualenv could not be found. Installing..."
    pip install virtualenv
fi

# Create the virtual environment
virtualenv -p $PYTHON_VERSION /var/lib/philologic/philologic_env

# Activate the virtual environment
source /var/lib/philologic/philologic_env/bin/activate

# Set the Numba cache directory
export NUMBA_CACHE_DIR="/tmp/numba"
mkdir -p /tmp/numba
chmod -R 775 /tmp/numba

# Install required packages
pip install build

echo -e "$PYTHON_INSTALL"
cd python
rm -rf dist/
python3 -m build --sdist
pip install dist/*gz

# Deactivate the virtual environment
deactivate

cd ..
sudo cp philoload5 /usr/local/bin/
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
    echo -e "$db_url" | sed "s/^ *//g" | sudo tee /etc/philologic/philologic5.cfg > /dev/null

    url_root="# Set the URL path to the same root directory for your philologic install.
    url_root = None
    # http://localhost/philologic/ is appropriate if you don't have a DNS hostname.\n"
    echo -e "$url_root" | sed "s/^ *//g" | sudo tee -a /etc/philologic/philologic5.cfg > /dev/null

    web_app_dir="## This should be set to the location of the PhiloLogic5 www directory
    web_app_dir = '/var/lib/philologic5/web_app/'"
    echo -e "$web_app_dir" | sed "s/^ *//g" | sudo tee -a /etc/philologic/philologic5.cfg > /dev/null
else
    echo -e "\n## WARNING ##"
    echo "/etc/philologic/philologic5.cfg already exists"
    echo "Please delete and rerun the install script to avoid incompatibilities\n"
fi
