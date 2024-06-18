#!/bin/bash


PYTHON_INSTALL="\n## INSTALLING PYTHON LIBRARY ##"
# Create the virtual environment
pip3 install virtualenv
virtualenv /var/lib/philologic/philologic_env
source /var/lib/philologic/philologic_env/bin/activate
pip3 install build
echo -e "$PYTHON_INSTALL"
cd python;
rm -rf dist/
python3 -m build --sdist
pip3 install dist/*gz
deactivate
cd ..;
sudo cp philoload5 /usr/local/bin/
sudo mkdir -p /etc/philologic/
sudo mkdir -p /var/lib/philologic5/web_app/
sudo rm -rf /var/lib/philologic5/web_app/*
if [ -d www/app/node_modules ]
    then
        sudo rm -rf www/app/node_modules
fi
sudo cp -R www/* /var/lib/philologic5/web_app/
sudo cp www/.htaccess  /var/lib/philologic5/web_app/

if [ ! -f /etc/philologic/philologic5.cfg ]
    then
        db_url="# Set the filesytem path to the root web directory for your PhiloLogic install.
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
    echo -e "/etc/philologic/philologic5.cfg already exists"
    echo -e "Please delete and rerun the install script to avoid incompatibilities\n"
fi
