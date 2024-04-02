#!/bin/sh


PYTHON_INSTALL="\n## INSTALLING PYTHON LIBRARY ##"
sudo pip3 install build
echo "$PYTHON_INSTALL"
cd python;
rm -rf dist/
python3 -m build
sudo -H pip3 install dist/*whl --force-reinstall
sudo mkdir -p /etc/philologic/

cd ..;
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
        echo "$db_url" | sed "s/^ *//g" | sudo tee /etc/philologic/philologic5.cfg > /dev/null
        url_root="# Set the URL path to the same root directory for your philologic install.
        url_root = None
        # http://localhost/philologic/ is appropriate if you don't have a DNS hostname.\n"
        echo "$url_root" | sed "s/^ *//g" | sudo tee -a /etc/philologic/philologic5.cfg > /dev/null
        web_app_dir="## This should be set to the location of the PhiloLogic5 www directory
        web_app_dir = '/var/lib/philologic5/web_app/'"
        echo "$web_app_dir" | sed "s/^ *//g" | sudo tee -a /etc/philologic/philologic5.cfg > /dev/null
else
    echo "\n## WARNING ##"
    echo "/etc/philologic/philologic5.cfg already exists"
    echo "Please delete and rerun the install script to avoid incompatibilities\n"
fi
