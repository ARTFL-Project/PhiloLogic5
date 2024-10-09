---
title: Installation
---

Installing PhiloLogic consists of two steps:

1. Run the install.sh script which installs PhiloLogic5 in `/var/lib/philologic5/`
2. Set up a directory in your web server to serve databases from
3. Edit /etc/philologic/philologic5.cfg according to your machine

You can find more detailed installation instructions for specific OSes here:

-   [RedHat (and CentOS)](specific_installations/redhat_installation.md)
-   [Ubuntu](specific_installations/ubuntu_installation.md)

### Downloading

IMPORTANT: Do not install from the master branch on github: this is the development branch and is in no way garanteed to be stable

You can find a copy of the latest version of PhiloLogic5 [here](../../../releases/).

### Prerequisites

-   Apache Webserver
-   Python 3.10 and up
-   LZ4
-   Brotli (for Apache compression)
-   Ripgrep

### Installing

Installing PhiloLogic's libraries requires administrator privileges.
Just run the install.sh in the top level directory of the PhiloLogic4 you downloaded to install PhiloLogic and its dependencies:

`./install.sh`

You can specify a different version of Python with the `-p` flag followed by the python executable to use, e.g.:
`./install.sh -p python3.12`

### <a name="global-config"></a>Global Configuration

The installer creates a file in `/etc/philologic/philologic5.cfg` which contains several important global variables:

-   `database_root` defines the filesytem path to the root web directory for your PhiloLogic install such as `/var/www/html/philologic`. Make sure your user or group has full write permissions to that directory.
-   `url_root` defines the URL path to the same root directory for your philologic install, such as http://localhost/philologic/

### Setting up PhiloLogic Web Application

Each new PhiloLogic database you load, containing one or more files, will be served
by a its own dedicated copy of PhiloLogic web application.
By convention, this database and web app reside together in a directory
accessible via an HTTP server configured to run Python CGI scripts.

Make sure you configure the `/etc/philologic/philologic5.cfg` appropriately.

Configuring your web server is outside of the scope of this document; but the web install
does come with a preconfigured .htaccess file that allows you to run the Web App.
Therefore, you need to make sure your server is configured to allow htaccess files.
