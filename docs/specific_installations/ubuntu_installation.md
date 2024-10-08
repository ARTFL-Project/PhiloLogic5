---
title: Installing PhiloLogic5 on Ubuntu
---

-   The following dependencies need to be installed:

    -   libxml2-dev
    -   libxslt-dev
    -   zlib1g-dev
    -   apache2
    -   python3-pip
    -   liblz4-tool
    -   brotli
    -   ripgrep

    Run the following command:

    `sudo apt-get install libxml2-dev libxslt-dev zlib1g-dev apache2 python3-pip liblz4-tool brotli ripgrep`

-   Run install script inside the PhiloLogic5 directory

    `./install.sh`

-   Set-up Apache:
    -   enable mod_rewrite: `sudo a2enmod rewrite`
    -   enable mod_cgi: `sudo a2enmod cgi`
    -   enable brotli: `sudo a2enmod brotli`
    -   Make sure to set `AllowOverride` to `all` for the directory containined your philologic databases in your Apache config
