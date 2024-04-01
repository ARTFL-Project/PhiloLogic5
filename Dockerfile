FROM ubuntu:22.04


ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y software-properties-common

# Add the deadsnakes PPA
RUN add-apt-repository ppa:deadsnakes/ppa

# Install Python 3.12
RUN apt-get update && apt-get install -y python3.12 python3.12-venv python3.12-dev curl

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Update the alternatives to set Python 3.12 as the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends libxml2-dev libxslt-dev zlib1g-dev apache2 libgdbm-dev liblz4-tool brotli ripgrep gcc make wget sudo && \
    apt-get clean && rm -rf /var/lib/apt

# Install Node.js repository
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - &&\
    sudo apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt

# Install PhiloLogic
COPY . /PhiloLogic5
WORKDIR /PhiloLogic5
RUN sh install.sh && a2enmod rewrite && a2enmod cgi && a2enmod brotli

# Configure global variables
RUN sed -i 's/database_root = None/database_root = "\/var\/www\/html\/philologic\/"/' /etc/philologic/philologic5.cfg && \
    sed -i 's/url_root = None/url_root = "http:\/\/localhost\/philologic\/"/' /etc/philologic/philologic5.cfg

COPY docker_apache_restart.sh /autostart.sh
RUN chmod +x /autostart.sh

# Set up Apache
RUN perl -i -p0e 's/<Directory \/var\/www\/>\n\tOptions Indexes FollowSymLinks\n\tAllowOverride None/<Directory \/var\/www\/>\n\tOptions Indexes FollowSymLinks\n\tAllowOverride all/smg' /etc/apache2/apache2.conf
EXPOSE 80
ENTRYPOINT ["/autostart.sh"]