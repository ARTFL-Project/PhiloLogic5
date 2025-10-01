FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12
RUN apt-get update && apt-get install -y python3 python3-venv python3-dev curl python3-pip

# Install dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends libxml2-dev libxslt-dev zlib1g-dev apache2 libgdbm-dev liblz4-tool brotli ripgrep gcc make wget sudo && \
    apt-get clean && rm -rf /var/lib/apt

# Install nvm and Node.js
ENV NVM_DIR=/root/.nvm
ENV NODE_VERSION=22.14.0
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash \
    && . $NVM_DIR/nvm.sh \
    && nvm install $NODE_VERSION \
    && nvm alias default $NODE_VERSION \
    && nvm use default
ENV PATH=$NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

# Install PhiloLogic
COPY . /PhiloLogic5
WORKDIR /PhiloLogic5
RUN ./install.sh -p python3 && a2enmod rewrite && a2enmod cgi && a2enmod brotli && a2enmod headers

# Configure global variables
RUN sed -i 's/database_root = None/database_root = "\/var\/www\/html\/philologic\/"/' /etc/philologic/philologic5.cfg && \
    sed -i 's/url_root = None/url_root = "http:\/\/localhost\/philologic\/"/' /etc/philologic/philologic5.cfg

# Set up the autostart script
COPY docker_apache_restart.sh /autostart.sh
RUN chmod +x /autostart.sh

WORKDIR /

# Set up Apache configuration
RUN perl -i -p0e 's/<Directory \/var\/www\/>\n\tOptions Indexes FollowSymLinks\n\tAllowOverride None/<Directory \/var\/www\/>\n\tOptions Indexes FollowSymLinks\n\tAllowOverride all/smg' /etc/apache2/apache2.conf
EXPOSE 80
ENTRYPOINT ["/autostart.sh"]