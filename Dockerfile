FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12
RUN apt-get update && apt-get install -y python3 python3-venv python3-dev curl python3-pip

# Install dependencies (no Apache needed — Gunicorn serves directly)
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends libxml2-dev libxslt-dev zlib1g-dev libgdbm-dev liblz4-tool brotli ripgrep gcc make wget sudo && \
    apt-get clean && rm -rf /var/lib/apt

# Install PhiloLogic (nvm and Node.js are installed by install.sh)
COPY . /PhiloLogic5
WORKDIR /PhiloLogic5
# Delete the tests directory to reduce image size
RUN rm -rf tests
RUN ./install.sh && mkdir -p /var/www/html/philologic

# Configure global variables
RUN sed -i 's/database_root = None/database_root = "\/var\/www\/html\/philologic\/"/' /etc/philologic/philologic5.cfg && \
    sed -i 's/url_root = None/url_root = "http:\/\/localhost\/philologic\/"/' /etc/philologic/philologic5.cfg

COPY docker_apache_restart.sh /autostart.sh
RUN chmod +x /autostart.sh

WORKDIR /

EXPOSE 8000
ENTRYPOINT ["/autostart.sh"]
