FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libxml2-dev libxslt-dev zlib1g-dev \
        liblz4-tool ripgrep curl ca-certificates sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Use bash for RUN commands (nvm and uv installers need it)
SHELL ["/bin/bash", "-c"]

# Install PhiloLogic (uv, nvm, Node.js and Python are installed by install.sh)
COPY . /PhiloLogic5
WORKDIR /PhiloLogic5
RUN ./install.sh && mkdir -p /var/www/html/philologic

# Configure global variables
RUN sed -i 's/database_root = None/database_root = "\/var\/www\/html\/philologic\/"/' /etc/philologic/philologic5.cfg && \
    sed -i 's/url_root = None/url_root = "http:\/\/localhost\/philologic\/"/' /etc/philologic/philologic5.cfg

COPY docker_entrypoint.sh /docker_entrypoint.sh
RUN chmod +x /docker_entrypoint.sh

WORKDIR /

EXPOSE 8000
ENTRYPOINT ["/docker_entrypoint.sh"]
