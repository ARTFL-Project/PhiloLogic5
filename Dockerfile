FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libxml2-dev libxslt-dev zlib1g-dev \
        libicu-dev pkg-config g++ \
        liblz4-tool ripgrep curl ca-certificates sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Use bash for RUN commands (nvm and uv installers need it)
SHELL ["/bin/bash", "-c"]

# Install PhiloLogic (uv, nvm, Node.js and Python are installed by install.sh)
COPY . /PhiloLogic5
WORKDIR /PhiloLogic5
# Without the secret install.sh makes, which every container of the image would share: docker_entrypoint.sh draws one
RUN ./install.sh && rm /etc/philologic/philologic5.secret && mkdir -p /var/www/html/philologic

# Configure global variables
RUN sed -i 's/database_root = None/database_root = "\/var\/www\/html\/philologic\/"/' /etc/philologic/philologic5.cfg

COPY docker_entrypoint.sh /docker_entrypoint.sh
RUN chmod +x /docker_entrypoint.sh

WORKDIR /

EXPOSE 8000
ENTRYPOINT ["/docker_entrypoint.sh"]
