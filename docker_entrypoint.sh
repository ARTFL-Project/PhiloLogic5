#!/bin/bash
# The secret that signs auth cookies (see install.sh), the container's own
if [ ! -s /etc/philologic/philologic5.secret ]; then
    (umask 077 && /var/lib/philologic5/philologic_env/bin/python3 -c "import secrets; print(secrets.token_hex(32))" > /etc/philologic/philologic5.secret)
fi
exec /var/lib/philologic5/philologic_env/bin/gunicorn \
    --config /var/lib/philologic5/web_app/gunicorn.conf.py \
    --bind 0.0.0.0:8000 \
    app:application
