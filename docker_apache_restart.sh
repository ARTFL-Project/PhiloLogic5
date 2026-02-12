#!/bin/bash
exec /var/lib/philologic5/philologic_env/bin/gunicorn \
    --config /var/lib/philologic5/web_app/gunicorn.conf.py \
    --bind 0.0.0.0:8000 \
    dispatcher:application
