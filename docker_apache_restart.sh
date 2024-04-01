#!/bin/bash
service apache2 stop
rm /var/run/apache2/*
apachectl -D FOREGROUND