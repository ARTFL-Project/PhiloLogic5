#!/usr/bin/env python

import sys
sys.path.append('..')
from functions.wsgi_handler import wsgi_response
from render_template import render_template
import functions as f


def landing_page(environ,start_response):
    db, dbname, path_components, q = wsgi_response(environ,start_response)
    config = f.WebConfig(db.locals)
    return render_template(db=db,dbname=dbname,form=True, q=q, template_name='landing_page.mako',
                           config=config, report="landing_page")
