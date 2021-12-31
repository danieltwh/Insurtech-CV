#!/bin/sh
gunicorn wsgi:app -b 0.0.0.0:$PORT