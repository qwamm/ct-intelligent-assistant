#!/bin/sh

echo "Connection to DB"

while ! nc -z ${SQL_HOST} ${SQL_PORT}; do
  sleep .2s
done

echo "Connected!"
ls -la
python ./dj_nnapi/manage.py makemigrations nnmodel
python ./dj_nnapi/manage.py migrate nnmodel
#export wsgi_start=1
cd ./dj_nnapi
python manage.py rundramatiq --processes 4 --threads 4 -v 2 --queues predict_all
#python3 -m celery -A dj_nnapi worker -P solo -l info --without-heartbeat --concurrency=1
# gunicorn -w 1 -b 0.0.0.0:8000 -t 120 --log-level debug dj_nnapi.wsgi:application

exec "$@"
