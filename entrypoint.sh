#!/bin/sh

echo "Connection to DB"
echo "$PWD"
ls -la ./server

while ! nc -z ${SQL_HOST} ${SQL_PORT}; do
  sleep .2s
done

echo "Connected!"

if [ $CELERY -eq 1 ]
then
  bash -c "cd ./server && python -m celery -A medweb worker"
else
  cd server
  python manage.py makemigrations
  echo "MIGRATION BASE!"

  python manage.py makemigrations medml
  echo "MIGRATION MEDML!"
  python manage.py makemigrations inner_mail
  echo "MIGRATION inner_mail!"
  python manage.py migrate

  python manage.py base_configuration
  #python manage.py collectstatic --noinput

  python manage.py runserver

fi

exec "$@"