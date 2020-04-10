heroku buildpacks:clear
heroku buildpacks:add --index heroku/python
heroku ps:scale web=0
heroku ps:scale web=1
web: gunicorn gettingstarted.wsgi --log-file -