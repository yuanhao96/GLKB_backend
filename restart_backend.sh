pkill gunicorn
gunicorn -b localhost:8000 -w 1 -D run:app --timeout 300

pkexec systemctl restart nginx

/etc/nginx/nginx.conf