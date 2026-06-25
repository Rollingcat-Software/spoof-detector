# amispoof.fivucsas.com — static client-side anti-spoof demo served by nginx
# behind Traefik. Migrated off Hostinger (plan cancelled 2026-06) onto the box.
FROM nginx:1.27-alpine
RUN rm -f /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/amispoof.conf
COPY . /usr/share/nginx/html
RUN cd /usr/share/nginx/html && rm -rf \
    Dockerfile nginx.conf docker-compose.prod.yml .dockerignore \
    server.mjs README.md PIPELINE_DESIGN.md SUBDOMAIN_HTACCESS_TEMPLATE.txt .htaccess
EXPOSE 80
