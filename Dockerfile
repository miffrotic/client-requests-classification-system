FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    gettext \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY myhotelki/ .

EXPOSE 8000

CMD ["bash", "-c", "\
    python manage.py migrate && \
    python manage.py collectstatic --noinput && \
    gunicorn myhotelki.wsgi:application --bind 0.0.0.0:8000"]