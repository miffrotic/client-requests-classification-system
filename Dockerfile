FROM python:3.12-slim

ENV POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN pip3 install --no-cache-dir poetry

RUN poetry install --only main --no-cache \
    && rm -rf $POETRY_CACHE_DIR

COPY . ./

RUN alembic upgrade head

RUN useradd -m appuser

USER appuser

EXPOSE 8000

CMD ["uvicorn", "--reload", "--host=0.0.0.0", "--port=8000", "main:app"]