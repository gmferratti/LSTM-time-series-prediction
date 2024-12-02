FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY setup.py .
COPY pyproject.toml .
COPY README.md .
COPY src/ ./src
COPY secrets/key.json /app/credentials.json

RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=8080
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json

EXPOSE ${PORT}

CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT}