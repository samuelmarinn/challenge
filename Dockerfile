# syntax=docker/dockerfile:1.2
FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["sh", "-c", "uvicorn challenge.api:app --host 0.0.0.0 --port ${PORT:-8000}"]