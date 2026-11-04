# Churn prediction API image.
#   docker build -t churn-api:1.0 .
#   docker run --rm -p 8000:8000 churn-api:1.0
# Same Python version as the training venv, so the pickled model loads identically.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first: this layer is cached until requirements-api.txt changes
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Code, config and model change more often, so they come last
COPY src/ src/
COPY api/ api/
COPY config/ config/
COPY models/churn_model.pkl models/churn_model.pkl

# Run as a non-root user; the app writes logs/api.log, so logs/ must be writable
RUN useradd --create-home --uid 1000 appuser \
    && mkdir logs \
    && chown appuser logs
USER appuser

EXPOSE 8000

# Slim images have no curl, so Python does the request; non-2xx/connection error = unhealthy
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

# 0.0.0.0, not 127.0.0.1: otherwise only the container itself could reach the server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
