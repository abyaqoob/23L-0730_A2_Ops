FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5001

WORKDIR /app

# Install minimal curl for health check and clean package lists
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Principle of least privilege: create dedicated non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data /app/models && \
    chown -R appuser:appuser /app

# Exploit layer caching: install dependencies before copying app source
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy application files (model is mounted at runtime via compose volume)
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser templates/ /app/templates/
COPY --chown=appuser:appuser static/ /app/static/
COPY --chown=appuser:appuser app.py /app/

USER appuser

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

CMD ["python", "app.py"]
