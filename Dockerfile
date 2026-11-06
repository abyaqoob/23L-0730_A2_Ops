# 1. Slim base image for reduced attack surface and minimal footprint
FROM python:3.11-slim

# Set environment variables for clean container execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5001 \
    MODEL_PATH=models/champion_model.keras \
    DECISION_THRESHOLD=0.4494 \
    DATABASE_PATH=data/visionguard_audit.db

# Set isolated working directory
WORKDIR /app

# 2. Install curl for HEALTHCHECK and clean apt caches
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Create non-root system user and necessary directories
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data /app/models && \
    chown -R appuser:appuser /app

# 4. Layer Cache Optimization: copy and install dependencies first
COPY requirements-docker.txt /app/
RUN pip install --no-cache-dir -r requirements-docker.txt

# 5. Copy application source code and models
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser templates/ /app/templates/
COPY --chown=appuser:appuser static/ /app/static/
COPY --chown=appuser:appuser models/champion_model.keras /app/models/
COPY --chown=appuser:appuser app.py /app/

# 6. Switch to unprivileged non-root user
USER appuser

# Expose internal serving port
EXPOSE 5001

# 7. Container Health Check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Start the production Flask web server
CMD ["python", "app.py"]
