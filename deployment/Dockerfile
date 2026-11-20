
FROM python:3.12-slim

# Python configuration
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Working directory
WORKDIR /app

# Install runtime dependencies
COPY deployment/requirements-runtime.txt /app/requirements-runtime.txt

RUN python -m pip install --no-cache-dir \
    -r /app/requirements-runtime.txt

# Copy application code
COPY src/ /app/src/
COPY dashboard_app.py /app/dashboard_app.py

# Create directories for mounted reports and models
RUN mkdir -p \
    /app/reports \
    /app/models \
    /app/data/warehouse

# Default API port
EXPOSE 8000

# Default command (overridden in Docker Compose)
CMD [
    "python",
    "-m",
    "uvicorn",
    "src.api.main:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000"
]
