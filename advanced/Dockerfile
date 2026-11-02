# Multi-Stage Production Dockerfile for KnowledgeForge AI
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY advanced/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy monorepo source code
COPY . .

# Set PYTHONPATH so absolute imports resolve correctly
ENV PYTHONPATH="/app"

EXPOSE 8000 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command launches FastAPI backend
CMD ["uvicorn", "advanced.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
