# Production Dockerfile for SciSynth AI Research Assistant
FROM python:3.10-slim

# Set environment variables for Streamlit and Python
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies and security updates
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r scisynth && useradd -r -g scisynth scisynth

WORKDIR /app

# Copy requirements files for exact version reproduction
COPY requirements.txt requirements.lock ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.lock && \
    pip cache purge

# Copy the rest of the code (excluding files in .dockerignore)
COPY . .

# Change ownership to non-root user
RUN chown -R scisynth:scisynth /app

# Switch to non-root user
USER scisynth

# Healthcheck: ensure the app responds on port 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.headless=true"] 