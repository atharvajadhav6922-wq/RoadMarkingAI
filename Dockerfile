# ============================================================
# Dockerfile — RoadMarkingAI
# Base: python:3.11-slim
# ============================================================

FROM python:3.11-slim

# System dependencies: ffmpeg + OpenCV headless deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libgl1 \
        libgeos-dev \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app.py .
COPY config/ config/
COPY src/ src/
COPY models/ models/
COPY standards/ standards/

# Create data directories (bind-mounted at runtime, but ensure they exist in image)
RUN mkdir -p /app/data/uploads \
             /app/data/processed \
             /app/data/reports \
             /app/data/gps

# Streamlit telemetry off
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]
