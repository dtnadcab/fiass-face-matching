FROM python:3.11-slim as builder

# Stage 1: Build stage with all build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime image
FROM python:3.11-slim
WORKDIR /app

# Install runtime dependencies for face_recognition, OpenCV, and other requirements
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libjpeg62-turbo \
    zlib1g \
    libtiff6 \
    libwebpdemux2 \
    libwebp7 \
    libgtk-3-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create necessary directories
RUN mkdir -p static/uploads

# Copy application code
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8089/face/check || exit 1

# Run as non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Start FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8089"]






# FROM python:3.11-slim

# # Install system dependencies needed to build dlib & face_recognition
# RUN apt-get update && apt-get install -y \
#     cmake \
#     build-essential \
#     libopenblas-dev \
#     liblapack-dev \
#     libx11-dev \
#     libgtk-3-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Set working directory
# WORKDIR /app

# # Copy and install Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # 👇 Create static/uploads folders
# RUN mkdir -p static/uploads

# # Copy application code
# COPY . .

# # Start FastAPI app with uvicorn
# CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8089"]
