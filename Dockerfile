FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS base

# Update package lists and install dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    tar \
    wget \
    git \
    bash \
    vim \
    gcc \
    g++ \
    build-essential \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python and pip
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /root/app

# Set proxy environment variables
# ENV HTTP_PROXY=
# ENV HTTPS_PROXY=

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Set environment variables
ENV API_HOST=0.0.0.0
ENV API_PORT=8080

# Download model
COPY download_model.py .
RUN python download_model.py

# Run the application
COPY api.py .
CMD ["python", "api.py"]