# Start from Ubuntu 22.04
FROM ubuntu:22.04

# Run as root (default) and set non-interactive
USER root
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python, pip, and system libs for OpenCV
# We include all the libs we've found necessary
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements FIRST and install as ROOT (system-wide)
# This fixes the "[Errno 2] No such file" error
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# EXPOSE Port 4446 (matching your supervisor's working example)
# This fixes the "ETIMEDOUT" error
ENV HOST=0.0.0.0
ENV PORT=4446
EXPOSE 4446

RUN echo "Contents of /app:" && ls -la

# Run the server as root
CMD python3 -u dsp-server.py