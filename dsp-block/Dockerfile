FROM python:3.11

# Install OS dependencies required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /dsp-block


# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content to the container
COPY . .

# Server env: listen on all interfaces; **use 4446** (EI default)
ENV HOST=0.0.0.0
ENV PORT=4446
ENV PYTHONUNBUFFERED=1

EXPOSE 4446

# Unbuffered Python (-u) so logs appear immediately
ENTRYPOINT ["python3", "-u", "dsp-server.py"]