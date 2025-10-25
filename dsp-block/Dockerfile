FROM python:3.11

# Install OS dependencies required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /dsp-block-dockerized

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content to the container
COPY . .

# Define the entrypoint for the Edge Impulse runner
ENTRYPOINT ["python3", "dsp.py"]