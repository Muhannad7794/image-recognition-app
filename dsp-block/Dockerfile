# Start from the gold-standard Ubuntu 22.04
FROM ubuntu:22.04

# Set user to root *only* for the setup
USER root

# Avoid interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, pip, system libs for OpenCV, AND libcap2-bin (for setcap)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libcap2-bin \
 && rm -rf /var/lib/apt/lists/*

# Grant the python3 binary permission to bind to low ports
RUN setcap 'cap_net_bind_service=+ep' $(readlink -f /usr/bin/python3)

# Now, create a non-root user to run the server
RUN useradd --create-home --shell /bin/bash ei-user
WORKDIR /home/ei-user/dsp-block
USER ei-user

# Install python packages (as non-root)
COPY --chown=ei-user:ei-user requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the block code (as non-root)
COPY --chown=ei-user:ei-user . .

# Set Port to 80. Our python3 binary now has permission.
ENV HOST=0.0.0.0
ENV PORT=80
ENV PYTHONUNBUFFERED=1
EXPOSE 80

RUN echo "Contents of /dsp-block:" && ls -la

# Run the server (as ei-user)
ENTRYPOINT ["python3", "-u", "dsp-server.py"]