FROM python:3.11

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /dsp-block

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose Port 80
ENV HOST=0.0.0.0
ENV PORT=80
ENV PYTHONUNBUFFERED=1
EXPOSE 80

ENTRYPOINT ["python3", "-u", "dsp-server.py"]
