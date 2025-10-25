FROM python:3.11

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /dsp-block

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose Port
ENV HOST=0.0.0.0
ENV PORT=4446
ENV PYTHONUNBUFFERED=1
EXPOSE 4446

ENTRYPOINT ["python3", "-u", "dsp-server.py"]
