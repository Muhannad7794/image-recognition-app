# learning-block/Dockerfile
FROM python:3.11

# Set the working directory 
WORKDIR /learning-block-dockerized

# Copy the requirements file in first to leverage Docker caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your model's source code
COPY . .

# run the model
ENTRYPOINT ["python3"]