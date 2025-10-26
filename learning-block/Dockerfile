# learning-block/Dockerfile
FROM tensorflow/tensorflow:2.15.0-gpu

# Set the working directory 
WORKDIR /learning-block-dockerized

# Copy the requirements file in first to leverage Docker caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your model's source code
COPY . .

# run the model
ENTRYPOINT ["python3", "model.py"]