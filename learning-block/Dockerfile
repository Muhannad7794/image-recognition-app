# 1. Start from the STABLE TensorFlow GPU image
FROM tensorflow/tensorflow:2.15.0-gpu

# Make Keras cache path explicit and keep logs quiet
ENV KERAS_HOME=/root/.keras \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONUNBUFFERED=1

# Set the working directory 
WORKDIR /learning-block-dockerized

# Copy the requirements file in first to leverage Docker caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure we have wget and proper CA certs for HTTPS
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# --- PRE-DOWNLOAD WEIGHTS ---
# Create the directory Keras expects and fetch the exact 96x96 no-top weights
RUN mkdir -p $KERAS_HOME/models && \
    wget -q https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5 \
      -O $KERAS_HOME/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5
# --- END PRE-DOWNLOAD ---

# Copy the rest of your model's source code
COPY . .

# Run the model (Edge Impulse will append args to this entrypoint)
ENTRYPOINT ["python3", "model.py"]
