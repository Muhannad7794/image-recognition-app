# 1. Start from the STABLE TensorFlow GPU image
FROM tensorflow/tensorflow:2.15.0-gpu

# Set the working directory 
WORKDIR /learning-block-dockerized

# Copy the requirements file in first to leverage Docker caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- PRE-DOWNLOAD WEIGHTS ---
# Create the directory Keras expects (adjust path for TensorFlow user if needed)
# The default Keras cache is often ~/.keras/
RUN mkdir -p ~/.keras/models/
# Download the specific MobileNetV2 weights file
RUN wget https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5 -O ~/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5
# --- END PRE-DOWNLOAD ---

# 6. Copy the rest of your model's source code
COPY . .

# run the model
ENTRYPOINT ["python3", "model.py"]