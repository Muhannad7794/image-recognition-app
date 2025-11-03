# 1) Base image with TF 2.15
FROM tensorflow/tensorflow:2.15.0-gpu

# 2) Env
ENV KERAS_HOME=/root/.keras \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 3) Workdir
WORKDIR /finetune-dockerized

# 4) System deps for wget over HTTPS
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 5) PRE-DOWNLOAD Keras weights (96x96 no-top)
RUN mkdir -p $KERAS_HOME/models && \
    wget -q https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5 \
    -O $KERAS_HOME/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5

# 6) Copy requirements first (for caching)
COPY requirements.txt .


# 7) Now install the rest, but DO NOT let it upgrade numpy/ml-dtypes
#    If your requirements include those pins, this is still safe.
RUN pip install --no-cache-dir -r requirements.txt

# 8) Copy the rest of your code
COPY . .

# 9) Run training entrypoint (EI adds args)
ENTRYPOINT ["python3", "model.py"]
