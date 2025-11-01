# learning-block/baseline/model.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    Rescaling,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import json
import argparse
import sys
import traceback

# --- Script Arguments ---
parser = argparse.ArgumentParser(description="Train image classification model")
parser.add_argument("--data-directory", type=str, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--learning-rate", type=float, required=True)
parser.add_argument("--out-directory", type=str, required=True)
args, unknown = parser.parse_known_args()

# --- Define image shape ---
IMG_HEIGHT = 160
IMG_WIDTH = 160
CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

# --- Load Data ---
print(f"Loading data from directory: {args.data_directory}")

# Define split-aware file paths for both train and val
paths = {
    "x_train": ["X_split_train.npy", "X_train_features.npy"],
    "y_train": ["Y_split_train.npy", "y_train.npy"],
    "x_val": ["X_split_test.npy", "X_validate_features.npy"],
    "y_val": ["Y_split_test.npy", "y_validate.npy"],
}


def pick(path_list):
    for name in path_list:
        p = os.path.join(args.data_directory, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of {path_list} found under {args.data_directory}")


x_train_path = pick(paths["x_train"])
y_train_path = pick(paths["y_train"])
x_val_path = pick(paths["x_val"])
y_val_path = pick(paths["y_val"])

print("Directory listing:", sorted(os.listdir(args.data_directory)))
print(f"Train X: {x_train_path}")
print(f"Train y: {y_train_path}")
print(f"Val   X: {x_val_path}")
print(f"Val   y: {y_val_path}")

x_train = np.load(x_train_path)
y_train = np.load(y_train_path)
x_validate = np.load(x_val_path)
y_validate = np.load(y_val_path)

# --- Ensure NHWC shape and dtype ---
expected_feat_len = IMG_HEIGHT * IMG_WIDTH * CHANNELS


def to_nhwc(x):
    if x.ndim == 2:
        if x.shape[1] != expected_feat_len:
            raise ValueError(
                f"Feature length {x.shape[1]} != expected {expected_feat_len}"
            )
        x = x.reshape((-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    if x.ndim != 4 or x.shape[1:] != (IMG_HEIGHT, IMG_WIDTH, CHANNELS):
        raise ValueError(
            f"Bad tensor shape: got {x.shape}, expected (N,{IMG_HEIGHT},{IMG_WIDTH},{CHANNELS})"
        )
    return x.astype(np.float32, copy=False)


x_train = to_nhwc(x_train)
x_validate = to_nhwc(x_validate)

# --- Sanity checks BEFORE argmax ---
if y_train.ndim == 2 and y_validate.ndim == 2:
    if y_train.shape[1] != y_validate.shape[1]:
        raise ValueError(
            f"Label-space mismatch: train one-hot width={y_train.shape[1]} "
            f"vs val width={y_validate.shape[1]}. Recalculate features for ALL data."
        )
    NUM_CLASSES = y_train.shape[1]
elif y_train.ndim == 1 and y_validate.ndim == 1:
    NUM_CLASSES = int(max(y_train.max(), y_validate.max()) + 1)
else:
    raise ValueError(
        f"Inconsistent label dimensions: train={y_train.ndim}D, val={y_validate.ndim}D"
    )

# Convert to sparse only AFTER the width check
if y_train.ndim == 2:
    y_train = y_train.argmax(axis=1)
if y_validate.ndim == 2:
    y_validate = y_validate.argmax(axis=1)

# final diagnostics
print(f"x_train: {x_train.shape}, x_val: {x_validate.shape}")
print(f"y_train: {y_train.shape}, y_val: {y_validate.shape}, NUM_CLASSES={NUM_CLASSES}")


# --- Model Definition ---
inp = Input(shape=INPUT_SHAPE, name="image_input")
scaled = Rescaling(scale=2.0, offset=-1.0, name="to_minus1_plus1")(inp)

# Load local 160x160 no-top weights
weights_path = os.path.expanduser(
    "~/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5"
)
base_model = MobileNetV2(
    input_shape=INPUT_SHAPE, include_top=False, weights=weights_path
)
base_model.trainable = False

x = base_model(scaled, training=False)
x = GlobalAveragePooling2D(name="gap")(x)
x = Dropout(0.5, name="dropout")(x)
predictions = Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
model = Model(inputs=inp, outputs=predictions)

# --- Compile Model ---
optimizer = Adam(learning_rate=args.learning_rate)
model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.summary()

# --- Train Model ---
print(
    f"Starting training for {args.epochs} epochs with learning rate {args.learning_rate}..."
)
history = model.fit(
    x_train,
    y_train,
    epochs=args.epochs,
    validation_data=(x_validate, y_validate),
    batch_size=32,
    verbose=2,
)
print("Training finished.")

# --- Save Model (SavedModel + TFLite) ---
model_save_path = os.path.join(args.out_directory, "saved_model")
print(f"Saving model to {model_save_path}")
model.save(model_save_path)

# TFLite float32 export at /home/model.tflite (EI profiler expects this exact path)
tflite_path = os.path.join(args.out_directory, "model.tflite")
print(f"Converting to TFLite (float32) -> {tflite_path}")
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Ensure pure TFLite ops
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    # Keep float32 (no quantization)
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print("TFLite model written.")
except Exception as e:
    print(f"Primary TFLite conversion failed: {e}")
    print("Retrying from SavedModel directory...")
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(model_save_path)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print("TFLite model written from SavedModel.")
    except Exception as e2:
        print(f"Fallback TFLite conversion failed: {e2}")
        sys.exit(1)

# Save training history
history_save_path = os.path.join(args.out_directory, "training_history.json")
with open(history_save_path, "w") as f:
    json.dump(history.history, f)
print("Model and history saved.")
