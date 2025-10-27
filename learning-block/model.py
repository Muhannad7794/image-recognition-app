# learning-block/model.py
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

# --- Define image shape (must match DSP out_channels/size) ---
IMG_HEIGHT = 96
IMG_WIDTH = 96
CHANNELS = 3  # <- keep in sync with parameters.json: out_channels = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

# --- Load Data using NumPy ---
print(f"Loading data from directory: {args.data_directory}")
try:
    x_train_path = os.path.join(args.data_directory, "X_train_features.npy")
    y_train_path = os.path.join(args.data_directory, "y_train.npy")

    x_val_path = os.path.join(args.data_directory, "X_validate_features.npy")
    y_val_path = os.path.join(args.data_directory, "y_validate.npy")

    # Fallback for old file names
    if not os.path.exists(x_val_path):
        print(
            "Warning: X_validate_features.npy not found, falling back to X_split_test.npy"
        )
        x_val_path = os.path.join(args.data_directory, "X_split_test.npy")
    if not os.path.exists(y_val_path):
        print("Warning: y_validate.npy not found, falling back to Y_split_test.npy")
        y_val_path = os.path.join(args.data_directory, "Y_split_test.npy")

    print(f"Loading training features: {x_train_path}")
    x_train = np.load(x_train_path)

    print(f"Loading training labels: {y_train_path}")
    y_train = np.load(y_train_path)

    print(f"Loading validation features: {x_val_path}")
    x_validate = np.load(x_val_path)

    print(f"Loading validation labels: {y_val_path}")
    y_validate = np.load(y_val_path)

    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"\nError: A required .npy file was not found.\nDetails: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading data from .npy files: {e}")
    sys.exit(1)

# --- Process Data and Determine Classes ---
try:
    print(f"Loaded x_train shape: {x_train.shape}")
    print(f"Loaded x_validate shape: {x_validate.shape}")

    expected_feat_len = IMG_HEIGHT * IMG_WIDTH * CHANNELS

    # If flat (N, F), reshape to (N, H, W, C)
    if x_train.ndim == 2:
        if x_train.shape[1] != expected_feat_len:
            print(
                f"Error: x_train feature length {x_train.shape[1]} != expected {expected_feat_len}"
            )
            sys.exit(1)
        x_train = x_train.reshape((-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    if x_validate.ndim == 2:
        if x_validate.shape[1] != expected_feat_len:
            print(
                f"Error: x_validate feature length {x_validate.shape[1]} != expected {expected_feat_len}"
            )
            sys.exit(1)
        x_validate = x_validate.reshape((-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS))

    # If already 4D, verify shape
    if x_train.ndim != 4 or x_validate.ndim != 4:
        print("Error: Unexpected data dimensions after reshape attempt.")
        print(f"x_train shape: {x_train.shape}")
        print(f"x_validate shape: {x_validate.shape}")
        sys.exit(1)
    if x_train.shape[1:] != INPUT_SHAPE or x_validate.shape[1:] != INPUT_SHAPE:
        print("Error: Data shape does not match expected INPUT_SHAPE.")
        print(
            f"Expected: {INPUT_SHAPE}, got x_train: {x_train.shape[1:]}, x_validate: {x_validate.shape[1:]}"
        )
        sys.exit(1)

    # Ensure dtype float32 (DSP already outputs [0,1], keep that)
    x_train = x_train.astype(np.float32, copy=False)
    x_validate = x_validate.astype(np.float32, copy=False)

    # --- LABEL FIX (Keep this part) ---
    if y_train.ndim == 2:
        print(f"Converting y_train from one-hot (shape {y_train.shape}) to sparse...")
        y_train = np.argmax(y_train, axis=1)
    if y_validate.ndim == 2:
        print(
            f"Converting y_validate from one-hot (shape {y_validate.shape}) to sparse..."
        )
        y_validate = np.argmax(y_validate, axis=1)

    all_labels_for_check = np.concatenate((y_train, y_validate))
    min_label = np.min(all_labels_for_check)
    max_label = np.max(all_labels_for_check)
    NUM_CLASSES = len(np.unique(all_labels_for_check))

    if max_label >= NUM_CLASSES:
        print("Warning: Labels appear to be 1-indexed. Converting to 0-indexed.")
        y_train = y_train - 1
        y_validate = y_validate - 1
        NUM_CLASSES = len(np.unique(np.concatenate((y_train, y_validate))))

    print(f"Final x_train shape: {x_train.shape}")
    print(f"Final y_train shape: {y_train.shape}")
    print(f"Final x_validate shape: {x_validate.shape}")
    print(f"Final y_validate shape: {y_validate.shape}")
    print(f"Final Input shape for model: {INPUT_SHAPE}")
    print(f"Final Number of classes: {NUM_CLASSES}")

except Exception as e:
    print(f"Error processing data shapes: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- Model Definition ---
# Input layer to keep things explicit
inp = Input(shape=INPUT_SHAPE, name="image_input")

# Scale [0,1] -> [-1,1] for MobileNetV2
scaled = Rescaling(scale=2.0, offset=-1.0, name="to_minus1_plus1")(inp)

# Base model (imagenet weights expect 3-channel RGB and [-1,1] range)
weights_path = os.path.expanduser("~/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5")
base_model = MobileNetV2(input_shape=INPUT_SHAPE, include_top=False, weights=weights_path)
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

# --- Save Model ---
model_save_path = os.path.join(args.out_directory, "saved_model")
print(f"Saving model to {model_save_path}")
model.save(model_save_path)

# Save history (optional)
history_save_path = os.path.join(args.out_directory, "training_history.json")
with open(history_save_path, "w") as f:
    json.dump(history.history, f)

print("Model and history saved.")
