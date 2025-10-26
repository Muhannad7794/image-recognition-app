# learning-block/model.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
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

# --- DEFINE IMAGE SHAPE ---
# We know this from our DSP block's parameters.json
IMG_HEIGHT = 96
IMG_WIDTH = 96
# We will reshape to 1 channel first, then convert to 3
CHANNELS_IN = 1
CHANNELS_OUT = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS_OUT)  # MobileNetV2 needs 3 channels

# --- Load Data using NumPy ---
print(f"Loading data from directory: {args.data_directory}")
try:
    # --- FIX for data file names ---
    # The "Generate features" step creates "X_split_test.npy"
    # But the training job renames it to "X_validate_features.npy"
    # We will check for both names.
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
    print(f"\nError: A required .npy file was not found.")
    print(f"Details: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading data from .npy files: {e}")
    sys.exit(1)


# --- Reshape Data and Determine Classes ---
try:
    # --- 3-CHANNEL FIX IS HERE ---

    # Process Training Data
    if len(x_train.shape) == 2:
        print(f"Flattened training features detected. Reshaping to 1 channel...")
        x_train = x_train.reshape((-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS_IN))
        print(f"New x_train shape (1 channel): {x_train.shape}")

    print("Converting 1-channel training data to 3-channel (RGB)...")
    x_train = np.repeat(x_train, CHANNELS_OUT, axis=-1)
    print(f"New x_train shape (3 channel): {x_train.shape}")

    # Process Validation Data
    if len(x_validate.shape) == 2:
        print(f"Flattened validation features detected. Reshaping to 1 channel...")
        x_validate = x_validate.reshape((-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS_IN))
        print(f"New x_validate shape (1 channel): {x_validate.shape}")

    print("Converting 1-channel validation data to 3-channel (RGB)...")
    x_validate = np.repeat(x_validate, CHANNELS_OUT, axis=-1)
    print(f"New x_validate shape (3 channel): {x_validate.shape}")
    # --- END OF 3-CHANNEL FIX ---

    # --- LABEL FIX ---
    # Convert one-hot encoded labels to sparse labels (0-45)
    if len(y_train.shape) == 2:
        print(f"Converting y_train from one-hot (shape {y_train.shape}) to sparse...")
        y_train = np.argmax(y_train, axis=1)
    if len(y_validate.shape) == 2:
        print(
            f"Converting y_validate from one-hot (shape {y_validate.shape}) to sparse..."
        )
        y_validate = np.argmax(y_validate, axis=1)

    # --- FINAL 1-INDEXED FIX ---
    # Check if max label is > (num_classes - 1)
    # This handles the case where labels are 1-46 instead of 0-45
    all_labels_for_check = np.concatenate((y_train, y_validate))
    min_label = np.min(all_labels_for_check)
    max_label = np.max(all_labels_for_check)
    NUM_CLASSES = len(np.unique(all_labels_for_check))

    if max_label >= NUM_CLASSES:
        print(
            f"Warning: Labels appear to be 1-indexed (min: {min_label}, max: {max_label}). Converting to 0-indexed."
        )
        y_train = y_train - 1
        y_validate = y_validate - 1
        # Recalculate class count just in case
        NUM_CLASSES = len(np.unique(np.concatenate((y_train, y_validate))))
    # --- END FINAL FIX ---

    print(f"Final x_train shape: {x_train.shape}")
    print(f"Final y_train shape: {y_train.shape}")
    print(f"Final x_validate shape: {x_validate.shape}")
    print(f"Final y_validate shape: {y_validate.shape}")
    print(f"Final Input shape for model: {INPUT_SHAPE}")
    print(f"Final Number of classes: {NUM_CLASSES}")

    if NUM_CLASSES <= 1:
        print(f"Error: Only found {NUM_CLASSES} class(es). Need at least 2.")
        sys.exit(1)

except Exception as e:
    print(f"Error processing data shapes: {e}")
    traceback.print_exc()
    sys.exit(1)


# --- Model Definition ---
base_model = MobileNetV2(input_shape=INPUT_SHAPE, include_top=False, weights="imagenet")
base_model.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)

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
