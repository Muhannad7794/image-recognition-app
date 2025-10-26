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
IMG_HEIGHT = 96
IMG_WIDTH = 96
CHANNELS_IN = 1
CHANNELS_OUT = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS_OUT)  # MobileNetV2 needs 3 channels

# --- Load Data using NumPy ---
print(f"Loading data from directory: {args.data_directory}")
try:
    x_train_path = os.path.join(args.data_directory, "X_train_features.npy")
    y_train_path = os.path.join(args.data_directory, "y_train.npy")

    x_val_path = os.path.join(args.data_directory, "X_validate_features.npy")
    y_val_path = os.path.join(args.data_directory, "y_validate.npy")

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
    # --- 3-CHANNEL FIX ---
    # Process Training Data
    if len(x_train.shape) == 2:
        print(f"Flattened training features detected. Reshaping to 1 channel...")
        x_train = x_train.reshape((-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS_IN))
    print("Converting 1-channel training data to 3-channel (RGB)...")
    x_train = np.repeat(x_train, CHANNELS_OUT, axis=-1)

    # Process Validation Data
    if len(x_validate.shape) == 2:
        print(f"Flattened validation features detected. Reshaping to 1 channel...")
        x_validate = x_validate.reshape((-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS_IN))
    print("Converting 1-channel validation data to 3-channel (RGB)...")
    x_validate = np.repeat(x_validate, CHANNELS_OUT, axis=-1)
    # --- END OF 3-CHANNEL FIX ---

    # --- LABEL FIX ---
    # The labels are already one-hot encoded (e.g., shape (2472, 46)).
    # We can get the number of classes directly from the shape.
    NUM_CLASSES = y_train.shape[1]

    # Verify the label shapes match
    if y_train.shape[1] != y_validate.shape[1]:
        print(
            f"Error: Mismatch in number of classes between train ({y_train.shape[1]}) and validate ({y_validate.shape[1]})"
        )
        sys.exit(1)
    # --- END LABEL FIX ---

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

# --- LOSS FUNCTION FIX ---
# Use "categorical_crossentropy" because our labels are one-hot encoded
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)
# --- END LOSS FUNCTION FIX ---

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
