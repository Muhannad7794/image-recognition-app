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

# --- Script Arguments ---
parser = argparse.ArgumentParser(description="Train image classification model")
parser.add_argument("--data-directory", type=str, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--learning-rate", type=float, required=True)
parser.add_argument("--out-directory", type=str, required=True)

args = parser.parse_args()

# --- Load Data Info from Edge Impulse ---
try:
    with open(os.path.join(args.data_directory, "X_train_features.json")) as f:
        train_info = json.load(f)
    with open(os.path.join(args.data_directory, "Y_train.json")) as f:
        train_labels_info = json.load(f)
    with open(os.path.join(args.data_directory, "X_validate_features.json")) as f:
        val_info = json.load(f)
    with open(os.path.join(args.data_directory, "Y_validate.json")) as f:
        val_labels_info = json.load(f)
except Exception as e:
    print(f"Error loading data info JSON: {e}")
    sys.exit(1)

# --- Data Loading Function (Provided by Edge Impulse Environment) ---
# This function is available when running inside the EI container
# It efficiently loads the image data prepared by the preprocessing step
try:
    # pylint: disable=undefined-variable
    # Note: 'load_images' is injected by the Edge Impulse environment during runtime.
    (x_train, y_train) = load_images(
        train_info["matrix_files"],
        args.data_directory,
        train_labels_info["labels"],
        image_width=train_info["image_width"],
        image_height=train_info["image_height"],
        image_channels=train_info["image_channels"],
    )
    (x_validate, y_validate) = load_images(
        val_info["matrix_files"],
        args.data_directory,
        val_labels_info["labels"],
        image_width=val_info["image_width"],
        image_height=val_info["image_height"],
        image_channels=val_info["image_channels"],
    )
except NameError:
    print("\nError: Could not find the 'load_images' function.")
    print("This script is designed to run within the Edge Impulse environment.")
    print("You can test locally using 'edge-impulse-blocks runner'.\n")
    sys.exit(1)
except Exception as e:
    print(f"Error loading image data: {e}")
    sys.exit(1)

# --- Model Definition ---
NUM_CLASSES = len(np.unique(y_train))
INPUT_SHAPE = (
    train_info["image_width"],
    train_info["image_height"],
    train_info["image_channels"],
)

print(f"Input shape: {INPUT_SHAPE}")
print(f"Number of classes: {NUM_CLASSES}")

# Load MobileNetV2 base model, pre-trained on ImageNet, without the top classification layer
base_model = MobileNetV2(input_shape=INPUT_SHAPE, include_top=False, weights="imagenet")

# Freeze the layers of the base model (we won't re-train them)
base_model.trainable = False

# Add custom layers on top for our specific task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Add dropout for regularization
predictions = Dense(NUM_CLASSES, activation="softmax")(x)

# Combine the base model and our custom layers
model = Model(inputs=base_model.input, outputs=predictions)

# --- Compile Model ---
optimizer = Adam(learning_rate=args.learning_rate)
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",  # Use sparse for integer labels
    metrics=["accuracy"],
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
    batch_size=32,  # A common batch size
    verbose=2,  # Show progress per epoch
)

print("Training finished.")

# --- Save Model ---
# Edge Impulse expects the model saved in TensorFlow SavedModel format
model_save_path = os.path.join(args.out_directory, "saved_model")
print(f"Saving model to {model_save_path}")
model.save(model_save_path)

# Also save the history for potential analysis (optional)
history_save_path = os.path.join(args.out_directory, "training_history.json")
with open(history_save_path, "w") as f:
    json.dump(history.history, f)

print("Model and history saved.")
