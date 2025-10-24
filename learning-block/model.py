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


args, unknown = parser.parse_known_args()

# --- Load Data using NumPy ---
print(f"Loading data from directory: {args.data_directory}")
try:
    x_train_path = os.path.join(args.data_directory, "X_train_features.npy")
    y_train_path = os.path.join(
        args.data_directory, "y_train.npy"
    )  
    x_val_path = os.path.join(
        args.data_directory, "X_split_test.npy"
    )  
    y_val_path = os.path.join(
        args.data_directory, "Y_split_test.npy"
    )

    print(f"Loading training features: {x_train_path}")
    x_train = np.load(x_train_path)

    print(f"Loading training labels: {y_train_path}")
    y_train = np.load(y_train_path)

    print(f"Loading validation features: {x_val_path}")
    x_validate = np.load(x_val_path)

    print(f"Loading validation labels: {y_val_path}")
    y_validate = np.load(y_val_path)

    print("Data loaded successfully.")
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_validate shape: {x_validate.shape}")
    print(f"y_validate shape: {y_validate.shape}")


except FileNotFoundError as e:
    print(f"\nError: A required .npy file was not found.")
    print(f"Ensure feature generation completed and produced expected files:")
    print(f"- X_train_features.npy")
    print(f"- Y_train.npy")
    print(f"- X_validate_features.npy")
    print(f"- Y_validate.npy")
    print(f"Files found in {args.data_directory}:")
    try:
        for filename in os.listdir(args.data_directory):
            print(f"  - {filename}")
    except Exception as list_e:
        print(f"    Error listing directory: {list_e}")
    print(f"\nDetails: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading data from .npy files: {e}")
    sys.exit(1)


# --- Determine Input Shape and Classes from Loaded Data ---
try:
    # The input shape for the model might need adjustment if features are flattened
    # Check the shape of x_train. For MobileNetV2, we need (height, width, channels)
    # If x_train is already (num_samples, features), we might need reshaping or a different base model.
    # Let's assume the feature generator produced features suitable for GlobalAveragePooling2D
    # If the second dimension is the number of features (e.g., shape is (N, F)), GAP won't work directly.
    # For now, let's derive a placeholder shape, MobileNetV2 might need feature vector input if features are flat
    if (
        len(x_train.shape) == 2
    ):  # Features might be flattened (Num Samples, Num Features)
        print(
            "Warning: Training features appear flattened. Model assumes features suitable for GlobalAveragePooling2D."
        )
        # If flattened, MobileNetV2 expects specific input dimensions.
        # This example assumes features are NOT flat (Num Samples, Height, Width, Channels). Adjust if needed.
        # INPUT_SHAPE = (x_train.shape[1],) # Example if input should be flat vector
        print(
            "Error: Flattened features detected, but model expects image-like input. Adjust model architecture or feature generation."
        )
        sys.exit(1)
    elif len(x_train.shape) == 4:  # Expected shape (N, H, W, C)
        INPUT_SHAPE = x_train.shape[1:]  # Get (Height, Width, Channels)
    else:
        print(f"Error: Unexpected x_train shape: {x_train.shape}")
        sys.exit(1)

    NUM_CLASSES = len(np.unique(y_train))
    if NUM_CLASSES <= 1:
        print(f"Error: Only found {NUM_CLASSES} class(es) in y_train. Need at least 2.")
        sys.exit(1)

except Exception as e:
    print(f"Error determining input shape or number of classes: {e}")
    sys.exit(1)


print(f"Deduced Input shape: {INPUT_SHAPE}")
print(f"Number of classes: {NUM_CLASSES}")

# --- Model Definition ---
# Load MobileNetV2 base model
# Note: Ensure INPUT_SHAPE matches what MobileNetV2 expects or adjust the model head
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
