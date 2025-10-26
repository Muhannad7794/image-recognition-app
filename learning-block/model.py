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

# --- DEFINE IMAGE SHAPE ---
# We know this from our DSP block's parameters.json
IMG_HEIGHT = 96
IMG_WIDTH = 96
# The log `x_validate shape: (495, 96, 96, 1)` implies 1 channel.
# And 96 * 96 * 1 = 9216, which matches x_train's flat features.
CHANNELS = 1
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

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

except FileNotFoundError as e:
    print(f"\nError: A required .npy file was not found.")
    print(f"Details: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading data from .npy files: {e}")
    sys.exit(1)


# --- Reshape Data and Determine Classes ---
try:
    # --- THIS IS THE FIX ---
    # Check if x_train is flat and reshape it
    if len(x_train.shape) == 2:
        print(f"Flattened training features detected (Shape: {x_train.shape}). Reshaping to {(-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS)}...")
        x_train = x_train.reshape((-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS))
        print(f"New x_train shape: {x_train.shape}")

    # Check if x_validate is flat and reshape it
    # (The log says it's 3D, but it *should* be flat if the DSP ran correctly)
    if len(x_validate.shape) == 2:
        print(f"Flattened validation features detected (Shape: {x_validate.shape}). Reshaping to {(-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS)}...")
        x_validate = x_validate.reshape((-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS))
        print(f"New x_validate shape: {x_validate.shape}")
    # --- END OF FIX ---

    # Verify shapes
    if len(x_train.shape) != 4 or len(x_validate.shape) != 4:
         print(f"Error: Mismatched data dimensions after reshape.")
         print(f"x_train shape: {x_train.shape}")
         print(f"x_validate shape: {x_validate.shape}")
         sys.exit(1)

    # --- FIX FOR LABELS ---
    # The log shows y_validate has 46 classes, so we must trust that.
    # Your y_train (shape 4) is likely wrong or from a different dataset.
    # We will get the number of classes from the VALIDATION set.
    
    # We also must use sparse labels for "sparse_categorical_crossentropy"
    # The labels should be (samples,) not (samples, classes)
    if len(y_train.shape) == 2:
        print(f"Warning: y_train is one-hot encoded (shape {y_train.shape}). Converting to sparse labels.")
        y_train = np.argmax(y_train, axis=1)
        print(f"New y_train shape: {y_train.shape}")

    if len(y_validate.shape) == 2:
        print(f"Warning: y_validate is one-hot encoded (shape {y_validate.shape}). Converting to sparse labels.")
        y_validate = np.argmax(y_validate, axis=1)
        print(f"New y_validate shape: {y_validate.shape}")

    # Get class count from the set with all classes
    NUM_CLASSES = len(np.unique(y_validate))
    if NUM_CLASSES <= 1:
        print(f"Error: Only found {NUM_CLASSES} class(es). Need at least 2.")
        sys.exit(1)
    
    print(f"Input shape: {INPUT_SHAPE}")
    print(f"Number of classes: {NUM_CLASSES}")
    
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