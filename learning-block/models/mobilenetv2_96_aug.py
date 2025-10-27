# learning-block/models/mobilenetv2_96_aug.py
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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
import json
import argparse
import sys
import traceback

# --- Script Arguments ---
parser = argparse.ArgumentParser(
    description="Train image classification model (AUGMENT)"
)
parser.add_argument("--data-directory", type=str, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--learning-rate", type=float, required=True)
parser.add_argument("--out-directory", type=str, required=True)
args, _ = parser.parse_known_args()

# --- Define image shape (must match DSP out_channels/size) ---
IMG_HEIGHT = 96
IMG_WIDTH = 96
CHANNELS = 3  # keep in sync with DSP out_channels
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

# --- Load Data using NumPy ---
print(f"[AUGMENT] Loading data from: {args.data_directory}")
try:
    x_train_path = os.path.join(args.data_directory, "X_train_features.npy")
    y_train_path = os.path.join(args.data_directory, "y_train.npy")

    x_val_path = os.path.join(args.data_directory, "X_validate_features.npy")
    y_val_path = os.path.join(args.data_directory, "y_validate.npy")

    # Fallback for old file names
    if not os.path.exists(x_val_path):
        print(
            "[AUGMENT] Warning: X_validate_features.npy not found, using X_split_test.npy"
        )
        x_val_path = os.path.join(args.data_directory, "X_split_test.npy")
    if not os.path.exists(y_val_path):
        print("[AUGMENT] Warning: y_validate.npy not found, using Y_split_test.npy")
        y_val_path = os.path.join(args.data_directory, "Y_split_test.npy")

    x_train = np.load(x_train_path)
    y_train = np.load(y_train_path)
    x_validate = np.load(x_val_path)
    y_validate = np.load(y_val_path)
    print("[AUGMENT] Data loaded.")
except Exception as e:
    print(f"[AUGMENT] Error loading data: {e}")
    sys.exit(1)

# --- Process Data and Determine Classes ---
try:
    print(f"[AUGMENT] x_train shape (raw): {x_train.shape}")
    print(f"[AUGMENT] x_validate shape (raw): {x_validate.shape}")

    expected_feat_len = IMG_HEIGHT * IMG_WIDTH * CHANNELS

    # If flat (N, F), reshape to (N, H, W, C)
    if x_train.ndim == 2:
        if x_train.shape[1] != expected_feat_len:
            print(
                f"[AUGMENT] Error: x_train feature length {x_train.shape[1]} != {expected_feat_len}"
            )
            sys.exit(1)
        x_train = x_train.reshape((-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    if x_validate.ndim == 2:
        if x_validate.shape[1] != expected_feat_len:
            print(
                f"[AUGMENT] Error: x_validate feature length {x_validate.shape[1]} != {expected_feat_len}"
            )
            sys.exit(1)
        x_validate = x_validate.reshape((-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS))

    # If already 4D, verify shape
    if x_train.ndim != 4 or x_validate.ndim != 4:
        print("[AUGMENT] Error: unexpected dimensions after reshape.")
        print(f"  x_train: {x_train.shape}, x_validate: {x_validate.shape}")
        sys.exit(1)
    if x_train.shape[1:] != INPUT_SHAPE or x_validate.shape[1:] != INPUT_SHAPE:
        print("[AUGMENT] Error: data shape != INPUT_SHAPE.")
        print(
            f"  Expected: {INPUT_SHAPE}, got train: {x_train.shape[1:]}, val: {x_validate.shape[1:]}"
        )
        sys.exit(1)

    # Ensure dtype float32
    x_train = x_train.astype(np.float32, copy=False)
    x_validate = x_validate.astype(np.float32, copy=False)

    # Labels
    if y_train.ndim == 2:
        print(f"[AUGMENT] Converting y_train from one-hot {y_train.shape} -> sparse")
        y_train = np.argmax(y_train, axis=1)
    if y_validate.ndim == 2:
        print(
            f"[AUGMENT] Converting y_validate from one-hot {y_validate.shape} -> sparse"
        )
        y_validate = np.argmax(y_validate, axis=1)

    all_labels = np.concatenate((y_train, y_validate))
    NUM_CLASSES = int(np.unique(all_labels).size)
    max_label = int(all_labels.max())
    if max_label >= NUM_CLASSES:
        print("[AUGMENT] Labels look 1-indexed; shifting to 0-indexed.")
        y_train = y_train - 1
        y_validate = y_validate - 1
        NUM_CLASSES = int(np.unique(np.concatenate((y_train, y_validate))).size)

    # Helpful diagnostics
    def _hist(lbls, name):
        lbls = lbls.astype(int).tolist()
        hist = np.bincount(lbls, minlength=NUM_CLASSES)
        print(f"[AUGMENT] Class histogram {name} (len={len(lbls)}): {hist}")

    _hist(y_train, "train")
    _hist(y_validate, "val")

    print(f"[AUGMENT] Final Input shape: {INPUT_SHAPE}, NUM_CLASSES: {NUM_CLASSES}")
except Exception as e:
    print(f"[AUGMENT] Error processing data: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- Model Definition ---
inp = Input(shape=INPUT_SHAPE, name="image_input")

# Data augmentation (on-the-fly)
augment = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ],
    name="augment",
)

x = augment(inp)  # augment only on training
x = Rescaling(2.0, offset=-1.0, name="to_minus1_plus1")(x)  # [0,1] -> [-1,1]

# Load local 96x96 no-top weights
weights_path = os.path.expanduser(
    "~/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5"
)
base_model = MobileNetV2(
    input_shape=INPUT_SHAPE, include_top=False, weights=weights_path
)
base_model.trainable = False

x = base_model(x, training=False)
x = GlobalAveragePooling2D(name="gap")(x)
x = Dropout(0.5, name="dropout")(x)
predictions = Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
model = Model(inputs=inp, outputs=predictions)

# --- Compile Model ---
# Use label smoothing to reduce overconfidence and help with collapse
loss = tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.1)
optimizer = Adam(learning_rate=args.learning_rate)
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
model.summary()

# --- Callbacks for stability ---
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
    ),
]

# --- Train Model ---
print(f"[AUGMENT] Training for {args.epochs} epochs, lr={args.learning_rate} ...")
history = model.fit(
    x_train,
    y_train,
    epochs=args.epochs,
    validation_data=(x_validate, y_validate),
    batch_size=32,
    verbose=2,
    callbacks=callbacks,
)
print("[AUGMENT] Training finished.")

# --- Save Model (SavedModel + TFLite) ---
model_save_path = os.path.join(args.out_directory, "saved_model")
print(f"[AUGMENT] Saving SavedModel -> {model_save_path}")
model.save(model_save_path)

tflite_path = os.path.join(args.out_directory, "model.tflite")
print(f"[AUGMENT] Converting to TFLite (float32) -> {tflite_path}")
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print("[AUGMENT] TFLite model written.")
except Exception as e:
    print(f"[AUGMENT] Primary TFLite conversion failed: {e}")
    print("[AUGMENT] Retrying from SavedModel ...")
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(model_save_path)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print("[AUGMENT] TFLite model written (from SavedModel).")
    except Exception as e2:
        print(f"[AUGMENT] Fallback TFLite conversion failed: {e2}")
        sys.exit(1)

# Save training history
history_save_path = os.path.join(args.out_directory, "training_history.json")
with open(history_save_path, "w") as f:
    json.dump(history.history, f)
print("[AUGMENT] Model and history saved.")
