# learning-block/aug/model.py
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
import os, json, argparse, sys

# --- Args ---
parser = argparse.ArgumentParser(
    description="Train image classification model (AUGMENT w/ robust loading)"
)
parser.add_argument("--data-directory", type=str, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--learning-rate", type=float, required=True)
parser.add_argument("--out-directory", type=str, required=True)
args, _ = parser.parse_known_args()

IMG_HEIGHT = 96
IMG_WIDTH = 96
CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

print(f"[AUG] Loading from: {args.data_directory}")


# --- Resolve file paths (prefer split-aware pairs for BOTH train & val) ---
def _pick(root: str, names: list[str]) -> str:
    for n in names:
        p = os.path.join(root, n)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of {names} found under {root}")


x_train_path = _pick(args.data_directory, ["X_split_train.npy", "X_train_features.npy"])
y_train_path = _pick(args.data_directory, ["Y_split_train.npy", "y_train.npy"])
x_val_path = _pick(args.data_directory, ["X_split_test.npy", "X_validate_features.npy"])
y_val_path = _pick(args.data_directory, ["Y_split_test.npy", "y_validate.npy"])

print("[AUG] Directory listing:", sorted(os.listdir(args.data_directory)))
print(f"[AUG] Train X: {os.path.basename(x_train_path)}")
print(f"[AUG] Train y: {os.path.basename(y_train_path)}")
print(f"[AUG] Val   X: {os.path.basename(x_val_path)}")
print(f"[AUG] Val   y: {os.path.basename(y_val_path)}")

# --- Load ---
x_train = np.load(x_train_path)
y_train = np.load(y_train_path)
x_val = np.load(x_val_path)
y_val = np.load(y_val_path)

print(f"[AUG] x_train (raw): {x_train.shape}, x_val (raw): {x_val.shape}")
print(f"[AUG] y_train (raw): {y_train.shape}, y_val (raw): {y_val.shape}")

# --- Ensure NHWC (N, 96, 96, 3) and float32 ---
expected_feat_len = IMG_HEIGHT * IMG_WIDTH * CHANNELS


def _to_nhwc(x: np.ndarray, name: str) -> np.ndarray:
    if x.ndim == 2:
        if x.shape[1] != expected_feat_len:
            raise ValueError(
                f"[AUG][FATAL] {name} feature len {x.shape[1]} != {expected_feat_len}"
            )
        x = x.reshape((-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    if x.ndim != 4 or x.shape[1:] != (IMG_HEIGHT, IMG_WIDTH, CHANNELS):
        raise ValueError(
            f"[AUG][FATAL] Bad tensor shape for {name}: {x.shape}, expected (N,{IMG_HEIGHT},{IMG_WIDTH},{CHANNELS})"
        )
    return x.astype(np.float32, copy=False)


x_train = _to_nhwc(x_train, "x_train")
x_val = _to_nhwc(x_val, "x_val")

# --- Labels: enforce identical label spaces BEFORE converting to sparse ---
if y_train.ndim == 2 and y_val.ndim == 2:
    # One-hot width must match exactly across splits
    if y_train.shape[1] != y_val.shape[1]:
        raise ValueError(
            f"[AUG][FATAL] Label-space mismatch: train one-hot width={y_train.shape[1]} "
            f"vs val width={y_val.shape[1]}. Recalculate features for ALL data."
        )
    NUM_CLASSES = int(y_train.shape[1])
    y_train = y_train.argmax(axis=1)
    y_val = y_val.argmax(axis=1)

elif y_train.ndim == 1 and y_val.ndim == 1:
    # Sparse labels: compute NUM_CLASSES and validate ranges
    NUM_CLASSES = int(max(y_train.max(), y_val.max()) + 1)
else:
    raise ValueError(
        f"[AUG][FATAL] Inconsistent label dims: train={y_train.ndim}D, val={y_val.ndim}D"
    )

# Optional: handle joint 1-indexed labels
if y_train.min() == 1 and y_val.min() == 1:
    print("[AUG] Shifting labels from 1-indexed to 0-indexed.")
    y_train = y_train - 1
    y_val = y_val - 1

# Sanity on label ranges
if (y_train.min() < 0) or (y_val.min() < 0):
    raise ValueError("[AUG][FATAL] Negative class index found after preprocessing.")
if (y_train.max() >= NUM_CLASSES) or (y_val.max() >= NUM_CLASSES):
    raise ValueError(
        f"[AUG][FATAL] Class index out of range: max(train)={y_train.max()}, "
        f"max(val)={y_val.max()}, NUM_CLASSES={NUM_CLASSES}"
    )


# Quick histograms (useful diagnostics)
def _hist(lbls: np.ndarray, name: str):
    h = np.bincount(lbls.astype(int), minlength=NUM_CLASSES)
    print(f"[AUG] Class histogram {name} (n={len(lbls)}): {h}")


_hist(y_train, "train")
_hist(y_val, "val")

print(f"[AUG] Final INPUT_SHAPE={INPUT_SHAPE}, NUM_CLASSES={NUM_CLASSES}")
print(f"[AUG] x_train: {x_train.shape}, x_val: {x_val.shape}")
print(f"[AUG] y_train: {y_train.shape}, y_val: {y_val.shape}")

# --- Model ---
inp = Input(shape=INPUT_SHAPE, name="image_input")
augment = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ],
    name="augment",
)

x = augment(inp)  # only active in training
x = Rescaling(2.0, offset=-1.0, name="to_minus1_plus1")(x)

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
preds = Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
model = Model(inputs=inp, outputs=preds)

# --- Compile ---
loss = tf.keras.losses.SparseCategoricalCrossentropy()
opt = Adam(learning_rate=args.learning_rate)
model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
model.summary()

# --- Callbacks ---
cbs = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
    ),
]

# --- Train ---
print(f"[AUG] Training for {args.epochs} epochs, lr={args.learning_rate} ...")
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=args.epochs,
    batch_size=32,
    verbose=2,
    callbacks=cbs,
)
print("[AUG] Training finished.")

# --- Save (SavedModel + TFLite) ---
out_dir = args.out_directory
saved_model_dir = os.path.join(out_dir, "saved_model")
print(f"[AUG] Saving SavedModel -> {saved_model_dir}")
model.save(saved_model_dir)

tflite_path = os.path.join(out_dir, "model.tflite")
print(f"[AUG] Converting to TFLite (float32) -> {tflite_path}")
try:
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = conv.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print("[AUG] TFLite model written.")
except Exception as e:
    print(f"[AUG] Primary TFLite conversion failed: {e}")
    print("[AUG] Retrying from SavedModel ...")
    conv = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = conv.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print("[AUG] TFLite model written (from SavedModel).")

# Save history
hist_path = os.path.join(out_dir, "training_history.json")
with open(hist_path, "w") as f:
    json.dump(history.history, f, default=lambda o: float(o))
print("[AUG] Model and history saved.")
