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
import os, json, argparse, sys, traceback

# --- Args ---
parser = argparse.ArgumentParser(
    description="Train image classification model (AUGMENT, strict labels)"
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

# --- Load (no fallbacks; fail fast to keep splits consistent) ---
x_train_path = os.path.join(args.data_directory, "X_train_features.npy")
y_train_path = os.path.join(args.data_directory, "y_train.npy")
x_val_path = os.path.join(args.data_directory, "X_validate_features.npy")
y_val_path = os.path.join(args.data_directory, "y_validate.npy")

for p in (x_train_path, y_train_path, x_val_path, y_val_path):
    if not os.path.exists(p):
        print(f"[AUG][FATAL] Missing required file: {p}")
        print(
            "[AUG] Ensure your training job uses the same split that created these files (no older split_test fallbacks)."
        )
        sys.exit(1)

x_train = np.load(x_train_path)
y_train = np.load(y_train_path)
x_validate = np.load(x_val_path)
y_validate = np.load(y_val_path)

print(f"[AUG] x_train (raw): {x_train.shape}, x_validate (raw): {x_validate.shape}")
print(f"[AUG] y_train (raw): {y_train.shape}, y_validate (raw): {y_validate.shape}")

# --- Shapes ---
expected_feat_len = IMG_HEIGHT * IMG_WIDTH * CHANNELS
if x_train.ndim == 2:
    if x_train.shape[1] != expected_feat_len:
        print(
            f"[AUG][FATAL] x_train feature len {x_train.shape[1]} != {expected_feat_len}"
        )
        sys.exit(1)
    x_train = x_train.reshape((-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS))
if x_validate.ndim == 2:
    if x_validate.shape[1] != expected_feat_len:
        print(
            f"[AUG][FATAL] x_validate feature len {x_validate.shape[1]} != {expected_feat_len}"
        )
        sys.exit(1)
    x_validate = x_validate.reshape((-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS))

if x_train.ndim != 4 or x_validate.ndim != 4:
    print(
        f"[AUG][FATAL] Unexpected dims after reshape. Train {x_train.shape}, Val {x_validate.shape}"
    )
    sys.exit(1)
if x_train.shape[1:] != INPUT_SHAPE or x_validate.shape[1:] != INPUT_SHAPE:
    print(
        f"[AUG][FATAL] Data shape != INPUT_SHAPE. Expected {INPUT_SHAPE}, got {x_train.shape[1:]}, {x_validate.shape[1:]}"
    )
    sys.exit(1)

x_train = x_train.astype(np.float32, copy=False)
x_validate = x_validate.astype(np.float32, copy=False)


# --- Labels ---
def to_sparse(y):
    # if one-hot, convert to sparse indices
    return np.argmax(y, axis=1) if y.ndim == 2 else y


y_train = to_sparse(y_train)
y_validate = to_sparse(y_validate)

unique_train = np.unique(y_train)
unique_val = np.unique(y_validate)

NUM_CLASSES_T = unique_train.size
NUM_CLASSES_V = unique_val.size

print(f"[AUG] unique_train: {unique_train[:10]}... (K={NUM_CLASSES_T})")
print(f"[AUG] unique_val:   {unique_val[:10]}... (K={NUM_CLASSES_V})")

# If labels are 1-indexed, shift both
if unique_train.min() == 1 and unique_val.min() == 1:
    print("[AUG] Shifting labels from 1-indexed to 0-indexed.")
    y_train = y_train - 1
    y_validate = y_validate - 1
    unique_train = np.unique(y_train)
    unique_val = np.unique(y_validate)
    NUM_CLASSES_T = unique_train.size
    NUM_CLASSES_V = unique_val.size

# Class count must match between splits
if NUM_CLASSES_T != NUM_CLASSES_V:
    print(
        f"[AUG][FATAL] Class count mismatch: train={NUM_CLASSES_T}, val={NUM_CLASSES_V}."
    )
    print(
        "[AUG] Your current run seems to mix splits with different label spaces (e.g., older split_test vs new validate)."
    )
    sys.exit(1)

NUM_CLASSES = int(NUM_CLASSES_T)


# Histograms (helpful)
def hist(lbls, name):
    h = np.bincount(lbls.astype(int), minlength=max(int(lbls.max()) + 1, NUM_CLASSES))
    print(f"[AUG] Class histogram {name} (len={len(lbls)}): {h}")


hist(y_train, "train")
hist(y_validate, "val")

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

x = augment(inp)
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
predictions = Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
model = Model(inputs=inp, outputs=predictions)

# --- Compile ---
# NOTE: label_smoothing not supported here → use plain sparse CE
loss = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = Adam(learning_rate=args.learning_rate)
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
model.summary()

# --- Callbacks ---
callbacks = [
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
    epochs=args.epochs,
    validation_data=(x_validate, y_validate),
    batch_size=32,
    verbose=2,
    callbacks=callbacks,
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
    try:
        conv = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = conv.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print("[AUG] TFLite model written (from SavedModel).")
    except Exception as e2:
        print(f"[AUG] Fallback TFLite conversion failed: {e2}")
        sys.exit(1)

hist_path = os.path.join(out_dir, "training_history.json")
with open(hist_path, "w") as f:
    json.dump(history.history, f)
print("[AUG] Model and history saved.")
