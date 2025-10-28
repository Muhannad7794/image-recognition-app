# learning-block/finetune/model.py  (96x96, robust loading + full fine-tune)
import os, json, argparse, sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# ---------------- Args ----------------
parser = argparse.ArgumentParser(
    description="Fine-tune MobileNetV2 @96x96 (robust loader)"
)
parser.add_argument("--data-directory", type=str, required=True)
parser.add_argument("--out-directory", type=str, required=True)

# Optional CLI fallbacks (used only if parameters.json doesn't provide them)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--learning-rate", type=float, default=1e-4)  # warmup LR
parser.add_argument("--warmup-epochs", type=int, default=6)
parser.add_argument("--fine-tune-start-lr", type=float, default=1e-4)  # cosine start LR
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--label-smoothing", type=float, default=0.1)
parser.add_argument("--early-stopping-patience", type=int, default=8)
parser.add_argument("--unfreeze-pct", type=float, default=0.7)  # top % to unfreeze
parser.add_argument("--mixup-alpha", type=float, default=0.0)  # 0 disables
parser.add_argument("--cutmix-alpha", type=float, default=0.0)  # 0 disables
args, _ = parser.parse_known_args()

IMG_HEIGHT = 96
IMG_WIDTH = 96
CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

# Use the arguments from the UI
WARMUP_EPOCHS = args.warmup_epochs
LABEL_SMOOTH = args.label_smoothing
BATCH_SIZE = args.batch_size
FT_START_LR = args.fine_tune_start_lr
print(f"[FT] Data dir: {args.data_directory}")


# ---------------- File resolution (prefer split-aware) ----------------
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

print("[FT] Dir listing:", sorted(os.listdir(args.data_directory)))
print(
    f"[FT] Train X: {os.path.basename(x_train_path)} | Train y: {os.path.basename(y_train_path)}"
)
print(
    f"[FT] Val   X: {os.path.basename(x_val_path)}   | Val   y: {os.path.basename(y_val_path)}"
)

# ---------------- Load ----------------
x_train = np.load(x_train_path)
y_train = np.load(y_train_path)
x_val = np.load(x_val_path)
y_val = np.load(y_val_path)

print(f"[FT] x_train (raw): {x_train.shape}, x_val (raw): {x_val.shape}")
print(f"[FT] y_train (raw): {y_train.shape}, y_val (raw): {y_val.shape}")

# ---------------- Ensure NHWC (N,96,96,3) ----------------
expected_feat_len = IMG_HEIGHT * IMG_WIDTH * CHANNELS


def _to_nhwc(x: np.ndarray, name: str) -> np.ndarray:
    if x.ndim == 2:
        if x.shape[1] != expected_feat_len:
            raise ValueError(
                f"[FT][FATAL] {name} feature len {x.shape[1]} != {expected_feat_len}"
            )
        x = x.reshape((-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    if x.ndim != 4 or x.shape[1:] != (IMG_HEIGHT, IMG_WIDTH, CHANNELS):
        raise ValueError(
            f"[FT][FATAL] Bad tensor shape for {name}: {x.shape}, expected (N,{IMG_HEIGHT},{IMG_WIDTH},{CHANNELS})"
        )
    return x.astype(np.float32, copy=False)


x_train = _to_nhwc(x_train, "x_train")
x_val = _to_nhwc(x_val, "x_val")

# ---------------- Label-space checks BEFORE argmax ----------------
if y_train.ndim == 2 and y_val.ndim == 2:
    if y_train.shape[1] != y_val.shape[1]:
        raise ValueError(
            f"[FT][FATAL] Label-space mismatch: train one-hot width={y_train.shape[1]} "
            f"vs val width={y_val.shape[1]}. Recalculate features for ALL data."
        )
    NUM_CLASSES = int(y_train.shape[1])
    y_train = y_train.argmax(axis=1)
    y_val = y_val.argmax(axis=1)
elif y_train.ndim == 1 and y_val.ndim == 1:
    NUM_CLASSES = int(max(y_train.max(), y_val.max()) + 1)
else:
    raise ValueError(
        f"[FT][FATAL] Inconsistent label dims: train={y_train.ndim}D, val={y_val.ndim}D"
    )

# Optional: shift if both sets look 1-indexed
if y_train.min() == 1 and y_val.min() == 1:
    print("[FT] Shifting labels 1-indexed → 0-indexed.")
    y_train -= 1
    y_val -= 1

# Sanity on ranges
if (y_train.min() < 0) or (y_val.min() < 0):
    raise ValueError("[FT][FATAL] Negative class index found.")
if (y_train.max() >= NUM_CLASSES) or (y_val.max() >= NUM_CLASSES):
    raise ValueError(
        f"[FT][FATAL] Class index out of range w.r.t NUM_CLASSES={NUM_CLASSES}"
    )

# Class weights (balanced)
class_counts = np.bincount(y_train.astype(int), minlength=NUM_CLASSES)
inv_freq = class_counts.max() / np.maximum(class_counts, 1)
class_weight = {i: float(inv_freq[i]) for i in range(NUM_CLASSES)}

print(f"[FT] INPUT_SHAPE={INPUT_SHAPE}, NUM_CLASSES={NUM_CLASSES}")
print(f"[FT] Class counts (first 20): {class_counts[:20]}")


# ---------------- tf.data (one-hot for label smoothing) ----------------
def to_one_hot(y):
    return tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES)


augment = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.10, 0.10),
        layers.RandomContrast(0.20),
    ],
    name="augment",
)


def make_ds(x, y, training=True):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(2048, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda xi, yi: (xi, to_one_hot(yi)), num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)  # <-- NOW USES THE VARIABLE
    return ds


train_ds = make_ds(x_train, y_train, training=True)
val_ds = make_ds(x_val, y_val, training=False)

# ---------------- Model ----------------
inputs = layers.Input(shape=INPUT_SHAPE, name="image_input")
x = augment(inputs)  # active only in training
x = layers.Rescaling(scale=2.0, offset=-1.0)(x)  # [-1, 1] (matches your baseline)

# Prefer local 96x96 no-top weights if present; else "imagenet"
weights_path = os.path.expanduser(
    "~/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5"
)
if os.path.exists(weights_path):
    base = keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE, include_top=False, weights=weights_path
    )
else:
    base = keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE, include_top=False, weights="imagenet"
    )

# Warmup: freeze, then unfreeze ALL for fine-tune
base.trainable = False

x = base(x, training=False)
x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.Dropout(0.3, name="dropout")(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
model = keras.Model(inputs, outputs)

# Loss & metrics
loss = keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH)
metrics = [
    keras.metrics.CategoricalAccuracy(name="acc"),
    keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
]

# ---------------- Phase 1: Warmup (frozen backbone) ----------------
print(f"[FT] Warmup {WARMUP_EPOCHS} epochs @ lr={args.learning_rate}")
model.compile(
    optimizer=keras.optimizers.Adam(args.learning_rate), loss=loss, metrics=metrics
)

es = EarlyStopping(
    monitor="val_loss",
    patience=args.early_stopping_patience,
    restore_best_weights=True,
    verbose=1,
)

hist_warm = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=WARMUP_EPOCHS,
    class_weight=class_weight,
    callbacks=[es],
    verbose=2,
)

# ---------------- Phase 2: Full fine-tune (unfreeze all) ----------------
for layer in base.layers:
    layer.trainable = True

steps_per_epoch = int(np.ceil(len(x_train) / BATCH_SIZE))
decay_steps = steps_per_epoch * max(args.epochs - WARMUP_EPOCHS, 1)
cosine_lr = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=FT_START_LR, decay_steps=decay_steps
)

print(
    f"[FT] Finetune up to epoch {args.epochs} with cosine LR (start {FT_START_LR}), backbone unfrozen"
)
model.compile(optimizer=keras.optimizers.Adam(cosine_lr), loss=loss, metrics=metrics)

hist_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=len(hist_warm.epoch),
    epochs=args.epochs,
    class_weight=class_weight,
    callbacks=[es],
    verbose=2,
)

# ---------------- Save (SavedModel + TFLite) ----------------
out_dir = args.out_directory
saved_model_dir = os.path.join(out_dir, "saved_model")
print(f"[FT] Saving SavedModel -> {saved_model_dir}")
model.save(saved_model_dir)

tflite_path = os.path.join(out_dir, "model.tflite")
print(f"[FT] Converting to TFLite (float32) -> {tflite_path}")
try:
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = conv.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print("[FT] TFLite model written.")
except Exception as e:
    print(f"[FT] Primary TFLite conversion failed: {e}")
    print("[FT] Retrying from SavedModel ...")
    conv = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = conv.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print("[FT] TFLite model written (from SavedModel).")

# Save merged history
hist_path = os.path.join(out_dir, "training_history.json")
with open(hist_path, "w") as f:
    merged = {}
    for k, v in hist_warm.history.items():
        merged[k] = list(v)
    for k, v in hist_ft.history.items():
        merged.setdefault(k, [])
        merged[k].extend(v)
    json.dump(merged, f, default=lambda o: float(o))
print("[FT] Model and merged history saved.")
