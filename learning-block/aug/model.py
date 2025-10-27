### learning-block/aug/model.py  new run01
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


# --- Helper: pick first existing pair (validate, then split_test) ---
def _pick_paths(root: str):
    candidates = [
        (
            "X_train_features.npy",
            "y_train.npy",
            "X_validate_features.npy",
            "y_validate.npy",
        ),
        ("X_train_features.npy", "y_train.npy", "X_split_test.npy", "Y_split_test.npy"),
    ]
    for xt, yt, xv, yv in candidates:
        xp = os.path.join(root, xt)
        yp = os.path.join(root, yt)
        xvp = os.path.join(root, xv)
        yvp = os.path.join(root, yv)
        if all(os.path.exists(p) for p in (xp, yp, xvp, yvp)):
            src = "validate" if "validate" in xv else "split_test"
            return (xp, yp, xvp, yvp, src)
    # If we get here, print what’s missing
    print("[AUG][FATAL] Could not find a matching train/val file set.")
    print(f"  Tried:")
    for xt, yt, xv, yv in candidates:
        print(f"   - {xt}, {yt}, {xv}, {yv}")
    sys.exit(1)


x_train_path, y_train_path, x_val_path, y_val_path, val_src = _pick_paths(
    args.data_directory
)
print(
    f"[AUG] Using validation source: {val_src} ({os.path.basename(x_val_path)}, {os.path.basename(y_val_path)})"
)

# --- Load ---
x_train = np.load(x_train_path)
y_train = np.load(y_train_path)
x_val = np.load(x_val_path)
y_val = np.load(y_val_path)

print(f"[AUG] x_train (raw): {x_train.shape}, x_val (raw): {x_val.shape}")
print(f"[AUG] y_train (raw): {y_train.shape}, y_val (raw): {y_val.shape}")

# --- Shapes ---
expected_feat_len = IMG_HEIGHT * IMG_WIDTH * CHANNELS


def _ensure_4d(x, name):
    if x.ndim == 2:
        if x.shape[1] != expected_feat_len:
            print(
                f"[AUG][FATAL] {name} feature len {x.shape[1]} != {expected_feat_len}"
            )
            sys.exit(1)
        return x.reshape((-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    return x


x_train = _ensure_4d(x_train, "x_train")
x_val = _ensure_4d(x_val, "x_val")

if x_train.ndim != 4 or x_val.ndim != 4:
    print(
        f"[AUG][FATAL] Unexpected dims after reshape. Train {x_train.shape}, Val {x_val.shape}"
    )
    sys.exit(1)
if x_train.shape[1:] != INPUT_SHAPE or x_val.shape[1:] != INPUT_SHAPE:
    print(
        f"[AUG][FATAL] Data shape != INPUT_SHAPE. Expected {INPUT_SHAPE}, got {x_train.shape[1:]}, {x_val.shape[1:]}"
    )
    sys.exit(1)

x_train = x_train.astype(np.float32, copy=False)
x_val = x_val.astype(np.float32, copy=False)


# --- Labels: one-hot -> sparse, 1-indexed -> 0-indexed, unify spaces if needed ---
def to_sparse(y):
    return np.argmax(y, axis=1) if y.ndim == 2 else y


y_train = to_sparse(y_train)
y_val = to_sparse(y_val)

# Shift 1-indexed jointly
if (y_train.min() == 1) and (y_val.min() == 1):
    print("[AUG] Shifting labels from 1-indexed to 0-indexed.")
    y_train = y_train - 1
    y_val = y_val - 1

ut, uv = np.unique(y_train), np.unique(y_val)
kt, kv = ut.size, uv.size
print(f"[AUG] Train classes={kt} {ut[:10]}..., Val classes={kv} {uv[:10]}...")

# If class index sets differ, remap both to union (prevents crashes; warns loudly)
union = np.unique(np.concatenate([ut, uv]))
if not np.array_equal(ut, uv):
    print(
        f"[AUG][WARN] Train/Val class sets differ. Unifying via union mapping. "
        f"(Train K={kt}, Val K={kv}, Union K={union.size})"
    )
    remap = {c: i for i, c in enumerate(union)}
    y_train = np.vectorize(remap.get)(y_train)
    y_val = np.vectorize(remap.get)(y_val)

NUM_CLASSES = int(np.unique(np.concatenate([y_train, y_val])).size)


def _hist(lbls, name):
    h = np.bincount(lbls.astype(int), minlength=NUM_CLASSES)
    print(f"[AUG] Class histogram {name} (n={len(lbls)}): {h}")


_hist(y_train, "train")
_hist(y_val, "val")
print(f"[AUG] Final INPUT_SHAPE={INPUT_SHAPE}, NUM_CLASSES={NUM_CLASSES}")

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

# Save history
hist_path = os.path.join(out_dir, "training_history.json")
with open(hist_path, "w") as f:
    json.dump(history.history, f)
print("[AUG] Model and history saved.")
