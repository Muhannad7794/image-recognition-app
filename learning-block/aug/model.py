# learning-block/aug/model.py
# MobileNetV2 (96x96), robust loading, params from JSON (with CLI fallback),
# conservative on-model aug, optional MixUp/CutMix, clean saving for EI.

import os, json, argparse, sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# ---------------- Args (JSON is the source of truth; CLI is fallback) ----------------
parser = argparse.ArgumentParser(description="AUG MobileNetV2 @96x96 (robust loader)")
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

IMG_H, IMG_W, C = 96, 96, 3
INPUT_SHAPE = (IMG_H, IMG_W, C)
EXPECTED_FEAT_LEN = IMG_H * IMG_W * C


# ---------------- Parameters.json loader (prefer JSON, fallback to CLI) ----------------
def _load_params():
    # typical locations: in data dir or cwd
    candidates = [
        os.path.join(args.data_directory, "parameters.json"),
        os.path.join(os.getcwd(), "parameters.json"),
    ]
    params = {}
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r") as f:
                try:
                    params = json.load(f) or {}
                    print(f"[AUG] Loaded parameters from {p}")
                    break
                except Exception as e:
                    print(f"[AUG][WARN] Could not parse {p}: {e}")

    # helper: support both snake_case and kebab-case keys
    def get(name, default):
        if name in params:
            return params[name]
        kebab = name.replace("_", "-")
        if kebab in params:
            return params[kebab]
        return default

    # Compose effective hyperparams
    hp = {
        "epochs": int(get("epochs", args.epochs)),
        "learning_rate": float(get("learning_rate", args.learning_rate)),
        "warmup_epochs": int(get("warmup_epochs", args.warmup_epochs)),
        "fine_tune_start_lr": float(get("fine_tune_start_lr", args.fine_tune_start_lr)),
        "batch_size": int(get("batch_size", args.batch_size)),
        "label_smoothing": float(get("label_smoothing", args.label_smoothing)),
        "early_stopping_patience": int(
            get("early_stopping_patience", args.early_stopping_patience)
        ),
        "unfreeze_pct": float(get("unfreeze_pct", args.unfreeze_pct)),
        "mixup_alpha": float(get("mixup_alpha", args.mixup_alpha)),
        "cutmix_alpha": float(get("cutmix_alpha", args.cutmix_alpha)),
    }
    print("[AUG] Effective hyperparameters:", hp)
    return hp


HP = _load_params()


# ---------------- File resolution (prefer split-aware) ----------------
def _pick(root: str, names: list[str]) -> str:
    for n in names:
        p = os.path.join(root, n)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of {names} found under {root}")


# ---------------- Load data ----------------
print(f"[AUG] Data dir: {args.data_directory}")
x_train_path = _pick(args.data_directory, ["X_split_train.npy", "X_train_features.npy"])
y_train_path = _pick(args.data_directory, ["Y_split_train.npy", "y_train.npy"])
x_val_path = _pick(args.data_directory, ["X_split_test.npy", "X_validate_features.npy"])
y_val_path = _pick(args.data_directory, ["Y_split_test.npy", "y_validate.npy"])

print("[AUG] Dir listing:", sorted(os.listdir(args.data_directory)))
print(
    f"[AUG] Train X: {os.path.basename(x_train_path)} | Train y: {os.path.basename(y_train_path)}"
)
print(
    f"[AUG] Val   X: {os.path.basename(x_val_path)}   | Val   y: {os.path.basename(y_val_path)}"
)

x_train = np.load(x_train_path)
y_train = np.load(y_train_path)
x_val = np.load(x_val_path)
y_val = np.load(y_val_path)

print(f"[AUG] x_train (raw): {x_train.shape}, x_val (raw): {x_val.shape}")
print(f"[AUG] y_train (raw): {y_train.shape}, y_val (raw): {y_val.shape}")


# ---------------- Ensure NHWC (N,96,96,3) ----------------
def _to_nhwc(x: np.ndarray, name: str) -> np.ndarray:
    if x.ndim == 2:
        if x.shape[1] != EXPECTED_FEAT_LEN:
            raise ValueError(
                f"[AUG][FATAL] {name} feature len {x.shape[1]} != {EXPECTED_FEAT_LEN}"
            )
        x = x.reshape((-1, IMG_H, IMG_W, C))
    if x.ndim != 4 or x.shape[1:] != (IMG_H, IMG_W, C):
        raise ValueError(
            f"[AUG][FATAL] Bad tensor shape for {name}: {x.shape}, expected (N,{IMG_H},{IMG_W},{C})"
        )
    return x.astype(np.float32, copy=False)


x_train = _to_nhwc(x_train, "x_train")
x_val = _to_nhwc(x_val, "x_val")

# ---------------- Label-space checks BEFORE argmax ----------------
if y_train.ndim == 2 and y_val.ndim == 2:
    if y_train.shape[1] != y_val.shape[1]:
        raise ValueError(
            f"[AUG][FATAL] Label-space mismatch: train one-hot width={y_train.shape[1]} "
            f"vs val width={y_val.shape[1]}. Recalculate features for ALL data."
        )
    NUM_CLASSES = int(y_train.shape[1])
    y_train = y_train.argmax(axis=1)
    y_val = y_val.argmax(axis=1)
elif y_train.ndim == 1 and y_val.ndim == 1:
    NUM_CLASSES = int(max(y_train.max(), y_val.max()) + 1)
else:
    raise ValueError(
        f"[AUG][FATAL] Inconsistent label dims: train={y_train.ndim}D, val={y_val.ndim}D"
    )

# Optional: shift if both sets look 1-indexed
if y_train.min() == 1 and y_val.min() == 1:
    print("[AUG] Shifting labels 1-indexed → 0-indexed.")
    y_train -= 1
    y_val -= 1

# Range sanity
if (y_train.min() < 0) or (y_val.min() < 0):
    raise ValueError("[AUG][FATAL] Negative class index found.")
if (y_train.max() >= NUM_CLASSES) or (y_val.max() >= NUM_CLASSES):
    raise ValueError(
        f"[AUG][FATAL] Class index out of range w.r.t NUM_CLASSES={NUM_CLASSES}"
    )

# Balanced class weights
class_counts = np.bincount(y_train.astype(int), minlength=NUM_CLASSES)
inv_freq = class_counts.max() / np.maximum(class_counts, 1)
class_weight = {i: float(inv_freq[i]) for i in range(NUM_CLASSES)}
print(f"[AUG] INPUT_SHAPE={INPUT_SHAPE}, NUM_CLASSES={NUM_CLASSES}")
print(f"[AUG] Class counts (first 20): {class_counts[:20]}")


# ---------------- tf.data (one-hot inside pipeline) ----------------
def to_one_hot(y):
    return tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES)


augment = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ],
    name="augment",
)


def make_ds(x, y, training=True):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(8 * HP["batch_size"], reshuffle_each_iteration=True)
    ds = ds.map(
        lambda xi, yi: (xi, to_one_hot(yi)), num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(HP["batch_size"]).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_ds(x_train, y_train, training=True)
val_ds = make_ds(x_val, y_val, training=False)


# ---------------- Optional MixUp/CutMix (enabled if alpha > 0) ----------------
def _sample_beta(alpha, shape):
    if alpha <= 0.0:
        return tf.ones(shape)
    g1 = tf.random.gamma(shape=shape, alpha=alpha)
    g2 = tf.random.gamma(shape=shape, alpha=alpha)
    return g1 / (g1 + g2)


def apply_mixup(images, labels_oh, alpha):
    if alpha <= 0.0:
        return images, labels_oh
    bs = tf.shape(images)[0]
    lam = _sample_beta(alpha, (bs, 1, 1, 1))
    idx = tf.random.shuffle(tf.range(bs))
    mixed_x = lam * images + (1.0 - lam) * tf.gather(images, idx)
    lam_lbl = tf.reshape(lam[:, 0, 0, 0], (bs, 1))
    mixed_y = lam_lbl * labels_oh + (1.0 - lam_lbl) * tf.gather(labels_oh, idx)
    return mixed_x, mixed_y


def apply_cutmix(images, labels_oh, alpha):
    if alpha <= 0.0:
        return images, labels_oh
    bs = tf.shape(images)[0]
    h = tf.shape(images)[1]
    w = tf.shape(images)[2]
    lam = _sample_beta(alpha, (bs, 1))
    rx = tf.cast(tf.random.uniform((bs,)) * tf.cast(w, tf.float32), tf.int32)
    ry = tf.cast(tf.random.uniform((bs,)) * tf.cast(h, tf.float32), tf.int32)
    rw = tf.cast(tf.sqrt(1.0 - lam[:, 0]) * tf.cast(w, tf.float32), tf.int32)
    rh = tf.cast(tf.sqrt(1.0 - lam[:, 0]) * tf.cast(h, tf.float32), tf.int32)

    x1 = tf.clip_by_value(rx - rw // 2, 0, w)
    y1 = tf.clip_by_value(ry - rh // 2, 0, h)
    x2 = tf.clip_by_value(rx + rw // 2, 0, w)
    y2 = tf.clip_by_value(ry + rh // 2, 0, h)

    idx = tf.random.shuffle(tf.range(bs))
    xr = tf.range(w)
    yr = tf.range(h)
    X, Y = tf.meshgrid(xr, yr)
    X = tf.expand_dims(tf.expand_dims(X, 0), -1)  # [1,H,W,1]
    Y = tf.expand_dims(tf.expand_dims(Y, 0), -1)
    in_x = (X >= tf.cast(tf.expand_dims(x1, 1)[:, None, None, :], X.dtype)) & (
        X < tf.cast(tf.expand_dims(x2, 1)[:, None, None, :], X.dtype)
    )
    in_y = (Y >= tf.cast(tf.expand_dims(y1, 1)[:, None, None, :], Y.dtype)) & (
        Y < tf.cast(tf.expand_dims(y2, 1)[:, None, None, :], Y.dtype)
    )
    box = tf.cast(in_x & in_y, images.dtype)  # [B,H,W,1]

    mixed_x = images * (1.0 - box) + tf.gather(images, idx) * box
    box_area = tf.reduce_mean(box, axis=[1, 2, 3])
    lam_eff = tf.reshape(1.0 - box_area, (-1, 1))
    mixed_y = lam_eff * labels_oh + (1.0 - lam_eff) * tf.gather(labels_oh, idx)
    return mixed_x, mixed_y


def with_mixups(ds):
    if HP["mixup_alpha"] <= 0.0 and HP["cutmix_alpha"] <= 0.0:
        return ds

    def _aug(images, labels_oh):
        images, labels_oh = apply_mixup(images, labels_oh, HP["mixup_alpha"])
        images, labels_oh = apply_cutmix(images, labels_oh, HP["cutmix_alpha"])
        return images, labels_oh

    return ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)


# ---------------- Model ----------------
inputs = layers.Input(shape=INPUT_SHAPE, name="image_input")
x = augment(inputs)  # active only in training
x = layers.Rescaling(scale=2.0, offset=-1.0)(x)  # [-1,1] for MobileNetV2

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

base.trainable = False  # warmup: frozen
x = base(x, training=False)
x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.Dropout(0.3, name="dropout")(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
model = keras.Model(inputs, outputs)

# Loss & metrics
loss = keras.losses.CategoricalCrossentropy(label_smoothing=HP["label_smoothing"])
metrics = [
    keras.metrics.CategoricalAccuracy(name="acc"),
    keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
]

# ---------------- Phase 1: Warmup (frozen backbone) ----------------
print(f"[AUG] Warmup {HP['warmup_epochs']} epochs @ lr={HP['learning_rate']}")
model.compile(
    optimizer=keras.optimizers.Adam(HP["learning_rate"]), loss=loss, metrics=metrics
)
es = EarlyStopping(
    monitor="val_loss",
    patience=HP["early_stopping_patience"],
    restore_best_weights=True,
    verbose=1,
)

hist_warm = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HP["warmup_epochs"],
    class_weight=class_weight,
    callbacks=[es],
    verbose=2,
)

# ---------------- Phase 2: Fine-tune (unfreeze top N%) ----------------
# Unfreeze the top N% of backbone layers
n_layers = len(base.layers)
cutoff = int((1.0 - float(HP["unfreeze_pct"])) * n_layers)
for i, layer in enumerate(base.layers):
    layer.trainable = i >= cutoff

steps_per_epoch = int(np.ceil(len(x_train) / HP["batch_size"]))
decay_steps = steps_per_epoch * max(HP["epochs"] - HP["warmup_epochs"], 1)
cosine_lr = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=HP["fine_tune_start_lr"], decay_steps=decay_steps
)

print(
    f"[AUG] Finetune to epoch {HP['epochs']} with cosine LR (start {HP['fine_tune_start_lr']}); "
    f"unfreeze top {HP['unfreeze_pct']*100:.1f}% of backbone"
)

model.compile(optimizer=keras.optimizers.Adam(cosine_lr), loss=loss, metrics=metrics)

# apply mixup/cutmix only in fine-tune
train_ft = with_mixups(make_ds(x_train, y_train, training=True))

hist_ft = model.fit(
    train_ft,
    validation_data=val_ds,
    initial_epoch=len(hist_warm.epoch),
    epochs=HP["epochs"],
    class_weight=class_weight,
    callbacks=[es],
    verbose=2,
)

# ---------------- Save (SavedModel + TFLite + /home/model.tflite + merged history) ----------------
out_dir = args.out_directory
os.makedirs(out_dir, exist_ok=True)
saved_model_dir = os.path.join(out_dir, "saved_model")
print(f"[AUG] Saving SavedModel -> {saved_model_dir}")
model.save(saved_model_dir)

tflite_path = os.path.join(out_dir, "model.tflite")
print(f"[AUG] Converting to TFLite (float32) -> {tflite_path}")
tflite_model = None
try:
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = conv.convert()
except Exception as e:
    print(f"[AUG] Primary TFLite conversion failed: {e}")
    print("[AUG] Retrying from SavedModel ...")
    try:
        conv = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = conv.convert()
    except Exception as e2:
        print(f"[AUG][FATAL] TFLite conversion failed from SavedModel: {e2}")

if tflite_model is not None:
    try:
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print("[AUG] TFLite model written.")
    except Exception as e:
        print(f"[AUG][ERROR] Writing {tflite_path} failed: {e}")
    # EI Profiler expects /home/model.tflite
    try:
        with open("/home/model.tflite", "wb") as f:
            f.write(tflite_model)
        print("[AUG] Profiler copy written -> /home/model.tflite")
    except Exception as e:
        print(f"[AUG][WARN] Could not write /home/model.tflite: {e}")

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
print("[AUG] Model and merged history saved.")
