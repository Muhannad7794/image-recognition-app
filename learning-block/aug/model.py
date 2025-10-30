# learning-block/aug/model.py
# MobileNetV2 (96x96), robust JSON/CLI params, correct scaling, MixUp/CutMix, clean saving.

import os, sys, json, math, argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# -------------------- Argparse (aliases for dash/underscore) --------------------
p = argparse.ArgumentParser(description="AUG MobileNetV2 @96x96 (robust loader)")
p.add_argument(
    "--data-directory",
    "--data_directory",
    dest="data_directory",
    type=str,
    required=True,
)
p.add_argument(
    "--out-directory", "--out_directory", dest="out_directory", type=str, required=True
)

# Hyperparams as aliases (match parameters.json underscore names, accept dash forms too)
p.add_argument("--epochs", type=int)
p.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float)
p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int)
p.add_argument("--warmup-epochs", "--warmup_epochs", dest="warmup_epochs", type=int)
p.add_argument("--unfreeze-pct", "--unfreeze_pct", dest="unfreeze_pct", type=float)
p.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float)
p.add_argument(
    "--label-smoothing", "--label_smoothing", dest="label_smoothing", type=float
)
p.add_argument(
    "--use-class-weights", "--use_class_weights", dest="use_class_weights", type=str
)
p.add_argument("--mixup-alpha", "--mixup_alpha", dest="mixup_alpha", type=float)
p.add_argument("--cutmix-alpha", "--cutmix_alpha", dest="cutmix_alpha", type=float)
p.add_argument(
    "--early-stopping-patience",
    "--early_stopping_patience",
    dest="early_stopping_patience",
    type=int,
)
p.add_argument(
    "--early-stopping-min-delta",
    "--early_stopping_min_delta",
    dest="early_stopping_min_delta",
    type=float,
)

args, _ = p.parse_known_args()
print("[DBG] sys.argv =", sys.argv)


# -------------------- Load parameters.json (source of truth) --------------------
def load_params_json():
    candidates = [
        os.path.join(args.data_directory, "parameters.json"),
        os.path.join(os.getcwd(), "parameters.json"),
    ]
    cfg = {}
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    cfg = json.load(f) or {}
                print(f"[CFG] Loaded {path}")
                break
            except Exception as e:
                print(f"[CFG][WARN] Could not parse {path}: {e}")
    # normalize keys: kebab->underscore
    return {k.replace("-", "_"): v for k, v in cfg.items()}


cfg = load_params_json()


def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "t"}
    return False


def pick(name, default=None):
    # prefer JSON (cfg), fallback to CLI (args), else default
    v = cfg.get(name, None)
    if v is None:
        v = getattr(args, name, None)
    return default if v is None else v


HP = {
    "epochs": int(pick("epochs", 2)),
    "learning_rate": float(pick("learning_rate", 0.0005)),
    "batch_size": int(pick("batch_size", 64)),
    "warmup_epochs": int(pick("warmup_epochs", 1)),
    "unfreeze_pct": float(pick("unfreeze_pct", 0.7)),
    "weight_decay": float(pick("weight_decay", 0.0001)),
    "label_smoothing": float(pick("label_smoothing", 0.0)),
    "use_class_weights": _to_bool(pick("use_class_weights", False)),
    "mixup_alpha": float(pick("mixup_alpha", 0.0)),
    "cutmix_alpha": float(pick("cutmix_alpha", 0.0)),
    "early_stopping_patience": int(pick("early_stopping_patience", 8)),
    "early_stopping_min_delta": float(pick("early_stopping_min_delta", 0.002)),
}

# Validate the ones we truly need
missing = [
    k
    for k in [
        "epochs",
        "learning_rate",
        "batch_size",
        "warmup_epochs",
        "unfreeze_pct",
        "weight_decay",
        "label_smoothing",
    ]
    if HP.get(k) is None
]
if missing:
    print("[FATAL] Missing hyperparameters:", missing)
    sys.exit(2)

print("[HP] Effective hyperparameters:", HP)

# -------------------- Constants --------------------
IMG_H, IMG_W, C = 96, 96, 3
INPUT_SHAPE = (IMG_H, IMG_W, C)
EXPECTED_FEAT_LEN = IMG_H * IMG_W * C


# -------------------- File resolution --------------------
def pick_path(root: str, names: list[str]) -> str:
    for n in names:
        pth = os.path.join(root, n)
        if os.path.exists(pth):
            return pth
    raise FileNotFoundError(f"None of {names} found under {root}")


print(f"[AUG] Data dir: {args.data_directory}")
print("[AUG] Dir listing:", sorted(os.listdir(args.data_directory)))

x_train_path = pick_path(
    args.data_directory, ["X_split_train.npy", "X_train_features.npy"]
)
y_train_path = pick_path(args.data_directory, ["Y_split_train.npy", "y_train.npy"])
x_val_path = pick_path(
    args.data_directory, ["X_split_test.npy", "X_validate_features.npy"]
)
y_val_path = pick_path(args.data_directory, ["Y_split_test.npy", "y_validate.npy"])

print(
    f"[AUG] Train X: {os.path.basename(x_train_path)} | Train y: {os.path.basename(y_train_path)}"
)
print(
    f"[AUG] Val   X: {os.path.basename(x_val_path)}   | Val   y: {os.path.basename(y_val_path)}"
)

# -------------------- Load arrays --------------------
x_train = np.load(x_train_path)
y_train = np.load(y_train_path)
x_val = np.load(x_val_path)
y_val = np.load(y_val_path)

print(f"[AUG] x_train (raw): {x_train.shape}, x_val (raw): {x_val.shape}")
print(f"[AUG] y_train (raw): {y_train.shape}, y_val (raw): {y_val.shape}")

# -------------------- DEBUG / SANITY CHECKS (inserted) --------------------

d = args.data_directory

print("[DEBUG] Listing data-dir files (first 50):", sorted(os.listdir(d))[:50])

# Quick parse of common metadata files the EI split usually creates
for f in ("sample_id_details.json", "split_metadata.json", "dsp_metadata.json"):
    p = os.path.join(d, f)
    if os.path.exists(p):
        print(f"[DEBUG] Found {f} at {p}")
        try:
            with open(p, "r") as fh:
                meta = json.load(fh)
            # print a small sample
            if isinstance(meta, list):
                print(
                    f"[DEBUG] {f} is a list with {len(meta)} items, sample[0..2]:",
                    meta[:3],
                )
            elif isinstance(meta, dict):
                print(f"[DEBUG] {f} keys:", list(meta.keys())[:10])
            else:
                print(f"[DEBUG] {f} type:", type(meta))
        except Exception as e:
            print(f"[DEBUG] Could not parse {f}: {e}")

# Check Y format
print(
    "[DEBUG] y_train dtype/ndim/shape:",
    getattr(y_train, "dtype", None),
    getattr(y_train, "ndim", None),
    getattr(y_train, "shape", None),
)
print(
    "[DEBUG] y_val   dtype/ndim/shape:",
    getattr(y_val, "dtype", None),
    getattr(y_val, "ndim", None),
    getattr(y_val, "shape", None),
)

# Convert to integer labels (this mirrors your code's argmax behavior)
if y_train.ndim == 2:
    ytrain_idx = y_train.argmax(axis=1)
    yval_idx = y_val.argmax(axis=1)
else:
    ytrain_idx = y_train.astype(int)
    yval_idx = y_val.astype(int)

print(
    "[DEBUG] Unique train labels (counts, first 30):",
    np.unique(ytrain_idx, return_counts=True)[0][:30],
)
print(
    "[DEBUG] Unique train label counts (first 30):",
    np.unique(ytrain_idx, return_counts=True)[1][:30],
)
print(
    "[DEBUG] Unique val label counts (first 30):",
    np.unique(yval_idx, return_counts=True)[1][:30],
)

# Sanity: lengths must match x arrays
print("[DEBUG] x_train rows:", x_train.shape[0], "y_train rows:", ytrain_idx.shape[0])
print("[DEBUG] x_val   rows:", x_val.shape[0], "y_val   rows:", yval_idx.shape[0])

# Show first 20 labels to visually inspect possible systematic offsets
print("[DEBUG] train label head (first 20):", ytrain_idx[:20])
print("[DEBUG] val   label head (first 20):", yval_idx[:20])

# If sample_id_details.json exists, verify correspondence (common EI field name = 'sample_id' or similar)
sid_path = os.path.join(d, "sample_id_details.json")
if os.path.exists(sid_path):
    try:
        with open(sid_path, "r") as f:
            sid = json.load(f)
        # If sid is a list of sample objects, print first 6 and compare to labels
        if isinstance(sid, list):
            print("[DEBUG] sample_id_details sample (first 6):", sid[:6])
            print("[DEBUG] Count sample_id_details:", len(sid))
            # If sample entries have 'split' or 'dataset' field, try to correlate
            # If sample entries have 'id' or 'filename' print for first few
            if len(sid) >= max(5, x_train.shape[0] if x_train.shape[0] < 50 else 50):
                # only print if lengths comparable
                pass
        else:
            print(
                "[DEBUG] sample_id_details is not a list — keys:", list(sid.keys())[:10]
            )
    except Exception as e:
        print("[DEBUG] Error reading sample_id_details.json:", e)
else:
    print("[DEBUG] sample_id_details.json not present in data directory.")
# End debug


# Ensure NHWC float32
def to_nhwc(x: np.ndarray, name: str) -> np.ndarray:
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


x_train = to_nhwc(x_train, "x_train")
x_val = to_nhwc(x_val, "x_val")

# -------------------- Label-space checks BEFORE argmax --------------------
if y_train.ndim == 2 and y_val.ndim == 2:
    if y_train.shape[1] != y_val.shape[1]:
        raise ValueError(
            f"[AUG][FATAL] Label-space mismatch: train width={y_train.shape[1]} vs val width={y_val.shape[1]}"
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

# Optional 1-indexed shift
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

# Class weights (optional)
class_weight = None
if HP["use_class_weights"]:
    counts = np.bincount(y_train.astype(int), minlength=NUM_CLASSES)
    inv = counts.max() / np.maximum(counts, 1)
    class_weight = {i: float(inv[i]) for i in range(NUM_CLASSES)}
    print(f"[AUG] Class counts (first 20): {counts[:20]}")
else:
    print("[AUG] Class weights disabled.")

print(f"[AUG] INPUT_SHAPE={INPUT_SHAPE}, NUM_CLASSES={NUM_CLASSES}")

# -------------------- Scaling diagnostic and layer --------------------
print("[DBG] x_train min/max:", float(x_train.min()), float(x_train.max()))
if float(x_train.max()) > 1.5:  # likely 0..255
    rescale_layer = layers.Rescaling(1.0 / 127.5, offset=-1.0)  # -> [-1,1]
    print("[DBG] Using Rescaling(1/127.5, offset=-1.0) for 0..255 inputs")
else:  # likely 0..1
    rescale_layer = layers.Rescaling(2.0, offset=-1.0)  # -> [-1,1]
    print("[DBG] Using Rescaling(2.0, offset=-1.0) for 0..1 inputs")


# -------------------- tf.data pipeline (one-hot inside) --------------------
def to_one_hot(y):
    return tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES)


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


# -------------------- Optional MixUp/CutMix (fine-tune only) --------------------
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
    X = tf.expand_dims(tf.expand_dims(X, 0), -1)
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
        print("[AUG] MixUp/CutMix disabled.")
        return ds
    print(
        f"[AUG] Enabling MixUp (alpha={HP['mixup_alpha']}) and CutMix (alpha={HP['cutmix_alpha']})"
    )

    def _aug(images, labels_oh):
        images, labels_oh = apply_mixup(images, labels_oh, HP["mixup_alpha"])
        images, labels_oh = apply_cutmix(images, labels_oh, HP["cutmix_alpha"])
        return images, labels_oh

    return ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)


# -------------------- Model --------------------
augment = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ],
    name="augment",
)

inputs = layers.Input(shape=INPUT_SHAPE, name="image_input")
x = augment(inputs)
x = rescale_layer(x)  # dynamic scale -> [-1, 1]

# Prefer local 96x96 MobileNetV2 weights if present; else "imagenet"
weights_path = os.path.expanduser(
    "~/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5"
)
if os.path.exists(weights_path):
    base = keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE, include_top=False, weights=weights_path
    )
    print("[AUG] Using local 96x96 MobileNetV2 weights")
else:
    base = keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE, include_top=False, weights="imagenet"
    )
    print("[AUG] Using ImageNet MobileNetV2 weights")

base.trainable = False  # warmup: frozen
x = base(x, training=False)
x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.Dropout(0.3, name="dropout")(x)
outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    kernel_regularizer=keras.regularizers.l2(HP["weight_decay"]),
    name="predictions",
)(x)
model = keras.Model(inputs, outputs)

# Loss & metrics
loss = keras.losses.CategoricalCrossentropy(label_smoothing=HP["label_smoothing"])
metrics = [
    keras.metrics.CategoricalAccuracy(name="acc"),
    keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
]

# -------------------- Phase 1: Warmup (frozen backbone) --------------------
print(f"[AUG] Warmup {HP['warmup_epochs']} epochs @ lr={HP['learning_rate']}")
try:
    optimizer = keras.optimizers.AdamW(
        learning_rate=HP["learning_rate"], weight_decay=HP["weight_decay"]
    )
    print("[AUG] Optimizer: AdamW")
except Exception:
    optimizer = keras.optimizers.Adam(learning_rate=HP["learning_rate"])
    print("[AUG] Optimizer: Adam (AdamW not available)")

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
es = EarlyStopping(
    monitor="val_loss",
    patience=HP["early_stopping_patience"],
    min_delta=HP["early_stopping_min_delta"],
    restore_best_weights=True,
    verbose=1,
)

callbacks = []
if HP["early_stopping_patience"] > 0:
    callbacks.append(es)

hist_warm = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HP["warmup_epochs"],
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=2,
)

# -------------------- Phase 2: Fine-tune (unfreeze top N%) --------------------
n_layers = len(base.layers)
cutoff = int((1.0 - HP["unfreeze_pct"]) * n_layers)
for i, layer in enumerate(base.layers):
    layer.trainable = i >= cutoff
print(
    f"[AUG] Unfreezing top {HP['unfreeze_pct']*100:.1f}% (layers >= {cutoff}/{n_layers})"
)

steps_per_epoch = int(np.ceil(len(x_train) / HP["batch_size"]))
ft_epochs = max(HP["epochs"] - HP["warmup_epochs"], 1)
decay_steps = steps_per_epoch * ft_epochs
cosine_lr = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=HP["learning_rate"], decay_steps=decay_steps
)
try:
    ft_opt = keras.optimizers.AdamW(
        learning_rate=cosine_lr, weight_decay=HP["weight_decay"]
    )
    print("[AUG] FT Optimizer: AdamW (cosine schedule)")
except Exception:
    ft_opt = keras.optimizers.Adam(learning_rate=cosine_lr)
    print("[AUG] FT Optimizer: Adam (cosine schedule)")

model.compile(optimizer=ft_opt, loss=loss, metrics=metrics)

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

# -------------------- Save (SavedModel + TFLite + /home/model.tflite) --------------------
out_dir = args.out_directory
os.makedirs(out_dir, exist_ok=True)
saved_model_dir = os.path.join(out_dir, "saved_model")
print(f"[AUG] Saving SavedModel -> {saved_model_dir}")
try:
    model.save(saved_model_dir)
except Exception as e:
    print(f"[AUG][ERROR] Saving SavedModel failed: {e}")

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
        print(f"[AUG][ERROR] TFLite conversion failed from SavedModel: {e2}")

if tflite_model is not None:
    try:
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print("[AUG] TFLite model written.")
    except Exception as e:
        print(f"[AUG][ERROR] Writing {tflite_path} failed: {e}")
    # EI profiler path
    try:
        with open("/home/model.tflite", "wb") as f:
            f.write(tflite_model)
        print("[AUG] Profiler copy written -> /home/model.tflite")
    except Exception as e:
        print(f"[AUG][WARN] Could not write /home/model.tflite: {e}")

# -------------------- Save merged history --------------------
hist_path = os.path.join(out_dir, "training_history.json")
merged = {}
for k, v in hist_warm.history.items():
    merged[k] = list(v)
for k, v in hist_ft.history.items():
    merged.setdefault(k, [])
    merged[k].extend(v)
try:
    with open(hist_path, "w") as f:
        json.dump(merged, f, default=lambda o: float(o))
    print(f"[AUG] Training history written -> {hist_path}")
except Exception as e:
    print(f"[AUG][WARN] Could not write history: {e}")

print("[AUG] Done.")
