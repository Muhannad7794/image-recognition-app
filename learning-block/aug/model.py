# learning-block/aug/model.py
import os, sys, json, math, argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization

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
p.add_argument("--optimizer", type=str)  # "adam" | "adamw"
p.add_argument(
    "--finetune-optimizer", "--finetune_optimizer", dest="finetune_optimizer", type=str
)
p.add_argument(
    "--warmup-learning-rate",
    "--warmup_learning_rate",
    dest="warmup_learning_rate",
    type=float,
)
p.add_argument(
    "--finetune-learning-rate",
    "--finetune_learning_rate",
    dest="finetune_learning_rate",
    type=float,
)
p.add_argument(
    "--warmup-weight-decay",
    "--warmup_weight_decay",
    dest="warmup_weight_decay",
    type=float,
)
p.add_argument(
    "--finetune-weight-decay",
    "--finetune_weight_decay",
    dest="finetune_weight_decay",
    type=float,
)
p.add_argument(
    "--finetune-unfreeze-pct",
    "--finetune_unfreeze_pct",
    dest="finetune_unfreeze_pct",
    type=float,
)
# Augmentation params
p.add_argument(
    "--use-random-flip", "--use_random_flip", dest="use_random_flip", type=bool
)
p.add_argument(
    "--random-rotation-factor",
    "--random_rotation_factor",
    dest="random_rotation_factor",
    type=float,
)
p.add_argument(
    "--random-zoom-factor",
    "--random_zoom_factor",
    dest="random_zoom_factor",
    type=float,
)
p.add_argument(
    "--random-contrast-factor",
    "--random_contrast_factor",
    dest="random_contrast_factor",
    type=float,
)
p.add_argument("--mixup-alpha", "--mixup_alpha", dest="mixup_alpha", type=float)
p.add_argument("--cutmix-alpha", "--cutmix_alpha", dest="cutmix_alpha", type=float)

args, _ = p.parse_known_args()
print("[DBG] sys.argv =", sys.argv)

# -------------------- Load params from parameters.json --------------------
RUN_PARAMS = os.path.join(
    args.data_directory, "parameters.json"
)  # /home/parameters.json
BLOCK_PARAMS = os.path.join(
    os.getcwd(), "parameters.json"
)  # repo manifest (learning-block-*/parameters.json)


def load_json_safe(p):
    try:
        if os.path.exists(p):
            with open(p, "r") as f:
                return json.load(f) or {}
    except Exception as e:
        print(f"[CFG][WARN] Could not parse {p}: {e}")
    return {}


# 1) EI-provided run parameters (overrides)
run_cfg = load_json_safe(RUN_PARAMS)
# 2) Block manifest parameters (defaults)
repo_cfg = load_json_safe(BLOCK_PARAMS)


def extract_manifest_defaults(manifest: dict) -> dict:
    """Turn the block manifest's training.arguments list into a flat {name: defaultValue} map."""
    out = {}
    try:
        args = manifest.get("training", {}).get("arguments", []) or []
        for a in args:
            name = a.get("name")
            if not name:
                continue
            dv = a.get("defaultValue", None)
            # normalize kebab→underscore for names to match the code
            out[name.replace("-", "_")] = dv
    except Exception as e:
        print(f"[CFG][WARN] Could not extract defaults from manifest: {e}")
    return out


repo_defaults = extract_manifest_defaults(repo_cfg)


# normalize run_cfg keys once (kebab → underscore)
def norm_keys(d):
    return {k.replace("-", "_"): v for k, v in d.items()}


run_cfg = norm_keys(run_cfg)

# CLI overrides
cli_cfg = {}
for k in [
    "epochs",
    "learning_rate",
    "batch_size",
    "warmup_epochs",
    "unfreeze_pct",
    "weight_decay",
    "label_smoothing",
    "use_class_weights",
    "mixup_alpha",
    "cutmix_alpha",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "fine_tune_start_lr",
    "fine_tune_fraction",
    "optimizer",
    "finetune_optimizer",
    "warmup_learning_rate",
    "finetune_learning_rate",
    "warmup_weight_decay",
    "finetune_weight_decay",
    "finetune_unfreeze_pct",
    # AUG params
    "use_random_flip",
    "random_rotation_factor",
    "random_zoom_factor",
    "random_contrast_factor",
]:
    v = getattr(args, k, None)
    if v is not None:
        cli_cfg[k] = v

# FINAL precedence: repo_defaults  <  CLI  <  run_cfg
cfg = {**repo_defaults, **cli_cfg, **run_cfg}

print("[CFG] repo defaults:", json.dumps(repo_defaults, indent=2, sort_keys=True))
print("[CFG] run  params  :", json.dumps(run_cfg, indent=2, sort_keys=True))
print("[CFG] using        :", json.dumps(cfg, indent=2, sort_keys=True))


# -----------Args formatting --------------------
def pick(name, default=None):
    # prefer JSON (cfg), fallback to CLI (args), else default
    v = cfg.get(name, None)
    if v is None:
        v = getattr(args, name, None)
    return default if v is None else v


def as_int(x):
    return None if x is None else int(x)


def as_float(x):
    return None if x is None else float(x)


def as_bool(x):
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "y", "yes", "t"}


HP = {
    "epochs": as_int(cfg.get("epochs")),
    "learning_rate": as_float(cfg.get("learning_rate")),
    "batch_size": as_int(cfg.get("batch_size")),
    "warmup_epochs": as_int(cfg.get("warmup_epochs")),
    "unfreeze_pct": as_float(cfg.get("unfreeze_pct")),
    "weight_decay": as_float(cfg.get("weight_decay")),
    "label_smoothing": as_float(cfg.get("label_smoothing")),
    "use_class_weights": as_bool(cfg.get("use_class_weights")),
    "mixup_alpha": as_float(cfg.get("mixup_alpha")),
    "cutmix_alpha": as_float(cfg.get("cutmix_alpha")),
    "early_stopping_patience": as_int(cfg.get("early_stopping_patience")),
    "early_stopping_min_delta": as_float(cfg.get("early_stopping_min_delta")),
    "optimizer": (cfg.get("optimizer") or "adam"),
    "finetune_optimizer": (
        cfg.get("finetune_optimizer") or cfg.get("optimizer") or "adam"
    ),
    "warmup_learning_rate": as_float(cfg.get("warmup_learning_rate"))
    or as_float(cfg.get("learning_rate")),
    "finetune_learning_rate": as_float(cfg.get("finetune_learning_rate"))
    or as_float(cfg.get("learning_rate")),
    "warmup_weight_decay": as_float(cfg.get("warmup_weight_decay")),
    "finetune_weight_decay": as_float(cfg.get("finetune_weight_decay")),
    "finetune_unfreeze_pct": as_float(cfg.get("finetune_unfreeze_pct"))
    or as_float(cfg.get("unfreeze_pct")),
    "use_random_flip": as_bool(cfg.get("use_random_flip")),
    "random_rotation_factor": as_float(cfg.get("random_rotation_factor")),
    "random_zoom_factor": as_float(cfg.get("random_zoom_factor")),
    "random_contrast_factor": as_float(cfg.get("random_contrast_factor")),
}

# Required args:
required_aug = [
    "epochs",
    "learning_rate",
    "batch_size",
    "warmup_epochs",
    "unfreeze_pct",
    "weight_decay",
    "label_smoothing",
]
missing = [k for k in required_aug if HP[k] is None]
if missing:
    print("[FATAL] Missing hyperparameters (after merge):", missing)
    print("Ensure they exist in the block manifest or are passed via CLI/run params.")
    sys.exit(2)

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

# -------------------- Set train/val data --------------------
x_train = np.load(x_train_path)
y_train = np.load(y_train_path)
x_val = np.load(x_val_path)
y_val = np.load(y_val_path)

print(f"[AUG] x_train (raw): {x_train.shape}, x_val (raw): {x_val.shape}")
print(f"[AUG] y_train (raw): {y_train.shape}, y_val (raw): {y_val.shape}")

# -------------------- DEBUG / SANITY CHECKS (inserted) --------------------

d = args.data_directory

print("[DEBUG] Listing data-dir files (first 50):", sorted(os.listdir(d))[:50])

# Parsing of EI meatadata
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

# Convert to integer labels
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

# If sample_id_details.json exists, verify correspondence
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


def set_finetune_trainable(base_model, unfreeze_pct: float):
    n = len(base_model.layers)
    cutoff = int((1.0 - float(unfreeze_pct)) * n)
    for i, layer in enumerate(base_model.layers):
        if i >= cutoff:
            layer.trainable = not isinstance(layer, BatchNormalization)
        else:
            layer.trainable = False
    return cutoff, n


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

# Class weights
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


def make_optimizer(name: str, lr: float, wd: float | None):
    name = (name or "adam").strip().lower()
    wd = float(wd) if (wd is not None) else 0.0
    if name == "adamw" and wd > 0.0:
        try:
            return keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
        except Exception:
            print("[AUG][WARN] AdamW not available; falling back to Adam.")
            return keras.optimizers.Adam(learning_rate=lr)
    # Default: Adam (tutorial-style baseline)
    return keras.optimizers.Adam(learning_rate=lr)


# --------------------  MixUp/CutMix (fine-tune only) --------------------
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


# -------------------- Augmentation builder --------------------
aug_layers = []

if HP["use_random_flip"]:
    aug_layers.append(layers.RandomFlip("horizontal"))
    print("[AUG] Enabling RandomFlip")

if HP["random_rotation_factor"] and HP["random_rotation_factor"] > 0.0:
    aug_layers.append(layers.RandomRotation(HP["random_rotation_factor"]))
    print(f"[AUG] Enabling RandomRotation (factor={HP['random_rotation_factor']})")

if HP["random_zoom_factor"] and HP["random_zoom_factor"] > 0.0:
    aug_layers.append(layers.RandomZoom(HP["random_zoom_factor"]))
    print(f"[AUG] Enabling RandomZoom (factor={HP['random_zoom_factor']})")

if HP["random_contrast_factor"] and HP["random_contrast_factor"] > 0.0:
    aug_layers.append(layers.RandomContrast(HP["random_contrast_factor"]))
    print(f"[AUG] Enabling RandomContrast (factor={HP['random_contrast_factor']})")


# -------------------- Build model --------------------
augment = keras.Sequential(aug_layers, name="augment")

# -----Define model's inputs -------
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

# -----Define model's Head -------
base.trainable = False  # Freeze base model
x = base(x, training=False)
x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.Dropout(0.3, name="dropout")(x)

# -----Define model's outputs ------
outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
model = keras.Model(inputs, outputs)

# Loss & metrics
loss = keras.losses.CategoricalCrossentropy(label_smoothing=HP["label_smoothing"])
metrics = [
    keras.metrics.CategoricalAccuracy(name="acc"),
    keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
]

# -------------------- Phase 1: Warmup (frozen backbone) --------------------
print(
    f"[AUG] Warmup {HP['warmup_epochs']} epochs @ lr={HP['warmup_learning_rate']} "
    f"(opt={HP['optimizer']}, wd={HP['warmup_weight_decay']})"
)
# Optimizer
optimizer = make_optimizer(
    HP["optimizer"],
    HP["warmup_learning_rate"],
    HP["warmup_weight_decay"],
)
# Compile model
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

# DEBUG: print evaluate to confirm the head can separate and learn
eval_warm_val = model.evaluate(val_ds, verbose=0)
eval_warm_tr = model.evaluate(train_ds, verbose=0)
print(f"[DBG] Warmup VAL: {dict(zip(model.metrics_names, eval_warm_val))}")
print(f"[DBG] Warmup TR : {dict(zip(model.metrics_names, eval_warm_tr))}")


# -------------------- Phase 2: Fine-tune (Unfreeze backbone) --------------------
base.trainable = True  # Unfreeze base model

cutoff, n_layers = set_finetune_trainable(base, HP["finetune_unfreeze_pct"])
print(
    f"[AUG] Unfreezing top {HP['finetune_unfreeze_pct']*100:.1f}% "
    f"(layers >= {cutoff}/{n_layers}); BatchNorms frozen"
)

ft_opt = make_optimizer(
    HP["finetune_optimizer"],
    HP["finetune_learning_rate"],
    (
        HP["finetune_weight_decay"]
        if HP["finetune_weight_decay"] is not None
        else HP["weight_decay"]
    ),
)
print(
    f"[AUG] FT Optimizer: {HP['finetune_optimizer']} "
    f"(lr={HP['finetune_learning_rate']}, wd={HP['finetune_weight_decay'] if HP['finetune_weight_decay'] is not None else HP['weight_decay']})"
)

model.compile(optimizer=ft_opt, loss=loss, metrics=metrics)
# DEBUG: print trainable layers/vars
print(
    "FT – trainable base layers:",
    sum(int(l.trainable) for l in base.layers),
    "/",
    len(base.layers),
)
print("FT – trainable base vars:", sum(int(v.trainable) for v in base.variables))


train_ft = with_mixups(make_ds(x_train, y_train, training=True))
hist_ft = model.fit(
    train_ft,
    validation_data=val_ds,
    initial_epoch=len(hist_warm.epoch),
    epochs=HP["epochs"],
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=2,
)

# DEBUG: print what actually became trainable
trainable = sum(int(l.trainable) for l in base.layers)
print(f"[DBG] Base trainable layers: {trainable}/{len(base.layers)}")

# confirm class weights exist
if class_weight is not None:
    print(
        f"[DBG] class_weight keys: {len(class_weight)}; sample: {list(class_weight.items())[:5]}"
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
