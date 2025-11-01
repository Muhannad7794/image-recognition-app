# learning-block/finetune/model.py
import os, sys, json, argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization

# -------------------- Argparse (aliases for underscore/dash) --------------------
p = argparse.ArgumentParser(
    description="Fine-tune MobileNetV2 @160x160 (robust loader)"
)
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

# -------------------- Argparse (aliases for dash/underscore) --------------------
p.add_argument("--epochs", type=int)
p.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float)
p.add_argument("--warmup-epochs", "--warmup_epochs", dest="warmup_epochs", type=int)
p.add_argument(
    "--fine-tune-start-lr",
    "--fine_tune_start_lr",
    dest="fine_tune_start_lr",
    type=float,
)
p.add_argument(
    "--fine-tune-fraction",
    "--fine_tune_fraction",
    dest="fine_tune_fraction",
    type=float,
)
p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int)
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
    "--augment-strength", "--augment_strength", dest="augment_strength", type=str
)
p.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float)

args, _ = p.parse_known_args()
print("[DBG] sys.argv =", sys.argv)

# -------------------- Load params from parameters.json --------------------
RUN_PARAMS = os.path.join(
    args.data_directory, "parameters.json"
)  # /home/parameters.json
BLOCK_PARAMS = os.path.join(os.getcwd(), "parameters.json")  # repo manifest defaults


def load_json_safe(path):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f) or {}
    except Exception as e:
        print(f"[CFG][WARN] Could not parse {path}: {e}")
    return {}


# 1) EI-provided run parameters (overrides)
run_cfg_raw = load_json_safe(RUN_PARAMS)
# 2) Block manifest parameters (defaults)
repo_cfg_raw = load_json_safe(BLOCK_PARAMS)


def extract_manifest_defaults(manifest: dict) -> dict:
    out = {}
    try:
        args_list = manifest.get("training", {}).get("arguments", []) or []
        for a in args_list:
            name = a.get("name")
            if not name:
                continue
            dv = a.get("defaultValue", None)
            out[name.replace("-", "_")] = dv  # normalize kebab->underscore
    except Exception as e:
        print(f"[CFG][WARN] Could not extract defaults from manifest: {e}")
    return out


repo_defaults = extract_manifest_defaults(repo_cfg_raw)


def norm_keys(d):  # kebab->underscore once
    return {(k.replace("-", "_") if isinstance(k, str) else k): v for k, v in d.items()}


run_cfg = norm_keys(run_cfg_raw)

# CLI overrides
cli_cfg = {}
for k in [
    "epochs",
    "learning_rate",
    "batch_size",
    "warmup_epochs",
    "fine_tune_start_lr",
    "fine_tune_fraction",
    "label_smoothing",
    "use_class_weights",
    "early_stopping_patience",
    "augment_strength",
    "weight_decay",
]:
    v = getattr(args, k, None)
    if v is not None:
        cli_cfg[k] = v

# FINAL precedence: repo_defaults  <  CLI  <  run_cfg
cfg = {**repo_defaults, **cli_cfg, **run_cfg}

print("[CFG] repo defaults:", json.dumps(repo_defaults, indent=2, sort_keys=True))
print("[CFG] run  params  :", json.dumps(run_cfg, indent=2, sort_keys=True))
print("[CFG] using        :", json.dumps(cfg, indent=2, sort_keys=True))


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
    "warmup_epochs": as_int(cfg.get("warmup_epochs")),
    "fine_tune_start_lr": as_float(cfg.get("fine_tune_start_lr")),
    "fine_tune_fraction": as_float(cfg.get("fine_tune_fraction")),
    "batch_size": as_int(cfg.get("batch_size")),
    "label_smoothing": as_float(cfg.get("label_smoothing")),
    "use_class_weights": as_bool(cfg.get("use_class_weights")),
    "early_stopping_patience": as_int(cfg.get("early_stopping_patience")),
    "augment_strength": (
        str(cfg.get("augment_strength"))
        if cfg.get("augment_strength") is not None
        else "medium"
    ).lower(),
    # optional: AdamW if provided in JSON/CLI; safe default 0.0 => Adam fallback
    "weight_decay": as_float(cfg.get("weight_decay")),
}

# Required for finetune block
required_ft = [
    "epochs",
    "learning_rate",
    "warmup_epochs",
    "fine_tune_start_lr",
    "fine_tune_fraction",
    "batch_size",
    "label_smoothing",
    "early_stopping_patience",
]
missing = [k for k in required_ft if HP.get(k) is None]
if missing:
    print("[FATAL] Missing hyperparameters (after merge):", missing)
    print("Ensure they exist in the block manifest or are passed via CLI/run params.")
    sys.exit(2)

# Clamp fraction defensively
if HP["fine_tune_fraction"] <= 0 or HP["fine_tune_fraction"] > 1.0:
    print(
        f"[WARN] fine_tune_fraction={HP['fine_tune_fraction']} out of range, clamping to [0.1,1.0]"
    )
    HP["fine_tune_fraction"] = float(np.clip(HP["fine_tune_fraction"], 0.1, 1.0))

print("[HP] Effective hyperparameters:", json.dumps(HP, indent=2, sort_keys=True))

# -------------------- Constants --------------------
IMG_H, IMG_W, C = 160, 160, 3
INPUT_SHAPE = (IMG_H, IMG_W, C)
EXPECTED_FEAT_LEN = IMG_H * IMG_W * C


# -------------------- File resolution --------------------
def pick_path(root: str, names: list[str]) -> str:
    for n in names:
        pth = os.path.join(root, n)
        if os.path.exists(pth):
            return pth
    raise FileNotFoundError(f"None of {names} found under {root}")


print(f"[FT] Data dir: {args.data_directory}")
print("[FT] Dir listing:", sorted(os.listdir(args.data_directory)))

x_train_path = pick_path(
    args.data_directory, ["X_split_train.npy", "X_train_features.npy"]
)
y_train_path = pick_path(args.data_directory, ["Y_split_train.npy", "y_train.npy"])
x_val_path = pick_path(
    args.data_directory, ["X_split_test.npy", "X_validate_features.npy"]
)
y_val_path = pick_path(args.data_directory, ["Y_split_test.npy", "y_validate.npy"])

print(
    f"[FT] Train X: {os.path.basename(x_train_path)} | Train y: {os.path.basename(y_train_path)}"
)
print(
    f"[FT] Val   X: {os.path.basename(x_val_path)}   | Val   y: {os.path.basename(y_val_path)}"
)

# -------------------- Load arrays --------------------
x_train = np.load(x_train_path)
y_train = np.load(y_train_path)
x_val = np.load(x_val_path)
y_val = np.load(y_val_path)

print(f"[FT] x_train (raw): {x_train.shape}, x_val (raw): {x_val.shape}")
print(f"[FT] y_train (raw): {y_train.shape}, y_val (raw): {y_val.shape}")


# -------------------- DEBUG / SANITY CHECKS --------------------
def to_nhwc(x: np.ndarray, name: str) -> np.ndarray:
    if x.ndim == 2:
        if x.shape[1] != EXPECTED_FEAT_LEN:
            raise ValueError(
                f"[FT][FATAL] {name} feature len {x.shape[1]} != {EXPECTED_FEAT_LEN}"
            )
        x = x.reshape((-1, IMG_H, IMG_W, C))
    if x.ndim != 4 or x.shape[1:] != (IMG_H, IMG_W, C):
        raise ValueError(
            f"[FT][FATAL] Bad tensor shape for {name}: {x.shape}, expected (N,{IMG_H},{IMG_W},{C})"
        )
    return x.astype(np.float32, copy=False)


x_train = to_nhwc(x_train, "x_train")
x_val = to_nhwc(x_val, "x_val")

# Label-space checks BEFORE argmax
if y_train.ndim == 2 and y_val.ndim == 2:
    if y_train.shape[1] != y_val.shape[1]:
        raise ValueError(
            f"[FT][FATAL] Label-space mismatch: train width={y_train.shape[1]} vs val width={y_val.shape[1]}"
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

# Optional 1-indexed shift
if y_train.min() == 1 and y_val.min() == 1:
    print("[FT] Shifting labels 1-indexed → 0-indexed.")
    y_train -= 1
    y_val -= 1

# Range sanity
if (y_train.min() < 0) or (y_val.min() < 0):
    raise ValueError("[FT][FATAL] Negative class index found.")
if (y_train.max() >= NUM_CLASSES) or (y_val.max() >= NUM_CLASSES):
    raise ValueError(
        f"[FT][FATAL] Class index out of range w.r.t NUM_CLASSES={NUM_CLASSES}"
    )

# Quick label distribution debug
ytrain_idx = y_train.astype(int)
yval_idx = y_val.astype(int)
uniq_tr, cnt_tr = np.unique(ytrain_idx, return_counts=True)
uniq_vl, cnt_vl = np.unique(yval_idx, return_counts=True)
print("[FT][DBG] Unique train labels:", uniq_tr[:30])
print("[FT][DBG] Train counts head   :", cnt_tr[:30])
print("[FT][DBG] Unique val labels  :", uniq_vl[:30])
print("[FT][DBG] Val counts head    :", cnt_vl[:30])
print(f"[FT] INPUT_SHAPE={INPUT_SHAPE}, NUM_CLASSES={NUM_CLASSES}")

# Class weights (optional)
class_weight = None
if HP["use_class_weights"]:
    counts = np.bincount(y_train.astype(int), minlength=NUM_CLASSES)
    inv = counts.max() / np.maximum(counts, 1)
    class_weight = {i: float(inv[i]) for i in range(NUM_CLASSES)}
    print(f"[FT] Class counts (first 20): {counts[:20]}")
else:
    print("[FT] Class weights disabled.")

# -------------------- Scaling diagnostic and layer --------------------
print("[DBG] x_train min/max:", float(x_train.min()), float(x_train.max()))
if float(x_train.max()) > 1.5:
    rescale_layer = layers.Rescaling(1.0 / 127.5, offset=-1.0)  # -> [-1,1]
    print("[DBG] Using Rescaling(1/127.5, offset=-1.0) for 0..255 inputs")
else:
    rescale_layer = layers.Rescaling(2.0, offset=-1.0)  # -> [-1,1]
    print("[DBG] Using Rescaling(2.0, offset=-1.0) for 0..1 inputs")


# -------------------- Augmentation by strength --------------------
def make_augment(strength: str) -> keras.Sequential:
    s = (strength or "medium").lower()
    if s == "off":
        return keras.Sequential([], name="augment_off")
    if s == "light":
        return keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.05),
                layers.RandomContrast(0.10),
            ],
            name="augment_light",
        )
    if s == "strong":
        return keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.20),
                layers.RandomZoom(0.25),
                layers.RandomTranslation(0.15, 0.15),
                layers.RandomContrast(0.35),
            ],
            name="augment_strong",
        )
    # default "medium"
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.10),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.10, 0.10),
            layers.RandomContrast(0.20),
        ],
        name="augment_medium",
    )


augment = make_augment(HP["augment_strength"])
print(f"[FT] Augmentation: {augment.name}")


# -------------------- tf.data (one-hot in pipeline) --------------------
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


def set_finetune_trainable_fraction(base_model, fraction: float):
    n = len(base_model.layers)
    cutoff = int((1.0 - float(fraction)) * n)
    for i, layer in enumerate(base_model.layers):
        if i >= cutoff:
            # Train conv/etc., but DO NOT train BN
            layer.trainable = not isinstance(layer, BatchNormalization)
        else:
            layer.trainable = False
    return cutoff, n


# -------------------- Model --------------------
inputs = layers.Input(shape=INPUT_SHAPE, name="image_input")
x = augment(inputs)  # augmentation is active during training
x = rescale_layer(x)  # -> [-1, 1] for MobileNetV2

# Prefer local 160x160 MobileNetV2 weights if present; else "imagenet"
weights_path = os.path.expanduser(
    "~/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5"
)
if os.path.exists(weights_path):
    base = keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE, include_top=False, weights=weights_path
    )
    print("[FT] Using local 160x160 MobileNetV2 weights")
else:
    base = keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE, include_top=False, weights="imagenet"
    )
    print("[FT] Using ImageNet MobileNetV2 weights")

# Warmup: backbone frozen, BN in inference mode
base.trainable = False
x = base(x, training=False)
# --- head (NO L2 on Dense; AdamW will handle weight decay if enabled) ---
x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.Dropout(0.3, name="dropout")(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
model = keras.Model(inputs, outputs)

# Loss & metrics
loss = keras.losses.CategoricalCrossentropy(
    label_smoothing=(HP["label_smoothing"] or 0.0)
)
metrics = [
    keras.metrics.CategoricalAccuracy(name="acc"),
    keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
]

# -------------------- Phase 1: Warmup --------------------
print(
    f"[FT] Warmup {HP['warmup_epochs']} epochs @ lr={HP['learning_rate']} (backbone frozen)"
)
# Use AdamW if weight_decay > 0 and available; else Adam
use_adamw = HP["weight_decay"] is not None and HP["weight_decay"] > 0.0
if use_adamw:
    try:
        optimizer = keras.optimizers.AdamW(
            learning_rate=HP["learning_rate"], weight_decay=HP["weight_decay"]
        )
        print("[FT] Optimizer: AdamW (warmup)")
    except Exception:
        optimizer = keras.optimizers.Adam(learning_rate=HP["learning_rate"])
        use_adamw = False
        print("[FT] Optimizer: Adam (AdamW not available)")
else:
    optimizer = keras.optimizers.Adam(learning_rate=HP["learning_rate"])
    print("[FT] Optimizer: Adam (weight_decay<=0)")

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

es = EarlyStopping(
    monitor="val_loss",
    patience=HP["early_stopping_patience"],
    restore_best_weights=True,
    verbose=1,
)

callbacks = []
if HP["early_stopping_patience"] and HP["early_stopping_patience"] > 0:
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


# -------------------- Phase 2: Finetune (unfreeze fraction; BN frozen) --------------------
# Allow per-layer flags to take effect in Phase-2
base.trainable = True

cutoff, n_layers = set_finetune_trainable_fraction(base, HP["fine_tune_fraction"])
print(
    f"[FT] Unfreezing top {HP['fine_tune_fraction']*100:.1f}% (layers >= {cutoff}/{n_layers}); BatchNorms frozen"
)

# Cosine LR schedule starting at fine_tune_start_lr
steps_per_epoch = int(np.ceil(len(x_train) / HP["batch_size"]))
ft_epochs = max(HP["epochs"] - HP["warmup_epochs"], 1)
decay_steps = steps_per_epoch * ft_epochs
cosine_lr = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=HP["fine_tune_start_lr"], decay_steps=decay_steps
)
print(
    f"[FT] Finetune for {ft_epochs} epochs, cosine start LR={HP['fine_tune_start_lr']}, decay_steps={decay_steps}"
)

if use_adamw:
    try:
        ft_opt = keras.optimizers.AdamW(
            learning_rate=cosine_lr, weight_decay=HP["weight_decay"]
        )
        print("[FT] FT Optimizer: AdamW (cosine)")
    except Exception:
        ft_opt = keras.optimizers.Adam(learning_rate=cosine_lr)
        print("[FT] FT Optimizer: Adam (AdamW not available)")
else:
    ft_opt = keras.optimizers.Adam(learning_rate=cosine_lr)
    print("[FT] FT Optimizer: Adam (weight_decay<=0)")

model.compile(optimizer=ft_opt, loss=loss, metrics=metrics)
# DEBUG: print trainable layers/vars
print(
    "FT – trainable base layers:",
    sum(int(l.trainable) for l in base.layers),
    "/",
    len(base.layers),
)
print("FT – trainable base vars:", sum(int(v.trainable) for v in base.variables))

hist_ft = model.fit(
    train_ds,
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
print(f"[FT] Saving SavedModel -> {saved_model_dir}")
try:
    model.save(saved_model_dir)
except Exception as e:
    print(f"[FT][ERROR] Saving SavedModel failed: {e}")

tflite_path = os.path.join(out_dir, "model.tflite")
print(f"[FT] Converting to TFLite (float32) -> {tflite_path}")
tflite_model = None
try:
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = conv.convert()
except Exception as e:
    print(f"[FT] Primary TFLite conversion failed: {e}")
    print("[FT] Retrying from SavedModel ...")
    try:
        conv = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = conv.convert()
    except Exception as e2:
        print(f"[FT][ERROR] TFLite conversion failed from SavedModel: {e2}")

if tflite_model is not None:
    try:
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print("[FT] TFLite model written.")
    except Exception as e:
        print(f"[FT][ERROR] Writing {tflite_path} failed: {e}")
    # EI profiler path
    try:
        with open("/home/model.tflite", "wb") as f:
            f.write(tflite_model)
        print("[FT] Profiler copy written -> /home/model.tflite")
    except Exception as e:
        print(f"[FT][WARN] Could not write /home/model.tflite: {e}")

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
    print(f"[FT] Training history written -> {hist_path}")
except Exception as e:
    print(f"[FT][WARN] Could not write history: {e}")

print("[FT] Done.")
