# learning-block/finetune/model.py
# MobileNetV2 (96x96) fine-tune: robust JSON/CLI params, correct scaling, warmup->finetune with cosine LR,
# optional class weights, selectable augmentation strength, clean saving and debug logs.

import os, sys, json, argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# -------------------- Argparse (aliases for underscore/dash) --------------------
p = argparse.ArgumentParser(description="Fine-tune MobileNetV2 @96x96 (robust loader)")
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

# hyperparams (match parameters.json names; accept dash aliases too)
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
    v = cfg.get(name, None)
    if v is None:
        v = getattr(args, name, None)
    return default if v is None else v


# -------------------- Effective hyperparameters (JSON first, then CLI) --------------------
HP = {
    "epochs": int(pick("epochs", 50)),
    "learning_rate": float(pick("learning_rate", 1e-3)),  # warmup (frozen)
    "warmup_epochs": int(pick("warmup_epochs", 12)),
    "fine_tune_start_lr": float(pick("fine_tune_start_lr", 1e-4)),  # start of cosine
    "fine_tune_fraction": float(pick("fine_tune_fraction", 1.0)),  # 1.0 = unfreeze all
    "batch_size": int(pick("batch_size", 64)),
    "label_smoothing": float(pick("label_smoothing", 0.1)),
    "use_class_weights": _to_bool(pick("use_class_weights", True)),
    "early_stopping_patience": int(pick("early_stopping_patience", 10)),
    "augment_strength": str(pick("augment_strength", "medium")).lower(),
}

# Validate required ones
missing = [
    k
    for k in [
        "epochs",
        "learning_rate",
        "warmup_epochs",
        "fine_tune_start_lr",
        "fine_tune_fraction",
        "batch_size",
        "label_smoothing",
        "early_stopping_patience",
        "augment_strength",
    ]
    if HP.get(k) is None
]
if missing:
    print("[FATAL] Missing hyperparameters:", missing)
    print("Ensure these exist in parameters.json (exact names) or pass them via CLI.")
    sys.exit(2)

if HP["fine_tune_fraction"] <= 0 or HP["fine_tune_fraction"] > 1.0:
    print(
        f"[WARN] fine_tune_fraction={HP['fine_tune_fraction']} out of range, clamping to [0.1,1.0]"
    )
    HP["fine_tune_fraction"] = float(np.clip(HP["fine_tune_fraction"], 0.1, 1.0))

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


# Ensure NHWC float32
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

# -------------------- Label-space checks BEFORE argmax --------------------
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

# Optional: shift if both sets look 1-indexed
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

# Class weights (optional)
class_weight = None
if HP["use_class_weights"]:
    counts = np.bincount(y_train.astype(int), minlength=NUM_CLASSES)
    inv = counts.max() / np.maximum(counts, 1)
    class_weight = {i: float(inv[i]) for i in range(NUM_CLASSES)}
    print(f"[FT] Class counts (first 20): {counts[:20]}")
else:
    print("[FT] Class weights disabled.")

print(f"[FT] INPUT_SHAPE={INPUT_SHAPE}, NUM_CLASSES={NUM_CLASSES}")

# -------------------- Scaling diagnostic and layer --------------------
print("[DBG] x_train min/max:", float(x_train.min()), float(x_train.max()))
if float(x_train.max()) > 1.5:  # likely 0..255
    rescale_layer = layers.Rescaling(1.0 / 127.5, offset=-1.0)  # -> [-1,1]
    print("[DBG] Using Rescaling(1/127.5, offset=-1.0) for 0..255 inputs")
else:  # likely 0..1
    rescale_layer = layers.Rescaling(2.0, offset=-1.0)  # -> [-1,1]
    print("[DBG] Using Rescaling(2.0, offset=-1.0) for 0..1 inputs")


# -------------------- Augmentation by strength --------------------
def make_augment(strength: str) -> keras.Sequential:
    strength = (strength or "medium").lower()
    if strength == "off":
        return keras.Sequential([], name="augment_off")
    if strength == "light":
        return keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.05),
                layers.RandomContrast(0.10),
            ],
            name="augment_light",
        )
    if strength == "strong":
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
        ds = ds.shuffle(2048, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda xi, yi: (xi, to_one_hot(yi)), num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(HP["batch_size"]).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_ds(x_train, y_train, training=True)
val_ds = make_ds(x_val, y_val, training=False)

# -------------------- Model --------------------
inputs = layers.Input(shape=INPUT_SHAPE, name="image_input")
x = augment(inputs)  # active only in training
x = rescale_layer(x)  # -> [-1, 1] for MobileNetV2

# Prefer local 96x96 MobileNetV2 weights if present; else "imagenet"
weights_path = os.path.expanduser(
    "~/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5"
)
if os.path.exists(weights_path):
    base = keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE, include_top=False, weights=weights_path
    )
    print("[FT] Using local 96x96 MobileNetV2 weights")
else:
    base = keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE, include_top=False, weights="imagenet"
    )
    print("[FT] Using ImageNet MobileNetV2 weights")

# Warmup: backbone frozen
base.trainable = False
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

# -------------------- Phase 1: Warmup --------------------
print(
    f"[FT] Warmup {HP['warmup_epochs']} epochs @ lr={HP['learning_rate']} (backbone frozen)"
)
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

# -------------------- Phase 2: Finetune (unfreeze fraction) --------------------
# Unfreeze top fraction of layers; fine_tune_fraction=1.0 => unfreeze all
n_layers = len(base.layers)
cutoff = int((1.0 - HP["fine_tune_fraction"]) * n_layers)
for i, layer in enumerate(base.layers):
    layer.trainable = i >= cutoff
print(
    f"[FT] Unfreezing top {HP['fine_tune_fraction']*100:.1f}% (layers >= {cutoff}/{n_layers})"
)

steps_per_epoch = int(np.ceil(len(x_train) / HP["batch_size"]))
ft_epochs = max(HP["epochs"] - HP["warmup_epochs"], 1)
decay_steps = steps_per_epoch * ft_epochs
cosine_lr = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=HP["fine_tune_start_lr"], decay_steps=decay_steps
)
print(
    f"[FT] Finetune for {ft_epochs} epochs, cosine start LR={HP['fine_tune_start_lr']}, decay_steps={decay_steps}"
)

model.compile(optimizer=keras.optimizers.Adam(cosine_lr), loss=loss, metrics=metrics)

hist_ft = model.fit(
    train_ds,
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
