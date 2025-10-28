# model.py (AUG) — MobileNetV2 (96x96), robust loading, safe label-space, conservative aug, fixed saving
import os, sys, json, math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------- LR Schedule ----------------
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_epochs, steps_per_epoch, total_epochs):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = max(1, warmup_epochs * steps_per_epoch)
        self.total_steps = max(self.warmup_steps + 1, total_epochs * steps_per_epoch)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        # Linear warmup
        lr_warm = self.base_lr * (step / tf.cast(self.warmup_steps, tf.float32))
        # Cosine after warmup
        progress = (step - self.warmup_steps) / tf.cast(
            self.total_steps - self.warmup_steps, tf.float32
        )
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        lr_cos = 0.5 * self.base_lr * (1.0 + tf.cos(math.pi * progress))
        return tf.where(step < self.warmup_steps, lr_warm, lr_cos)


# ---------------- Optional MixUp/CutMix ----------------
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
    lam = _sample_beta(alpha, (bs, 1))  # beta for area
    rx = tf.cast(tf.random.uniform((bs,)) * tf.cast(w, tf.float32), tf.int32)
    ry = tf.cast(tf.random.uniform((bs,)) * tf.cast(h, tf.float32), tf.int32)
    rw = tf.cast(tf.sqrt(1.0 - lam[:, 0]) * tf.cast(w, tf.float32), tf.int32)
    rh = tf.cast(tf.sqrt(1.0 - lam[:, 0]) * tf.cast(h, tf.float32), tf.int32)

    x1 = tf.clip_by_value(rx - rw // 2, 0, w)
    y1 = tf.clip_by_value(ry - rh // 2, 0, h)
    x2 = tf.clip_by_value(rx + rw // 2, 0, w)
    y2 = tf.clip_by_value(ry + rh // 2, 0, h)

    idx = tf.random.shuffle(tf.range(bs))

    # Vectorized binary mask of cut region
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
    box_area = tf.reduce_mean(box, axis=[1, 2, 3])  # fraction replaced
    lam_eff = tf.reshape(1.0 - box_area, (-1, 1))
    mixed_y = lam_eff * labels_oh + (1.0 - lam_eff) * tf.gather(labels_oh, idx)
    return mixed_x, mixed_y


# ---------------- Model ----------------
def build_model(
    input_shape, num_classes, unfreeze_pct=0.7, weight_decay=1e-4, dropout=0.3
):
    inputs = keras.Input(shape=input_shape)
    # on-model light aug
    x = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="augment",
    )(inputs)
    # 0..1 -> [-1,1] for MobileNetV2
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)

    base = keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base.trainable = False

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(
        256, activation="relu", kernel_regularizer=keras.regularizers.l2(weight_decay)
    )(x)
    x = layers.Dropout(dropout)(x)
    logits = layers.Dense(
        num_classes,
        activation=None,
        kernel_regularizer=keras.regularizers.l2(weight_decay),
    )(x)
    model = keras.Model(inputs, logits, name="aug_mnetv2_96")
    return model, base


# ---------------- Train ----------------
def compile_and_train(
    x_train, y_train_sparse, x_val, y_val_sparse, num_classes, args, class_weights=None
):
    bs = int(args.get("batch_size", 64))
    epochs = int(args.get("epochs", 30))
    base_lr = float(args.get("learning_rate", 1e-4))
    warmup_epochs = int(args.get("warmup_epochs", 6))
    unfreeze_pct = float(args.get("unfreeze_pct", 0.7))
    weight_decay = float(args.get("weight_decay", 1e-4))
    label_smoothing = float(args.get("label_smoothing", 0.1))
    mixup_alpha = float(args.get("mixup_alpha", 0.2))
    cutmix_alpha = float(args.get("cutmix_alpha", 0.2))

    input_shape = x_train.shape[1:]
    model, base = build_model(input_shape, num_classes, unfreeze_pct, weight_decay)

    steps_per_epoch = max(1, int(np.ceil(len(x_train) / bs)))
    lr_schedule = WarmupCosine(base_lr, warmup_epochs, steps_per_epoch, epochs)
    try:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule, weight_decay=weight_decay
        )
    except TypeError:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    loss = keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=label_smoothing
    )
    metrics = [
        keras.metrics.CategoricalAccuracy(name="acc"),
        keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # one-hot inside the pipeline (keeps label space authoritative at NUM_CLASSES)
    def to_one_hot(y):
        return tf.one_hot(tf.cast(y, tf.int32), num_classes)

    # ds helpers
    def make_ds(x, y, training):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if training:
            ds = ds.shuffle(8 * bs, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda xi, yi: (xi, to_one_hot(yi)), num_parallel_calls=tf.data.AUTOTUNE
        )
        ds = ds.batch(bs).prefetch(tf.data.AUTOTUNE)
        return ds

    train_warm = make_ds(x_train, y_train_sparse, training=True)
    val_ds = make_ds(x_val, y_val_sparse, training=False)

    # -------- Warmup (frozen) --------
    hist_parts = []
    warm_hist = model.fit(
        train_warm.take(steps_per_epoch * warmup_epochs),
        validation_data=val_ds,
        epochs=warmup_epochs,
        class_weight=class_weights,
        verbose=2,
    )
    hist_parts.append(warm_hist.history)

    # -------- Unfreeze top N% --------
    n_layers = len(base.layers)
    cutoff = int((1.0 - unfreeze_pct) * n_layers)
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= cutoff

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # MixUp/CutMix only during fine-tune
    def aug_batch(images, labels_oh):
        images, labels_oh = apply_mixup(images, labels_oh, mixup_alpha)
        images, labels_oh = apply_cutmix(images, labels_oh, cutmix_alpha)
        return images, labels_oh

    aug_train = make_ds(x_train, y_train_sparse, training=True).map(
        aug_batch, num_parallel_calls=tf.data.AUTOTUNE
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, min_delta=0.002
        )
    ]
    ft_hist = model.fit(
        aug_train,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=warmup_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )
    hist_parts.append(ft_hist.history)

    return model, hist_parts


# ---------------- Saving utilities ----------------
def merge_histories(history_parts):
    merged = {}
    for h in history_parts or []:
        for k, v in h.items():
            merged.setdefault(k, [])
            merged[k].extend(list(v))
    return merged


def save_model_artifacts(model, history_parts, out_directory):
    """
    Saves:
      - SavedModel -> <out_directory>/saved_model/
      - TFLite (float32) -> <out_directory>/model.tflite  and /home/model.tflite (for EI profiler)
      - training_history.json -> <out_directory>/training_history.json
    """
    if not out_directory:
        print("[AUG][WARN] No out_directory provided; skipping saves.")
        return
    os.makedirs(out_directory, exist_ok=True)
    saved_model_dir = os.path.join(out_directory, "saved_model")
    tflite_path = os.path.join(out_directory, "model.tflite")
    tflite_profiler_path = "/home/model.tflite"
    hist_path = os.path.join(out_directory, "training_history.json")

    print(f"[AUG][SAVE] Saving Keras SavedModel -> {saved_model_dir}")
    try:
        model.save(saved_model_dir)
    except Exception as e:
        print(f"[AUG][ERROR] Saving SavedModel failed: {e}")

    print("[AUG][SAVE] Converting to TFLite (float32)")
    tflite_model = None
    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = conv.convert()
    except Exception as e:
        print(f"[AUG][WARN] TFLite from Keras model failed: {e}")
        print("[AUG][SAVE] Retrying TFLite from SavedModel...")
        try:
            conv = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            tflite_model = conv.convert()
        except Exception as e2:
            print(f"[AUG][ERROR] TFLite from SavedModel failed: {e2}")

    if tflite_model is not None:
        try:
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
            print(f"[AUG][SAVE] TFLite model written -> {tflite_path}")
        except Exception as e:
            print(f"[AUG][ERROR] Writing TFLite failed: {e}")
        try:
            with open(tflite_profiler_path, "wb") as f:
                f.write(tflite_model)
            print(f"[AUG][SAVE] Profiler copy -> {tflite_profiler_path}")
        except Exception as e:
            print(f"[AUG][WARN] Could not write profiler copy: {e}")

    try:
        merged_hist = merge_histories(history_parts)
        with open(hist_path, "w") as f:
            json.dump(merged_hist, f, default=lambda o: float(o))
        print(f"[AUG][SAVE] Training history written -> {hist_path}")
    except Exception as e:
        print(f"[AUG][ERROR] Writing training history failed: {e}")


# ---------------- Main ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train AUG MobileNetV2 (96x96)")
    parser.add_argument("--data-directory", type=str, required=True)
    parser.add_argument("--out-directory", type=str, required=True)
    # sensible defaults, overridable
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=6)
    parser.add_argument("--unfreeze-pct", type=float, default=0.7)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--cutmix-alpha", type=float, default=0.2)
    args, _ = parser.parse_known_args()

    # ---- load robustly (same split-aware approach) ----
    IMG_H, IMG_W, C = 96, 96, 3
    expected_feat_len = IMG_H * IMG_W * C
    dd = args.data_directory

    def pick(names):
        for n in names:
            p = os.path.join(dd, n)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"None of {names} found under {dd}")

    paths = {
        "x_train": ["X_split_train.npy", "X_train_features.npy"],
        "y_train": ["Y_split_train.npy", "y_train.npy"],
        "x_val": ["X_split_test.npy", "X_validate_features.npy"],
        "y_val": ["Y_split_test.npy", "y_validate.npy"],
    }

    x_train = np.load(pick(paths["x_train"]))
    y_train = np.load(pick(paths["y_train"]))
    x_val = np.load(pick(paths["x_val"]))
    y_val = np.load(pick(paths["y_val"]))

    def to_nhwc(x, name):
        if x.ndim == 2:
            if x.shape[1] != expected_feat_len:
                raise ValueError(
                    f"[AUG][FATAL] {name} feature len {x.shape[1]} != {expected_feat_len}"
                )
            x = x.reshape((-1, IMG_H, IMG_W, C))
        if x.ndim != 4 or x.shape[1:] != (IMG_H, IMG_W, C):
            raise ValueError(
                f"[AUG][FATAL] Bad tensor shape for {name}: {x.shape}, expected (N,{IMG_H},{IMG_W},{C})"
            )
        return x.astype(np.float32, copy=False)

    x_train = to_nhwc(x_train, "x_train")
    x_val = to_nhwc(x_val, "x_val")

    # ---- label space checks BEFORE argmax (mirrors finetune) ----
    if y_train.ndim == 2 and y_val.ndim == 2:
        if y_train.shape[1] != y_val.shape[1]:
            raise ValueError(
                f"[AUG][FATAL] Label-space mismatch: train one-hot width={y_train.shape[1]} "
                f"vs val width={y_val.shape[1]}. Recompute features for ALL data."
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

    # ---- train & save ----
    model, hist_parts = compile_and_train(
        x_train, y_train, x_val, y_val, NUM_CLASSES, vars(args), class_weights=None
    )
    save_model_artifacts(model, hist_parts, args.out_directory)
