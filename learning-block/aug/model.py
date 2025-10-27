# model.py (AUG) — MobileNetV2 (96x96), balanced training, conservative aug
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import json
import sys


# ---------- Schedules ----------
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
        lr_cos = 0.5 * self.base_lr * (1 + tf.cos(math.pi * progress))
        return tf.where(step < self.warmup_steps, lr_warm, lr_cos)


# ---------- Optional MixUp/CutMix ----------
def _sample_beta(alpha, shape):
    if alpha <= 0.0:
        return tf.ones(shape)
    gamma1 = tf.random.gamma(shape=shape, alpha=alpha)
    gamma2 = tf.random.gamma(shape=shape, alpha=alpha)
    lam = gamma1 / (gamma1 + gamma2)
    return lam


def apply_mixup(images, labels, alpha):
    if alpha <= 0.0:
        return images, labels
    bs = tf.shape(images)[0]
    lam = _sample_beta(alpha, (bs, 1, 1, 1))
    index = tf.random.shuffle(tf.range(bs))
    mixed_x = lam * images + (1 - lam) * tf.gather(images, index)
    lam_lbl = tf.reshape(lam[:, 0, 0, 0], (bs, 1))
    mixed_y = lam_lbl * labels + (1 - lam_lbl) * tf.gather(labels, index)
    return mixed_x, mixed_y


def apply_cutmix(images, labels, alpha):
    if alpha <= 0.0:
        return images, labels
    bs = tf.shape(images)[0]
    h = tf.shape(images)[1]
    w = tf.shape(images)[2]
    lam = _sample_beta(alpha, (bs, 1))
    rx = tf.cast(tf.random.uniform((bs,)) * tf.cast(w, tf.float32), tf.int32)
    ry = tf.cast(tf.random.uniform((bs,)) * tf.cast(h, tf.float32), tf.int32)
    rw = tf.cast(tf.sqrt(1 - lam[:, 0]) * tf.cast(w, tf.float32), tf.int32)
    rh = tf.cast(tf.sqrt(1 - lam[:, 0]) * tf.cast(h, tf.float32), tf.int32)

    x1 = tf.clip_by_value(rx - rw // 2, 0, w)
    y1 = tf.clip_by_value(ry - rh // 2, 0, h)
    x2 = tf.clip_by_value(rx + rw // 2, 0, w)
    y2 = tf.clip_by_value(ry + rh // 2, 0, h)

    index = tf.random.shuffle(tf.range(bs))
    images2 = tf.gather(images, index)
    labels2 = tf.gather(labels, index)

    def cut_one(img, img2, xi1, yi1, xi2, yi2):
        mask = tf.ones((yi2 - yi1, xi2 - xi1, tf.shape(img)[-1]), img.dtype)
        pad_top = yi1
        pad_left = xi1
        pad_bottom = tf.shape(img)[0] - yi2
        pad_right = tf.shape(img)[1] - xi2
        patch = tf.pad(mask, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
        return img * (1 - patch) + img2 * patch

    out_imgs = []
    new_lams = []
    for i in range(
        bs.numpy() if isinstance(bs, tf.Tensor) and not tf.executing_eagerly() else 0
    ):
        pass  # (Graph mode safe implementation below)

    # Vectorized cutmix (graph friendly)
    batch_indices = tf.range(bs)
    x1e = tf.expand_dims(x1, 1)
    y1e = tf.expand_dims(y1, 1)
    x2e = tf.expand_dims(x2, 1)
    y2e = tf.expand_dims(y2, 1)
    # Build binary masks
    xr = tf.range(w)
    yr = tf.range(h)
    X, Y = tf.meshgrid(xr, yr)
    X = tf.expand_dims(tf.expand_dims(X, 0), -1)  # [1,H,W,1]
    Y = tf.expand_dims(tf.expand_dims(Y, 0), -1)
    in_x = (X >= tf.cast(x1e[:, None, None, :], X.dtype)) & (
        X < tf.cast(x2e[:, None, None, :], X.dtype)
    )
    in_y = (Y >= tf.cast(y1e[:, None, None, :], Y.dtype)) & (
        Y < tf.cast(y2e[:, None, None, :], Y.dtype)
    )
    box = tf.cast(in_x & in_y, images.dtype)  # [B,H,W,1]

    mixed = images * (1 - box) + tf.gather(images, index) * box
    box_area = tf.reduce_mean(box, axis=[1, 2, 3])  # fraction replaced
    lam_eff = 1.0 - box_area
    lam_eff = tf.reshape(lam_eff, (-1, 1))
    mixed_y = lam_eff * labels + (1 - lam_eff) * tf.gather(labels, index)
    return mixed, mixed_y


# ---------- Model ----------
def build_model(
    input_shape, num_classes, unfreeze_pct=0.7, weight_decay=1e-4, dropout=0.3
):
    inputs = keras.Input(shape=input_shape)

    # 0..1 -> [-1,1] for MobileNetV2
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)

    # Conservative on-model augmentation
    aug = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ]
    )
    x = aug(x)

    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base.trainable = False  # will unfreeze later

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(
        256, kernel_regularizer=keras.regularizers.l2(weight_decay), activation="relu"
    )(x)
    x = layers.Dropout(dropout)(x)
    logits = layers.Dense(
        num_classes,
        activation=None,
        kernel_regularizer=keras.regularizers.l2(weight_decay),
    )(x)

    model = keras.Model(inputs, logits, name="aug_mnetv2_96")
    return model, base


def compile_and_train(
    x_train, y_train, x_val, y_val, num_classes, args, class_weights=None
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

    steps_per_epoch = max(1, math.ceil(len(x_train) / bs))
    lr_schedule = WarmupCosine(base_lr, warmup_epochs, steps_per_epoch, epochs)
    try:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule, weight_decay=weight_decay
        )
    except TypeError:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=label_smoothing
    )
    top5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc"), top5],
    )

    # tf.data pipeline + optional MixUp/CutMix after warmup
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(8 * bs)
        .batch(bs)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(bs)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Warmup phase with frozen base
    hist_parts = []
    warm_hist = model.fit(
        train_ds.take(steps_per_epoch * warmup_epochs),
        validation_data=val_ds,
        epochs=warmup_epochs,
        class_weight=class_weights,
        verbose=2,
    )
    hist_parts.append(warm_hist.history)

    # Unfreeze top N% of layers
    n_layers = len(base.layers)
    cutoff = int((1.0 - unfreeze_pct) * n_layers)
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= cutoff

    # Re-compile (BNs now trainable in unfrozen part)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc"), top5],
    )

    # Fine-tune with optional MixUp/CutMix
    def augment_batch(images, labels):
        images, labels = apply_mixup(images, labels, mixup_alpha)
        images, labels = apply_cutmix(images, labels, cutmix_alpha)
        return images, labels

    aug_train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(8 * bs)
        .batch(bs)
        .map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, min_delta=0.002
        )
    ]

    finetune_hist = model.fit(
        aug_train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=warmup_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )
    hist_parts.append(finetune_hist.history)

    return model, hist_parts


# ---------- Saving utilities (SavedModel, TFLite, history) ----------


def merge_histories(history_parts):
    """
    history_parts: list of dicts like [{'loss': [...], 'acc': [...]}, {...}, ...]
    returns a single dict with lists concatenated per key.
    """
    merged = {}
    for h in history_parts:
        for k, v in h.items():
            merged.setdefault(k, [])
            merged[k].extend(list(v))
    return merged


def save_model_artifacts(model, history_parts, out_directory):
    """
    Saves:
      - SavedModel -> <out_directory>/saved_model/
      - TFLite (float32) -> <out_directory>/model.tflite
      - training_history.json -> <out_directory>/training_history.json
    """
    if not out_directory:
        print("[AUG][WARN] No out_directory provided; skipping saves.")
        return

    os.makedirs(out_directory, exist_ok=True)
    saved_model_dir = os.path.join(out_directory, "saved_model")
    tflite_path = os.path.join(out_directory, "model.tflite")
    hist_path = os.path.join(out_directory, "training_history.json")

    # 1) SavedModel
    print(f"[AUG][SAVE] Saving Keras SavedModel -> {saved_model_dir}")
    try:
        model.save(saved_model_dir)
    except Exception as e:
        print(f"[AUG][ERROR] Saving SavedModel failed: {e}")
        # Don't exit; still attempt TFLite from in-memory model

    # 2) TFLite (float32) with fallback path
    print(f"[AUG][SAVE] Converting to TFLite (float32) -> {tflite_path}")
    tflite_model = None
    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = conv.convert()
    except Exception as e:
        print(f"[AUG][WARN] TFLite conversion from Keras model failed: {e}")
        print("[AUG][SAVE] Retrying TFLite conversion from SavedModel...")
        try:
            conv = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            tflite_model = conv.convert()
        except Exception as e2:
            print(f"[AUG][ERROR] TFLite conversion from SavedModel failed: {e2}")

    if tflite_model is not None:
        try:
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
            print("[AUG][SAVE] TFLite model written.")
        except Exception as e:
            print(f"[AUG][ERROR] Writing TFLite file failed: {e}")

    # 3) Training history (merged)
    try:
        merged_hist = merge_histories(history_parts or [])
        with open(hist_path, "w") as f:
            json.dump(merged_hist, f, default=lambda o: float(o))
        print(f"[AUG][SAVE] Training history written -> {hist_path}")
    except Exception as e:
        print(f"[AUG][ERROR] Writing training history failed: {e}")
