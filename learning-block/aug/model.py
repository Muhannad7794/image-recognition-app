import os, json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------- Inputs ----------
INPUT_SHAPE = (96, 96, 3)  # (H, W, C)
NUM_CLASSES = 47
WARMUP_EPOCHS = 8
TOTAL_EPOCHS = 50
HEAD_LR = 1e-3
FT_LR = 1e-4
LABEL_SMOOTH = 0.1
MIXUP_ALPHA = 0.2
BATCH_SIZE = 64

# ---------- Load data ----------
x_train = np.load("X_split_train.npy").astype("float32")
y_train_oh = np.load("Y_split_train.npy").astype("float32")  # one-hot (n,47)
x_val = np.load("X_split_test.npy").astype("float32")
y_val_oh = np.load("Y_split_test.npy").astype("float32")

# If stored as one-hot but model expects class ids:
y_train = np.argmax(y_train_oh, axis=1)
y_val = np.argmax(y_val_oh, axis=1)


# Resize to 96 if needed (keeps pipeline simple)
def _resize(imgs):
    return tf.image.resize(imgs, INPUT_SHAPE[:2]).numpy()


if x_train.shape[1:3] != INPUT_SHAPE[:2]:
    x_train = _resize(x_train)
    x_val = _resize(x_val)


# ---------- Datasets with MixUp ----------
def mixup(ds, alpha=MIXUP_ALPHA, num_classes=NUM_CLASSES):
    if alpha <= 0:
        return ds

    def _sample_beta(shape):
        gamma1 = tf.random.gamma(shape, alpha, 1.0)
        gamma2 = tf.random.gamma(shape, alpha, 1.0)
        return gamma1 / (gamma1 + gamma2)

    def _mix(a, b):
        (x1, y1), (x2, y2) = a, b
        lam = _sample_beta((tf.shape(x1)[0], 1, 1, 1))
        lam_y = tf.reshape(lam, (-1, 1))
        x = x1 * lam + x2 * (1.0 - lam)
        y = tf.one_hot(y1, num_classes) * lam_y + tf.one_hot(y2, num_classes) * (
            1.0 - lam_y
        )
        return x, y

    ds2 = ds.shuffle(1024, reshuffle_each_iteration=True)
    return tf.data.Dataset.zip((ds, ds2)).map(_mix, num_parallel_calls=tf.data.AUTOTUNE)


augment = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.2),
    ],
    name="augment",
)


def make_ds(x, y, training=True):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(2048, reshuffle_each_iteration=True)
        ds = ds.map(lambda xi, yi: (xi, yi), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    if training and MIXUP_ALPHA > 0:
        ds = mixup(ds, MIXUP_ALPHA)
    return ds


train_ds = make_ds(x_train, y_train, training=True)
val_ds = make_ds(x_val, y_val, training=False)

# ---------- Model ----------
inputs = layers.Input(shape=INPUT_SHAPE, name="image_input")
x = augment(inputs)
x = layers.Rescaling(scale=1 / 127.5, offset=-1)(x)  # [-1, 1]
base = keras.applications.MobileNetV2(
    input_shape=INPUT_SHAPE, include_top=False, weights="imagenet", alpha=1.0
)
base.trainable = False  # warmup: freeze

x = base(x, training=False)
x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
model = keras.Model(inputs, outputs)


# ---------- Loss, metrics, schedule ----------
def top5(y_true, y_pred):  # y_true as class ids
    return tf.keras.metrics.top_k_categorical_accuracy(
        tf.one_hot(tf.cast(y_true, tf.int32), NUM_CLASSES), y_pred, k=5
    )


loss = keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH)
metrics = [
    keras.metrics.SparseCategoricalAccuracy(name="acc"),
    keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
]

# class weights (balanced)
class_counts = np.bincount(y_train, minlength=NUM_CLASSES)
inv_freq = class_counts.max() / np.maximum(class_counts, 1)
class_weight = {i: float(inv_freq[i]) for i in range(NUM_CLASSES)}

# Warmup compile
model.compile(optimizer=keras.optimizers.Adam(HEAD_LR), loss=loss, metrics=metrics)

cbs = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    ),
]

# ---------- Phase 1: warmup ----------
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=WARMUP_EPOCHS,
    class_weight=class_weight,
    callbacks=cbs,
    verbose=2,
)

# ---------- Phase 2: fine-tune ----------
# Unfreeze top ~40% of layers
for layer in base.layers[int(len(base.layers) * 0.6) :]:
    layer.trainable = True

# Cosine decay schedule for fine-tuning
steps_per_epoch = int(np.ceil(len(x_train) / BATCH_SIZE))
decay = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=FT_LR,
    decay_steps=steps_per_epoch * (TOTAL_EPOCHS - WARMUP_EPOCHS),
)
model.compile(optimizer=keras.optimizers.Adam(decay), loss=loss, metrics=metrics)

model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=model.history.epoch[-1] + 1,
    epochs=TOTAL_EPOCHS,
    class_weight=class_weight,
    callbacks=cbs,
    verbose=2,
)


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
    conv = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = conv.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print("[AUG] TFLite model written (from SavedModel).")

# Save history
hist_path = os.path.join(out_dir, "training_history.json")
with open(hist_path, "w") as f:
    json.dump(history.history, f, default=lambda o: float(o))
print("[AUG] Model and history saved.")
