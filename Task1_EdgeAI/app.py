# EdgeAI_recyclables_train_and_tflite.py
# Run in Colab or local machine with TensorFlow installed (TF 2.9+ recommended).
# Assumes a dataset directory structured as:
# dataset/
# train/
# classA/
# classB/
# val/
# classA/
# classB/
# test/
# classA/
# classB/

import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools
import json

# ---------- Config ----------
DATA_DIR = "/content/dataset" # change as needed
BATCH_SIZE = 32
IMG_SIZE = (160, 160) # small for edge
EPOCHS = 15
MODEL_SAVE_DIR = "/content/model_output"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ---------- Helper: load datasets ----------
def load_datasets(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
    )
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, test_ds, class_names

# ---------- Build model (lightweight MobileNetV2 transfer) ----------
def build_model(num_classes, img_size=IMG_SIZE):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights='imagenet',
        alpha=0.35, # lightweight
        pooling='avg'
    )
    base_model.trainable = False # freeze for initial training

    inputs = keras.Input(shape=(img_size[0], img_size[1], 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # light augmentation
    data_augment = keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),
    ])
    x = data_augment(x)
    x = base_model(x, training=False)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base_model

# ---------- Train, fine-tune ----------
def train_and_finetune(train_ds, val_ds, class_names):
    model, base_model = build_model(len(class_names))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(MODEL_SAVE_DIR, "best_model.h5"),
                                           save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy')
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    # optional fine-tuning: unfreeze last few layers
    base_model.trainable = True
    # unfreeze only last n layers to avoid overfitting
    fine_tune_at = int(len(base_model.layers) * 0.9)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    ft_history = model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=callbacks)
    # Save final model
    model.save(os.path.join(MODEL_SAVE_DIR, "final_model.h5"))
    return model, history, ft_history

# ---------- Evaluate on test set ----------
def evaluate_and_report(model, test_ds, class_names):
    # Predictions
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model.predict(images)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    # write json report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(MODEL_SAVE_DIR, f"classification_report_{timestamp}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    # save confusion matrix image
    plot_confusion_matrix(cm, class_names, os.path.join(MODEL_SAVE_DIR, f"confusion_{timestamp}.png"))
    return report, cm

def plot_confusion_matrix(cm, classes, out_file):
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

# ---------- TFLite conversion and quantization ----------
def convert_to_tflite(keras_model_path, tflite_out_path, quantize=False, representative_data=None):
    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        # full integer quantization
        if representative_data is None:
            print("Representative data not provided — using float16 quantization.")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        else:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            def rep_gen():
                for input_value in representative_data.take(100):
                    # input_value: (images, labels)
                    yield [tf.cast(input_value[0], tf.float32)]
            converter.representative_dataset = rep_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    with open(tflite_out_path, "wb") as f:
        f.write(tflite_model)
    print("Saved TFLite model to:", tflite_out_path)
    return tflite_out_path

# ---------- Main ----------
if __name__ == "__main__":
    # Load data
    train_ds, val_ds, test_ds, class_names = load_datasets(DATA_DIR)
    print("Classes:", class_names)

    # Train
    model, history, ft_history = train_and_finetune(train_ds, val_ds, class_names)

    # Evaluate
    report, cm = evaluate_and_report(model, test_ds, class_names)
    print("Classification report (summary):")
    for cls, metrics in report.items():
        if cls in class_names:
            print(f"{cls}: precision {metrics['precision']:.3f} recall {metrics['recall']:.3f} f1 {metrics['f1-score']:.3f}")

    # Convert to TFLite (float16 quantization by default)
    keras_model_path = os.path.join(MODEL_SAVE_DIR, "final_model.h5")
    tflite_path = os.path.join(MODEL_SAVE_DIR, "model_fp16.tflite")
    convert_to_tflite(keras_model_path, tflite_path, quantize=True, representative_data=None)

    # Print file sizes
    print("Model files in", MODEL_SAVE_DIR)
    for fname in os.listdir(MODEL_SAVE_DIR):
        fp = os.path.join(MODEL_SAVE_DIR, fname)
        print(fname, f"{os.path.getsize(fp)/1024:.1f} KB")