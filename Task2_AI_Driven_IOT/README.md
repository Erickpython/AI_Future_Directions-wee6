# Edge AI & AI-Driven IoT Concepts  
This repository contains two prototype tasks:  
1. **Edge AI Prototype using TensorFlow Lite**  
2. **AI-Driven IoT Smart Agriculture Concept**

---

# 📌 Task 1: Edge AI Prototype (TensorFlow Lite)

### 🎯 Goal
- Train a lightweight image classification model (e.g., recyclable items recognition).  
- Convert the trained model into TensorFlow Lite format.  
- Test the model on a sample dataset and evaluate accuracy.  
- Explain Edge AI benefits and outline deployment steps (Raspberry Pi / Colab simulation).

---

## 🧠 Model Training (TensorFlow/Keras)

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Dataset dirs
train_dir = "dataset/train"
val_dir = "dataset/val"

# Image generators
train_gen = ImageDataGenerator(rescale=1/255)
val_gen = ImageDataGenerator(rescale=1/255)

train_data = train_gen.flow_from_directory(train_dir, target_size=(128, 128), batch_size=32)
val_data = val_gen.flow_from_directory(val_dir, target_size=(128, 128), batch_size=32)

# Model
model = models.Sequential([
    layers.Input(shape=(128,128,3)),
    layers.Conv2D(16, 3, activation='relu'), layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'), layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(32, activation="relu"),
    layers.Dense(len(train_data.class_indices), activation="softmax")
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(train_data, validation_data=val_data, epochs=10)

# Save original model
model.save("recyclables_model.h5")


---

🔄 Convert Model to TensorFlow Lite

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)


---

🧪 Testing the TFLite Model

import numpy as np
import tensorflow as tf
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load sample image
img = Image.open("sample.jpg").resize((128,128))
img = np.array(img) / 255.0
img = np.expand_dims(img, axis=0).astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()
pred = interpreter.get_tensor(output_details[0]['index'])

print("Predicted class:", np.argmax(pred))


---

📊 Accuracy Metrics (Example Format)

Metric Score

Training Accuracy 92%
Validation Accuracy 88%
Loss 0.38



---

🚀 Deployment Steps (Raspberry Pi / Edge Device)

1. Install TensorFlow Lite runtime.


2. Copy model.tflite to Raspberry Pi.


3. Write a Python script to read camera input and run inference.


4. Optimize using:

Quantization

Model pruning

Edge TPU (optional)





---

⚡ Why Edge AI Matters (Short Explanation)

Runs locally → faster, real-time predictions.

Reduces cloud cost and latency.

Works even without internet.

Protects user privacy since data stays on device.



---

📌 Task 2: AI-Driven IoT Smart Agriculture Concept

🌱 Scenario

Design an AI + IoT system that predicts crop yield and monitors farm conditions in real time.


---

🧩 Required Sensors

Soil moisture sensor

Temperature sensor

Humidity sensor

Rainfall sensor

Light intensity (LDR)

Soil pH sensor

CO₂ sensor (optional)



---

🤖 AI Model Proposal

Model type: Regression or Multivariate Time-Series Model (LSTM)

Predicts:

Crop yield based on sensor data trends

Expected plant performance

Early detection of crop stress


Input features:

Moisture levels

Temperature

pH

Weather trends

Light exposure

Historical yield data


Output:

Estimated crop yield in kg

Recommended actions (e.g., watering, fertilization)



---

🗺 Data Flow Diagram (ASCII Sketch)

[Sensors: Moisture, Temp, pH, Light]
                       |
                       v
            [Microcontroller / IoT Node]
                       |
                 (MQTT / HTTP)
                       |
                       v
               [Cloud / Edge Gateway]
                       |
                 AI Model Processes:
       - Data cleaning
       - Feature extraction
       - Yield prediction
                       |
                       v
             [Dashboard / Mobile App]
       - Live sensor data
       - Crop yield forecast
       - Alerts & recommendations

