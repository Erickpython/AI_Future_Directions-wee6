# rpi_infer_tflite.py
# For Raspberry Pi (Raspbian). Uses tflite-runtime or full TF if installed.
# Install tflite-runtime (recommended on Pi): pip install tflite-runtime
# For camera input, you can use OpenCV to capture frames.

import argparse
import numpy as np
import cv2
from PIL import Image
import time

try:
    # tflite-runtime import
    from tflite_runtime.interpreter import Interpreter
except Exception:
    # fallback to full TF
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

def load_labels(path):
    # expects a json or simple txt labels one per line. For simplicity:
    with open(path, 'r') as f:
        labels = [l.strip() for l in f.readlines()]
    return labels

def preprocess(image, input_shape, input_type):
    # image: HxWx3 numpy RGB
    img = cv2.resize(image, (input_shape[1], input_shape[2]))
    img = img.astype(np.float32)
    # If model expects uint8, normalize accordingly:
    if input_type == np.uint8:
        # scale 0-255
        return np.expand_dims(img.astype(np.uint8), axis=0)
    else:
        # for float models, use same preprocess as training (MobileNetV2)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return np.expand_dims(img, axis=0)

def run_inference(tflite_model_path, labels_path, camera_index=0):
    labels = load_labels(labels_path)
    interpreter = Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape'] # [1,H,W,3]
    input_type = input_details[0]['dtype']

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    print("Press Ctrl-C to stop")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inp = preprocess(rgb, input_shape, input_type)

            # set tensor
            interpreter.set_tensor(input_details[0]['index'], inp)
            start = time.time()
            interpreter.invoke()
            inference_time = (time.time() - start) * 1000
            out = interpreter.get_tensor(output_details[0]['index'])
            if output_details[0]['dtype'] == np.uint8:
                # dequantize if needed
                scale, zero_point = output_details[0]['quantization']
                probs = scale * (out - zero_point)
            else:
                probs = out[0]
            top_idx = np.argmax(probs)
            label = labels[top_idx]
            prob = probs[top_idx]
            # overlay on frame
            cv2.putText(frame, f"{label}: {prob:.2f} ({inference_time:.0f} ms)", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow('Edge Inference', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="tflite model path")
    parser.add_argument("--labels", required=True, help="labels.txt path (one per line)")
    parser.add_argument("--cam", type=int, default=0)
    args = parser.parse_args()
    run_inference(args.model, args.labels, args.cam)