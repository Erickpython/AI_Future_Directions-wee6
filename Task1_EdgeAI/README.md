# Edge AI Prototype — Recyclable Item Classifier

## Project summary
Train a lightweight image classifier to recognize recyclable items (e.g., plastic, glass, metal, paper, cardboard). Convert to TensorFlow Lite and deploy to a Raspberry Pi for low-latency, on-device inference.

## Dataset
- Source: user-provided dataset arranged into `train/ val/ test/` subfolders.
- Classes: [list your classes, e.g., plastic, glass, metal, cardboard, paper]
- Image size used: 160×160
- Train/Val/Test split: use provided folders; typical split 70/15/15.

## Model
- Architecture: MobileNetV2 (alpha=0.35) top + small classification head.
- Total params: (varies) — typically ~100k–500k for alpha=0.35.
- Input: 160×160×3.

## Training procedure
- Initial training: base frozen, Adam lr=1e-3, 15 epochs.
- Fine-tuning: unfreeze last layers, lr=1e-4, 5 epochs.
- Augmentation: random flip, small rotation, small zoom.
- Early stopping and best-model checkpoint used.

## Conversion to TFLite
- Conversion: TensorFlow Lite converter.
- Quantization: float16 or post-training integer quantization (recommended on Pi for speed and smaller size).
- Typical TFLite sizes:
  - float32 TF model (`final_model.h5`): ~5–25 MB (depends on head)
  - TFLite float16: often 1–8 MB
  - Integer quantized: can be ~1–4 MB

## Sample results (example)
> These are **example** results to illustrate what to expect. Your real metrics depend on dataset size/quality and augmentation.

- Test accuracy: **~0.85 – 0.92**
- Per-class F1-scores (example):
  - plastic: 0.88
  - glass: 0.84
  - metal: 0.86
  - paper: 0.90
  - cardboard: 0.89
- Confusion matrix: (see `model_output/confusion_*.png`)

## Performance on Raspberry Pi (example)
- Device: Raspberry Pi 4 (4GB)
- Inference time per frame: **~30–100 ms** (depends on quantization and whether using tflite-runtime + NNAPI/Edge TPU)
- CPU usage: light to moderate. On-device inference avoids network latency.

## Deployment steps (summary)
1. Train model in Colab or desktop using `EdgeAI_recyclables_train_and_tflite.py`.
2. Convert to TFLite with `convert_to_tflite(...)`. Use integer quantization for smallest model & best CPU speed (needs representative dataset).
3. Copy `model_fp16.tflite` (or int8 tflite) and `labels.txt` to Raspberry Pi.
4. Install tflite runtime on Pi: `pip3 install tflite-runtime` (choose proper wheel).
5. Copy `rpi_infer_tflite.py` and run:  
   `python3 rpi_infer_tflite.py --model model_fp16.tflite --labels labels.txt`
6. (Optional) For extreme speed, compile for and run on Google Coral Edge TPU — requires compiling TFLite for Edge TPU and hardware.

## Notes & recommendations
- Representative dataset for quantization: include diverse samples for each class; this improves integer quantization accuracy.
- If inference speed is too slow on Pi CPU, use Coral Edge TPU or a small NCS2 stick, or optimize model further (smaller alpha, smaller input size).
- Monitor class imbalance; use class weights or oversampling if needed.

## Files produced by pipeline
- `final_model.h5` — Keras model
- `model_fp16.tflite` — TFLite model (float16 quantized example)
- `classification_report_*.json` — per-class metrics
- `confusion_*.png` — confusion matrix