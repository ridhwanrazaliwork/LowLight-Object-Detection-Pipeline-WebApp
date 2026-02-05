# YOLO Web App — Low-Light Object Detection Pipeline

What this project does
- Lightweight web app that runs a low-light object-detection pipeline:
  - Upload an image
  - Denoise
  - Enhance
  - Run YOLO detection
  - Return annotated results (image + detections)

Key files
- `app_gradio.py` — Gradio frontend for uploading images and viewing results
- `backend.py` — Backend orchestrator / API (processes images)
- `denoise_wrapper.py` — Glue code for denoiser models
- `models/` or `best.pt` — Detection model weights
- `requirements.txt` — Python dependencies


Quick start (local)
1. Create and activate a virtualenv:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Run the frontend (Gradio):
```bash
python app_gradio.py
```
3. (Option) Run backend directly if needed:
```bash
python backend.py
```

Pipeline flow
- The frontend sends the uploaded image to the processing pipeline.
- `denoise_wrapper.py` runs the denoiser on the image.
- The enhancer (if present) further processes the image.
- The processed image is fed to the YOLO model for detection.
- The app returns the annotated image and detection metadata to the user.

Docker (local containerized run)
This project includes Docker so you can run frontend and backend in containers.

Build and run with Docker Compose
```bash
docker-compose build
docker-compose up
```

Notes
- Place detection weights in `models/` or at project root as `best.pt`.
- `requirements.txt` reproduces the Python environment used for development.


### Models training

**Dataset preparation**

- Stratified sampling of the ExDark dataset to preserve class distribution across train/val/test splits (suggested: 80/10/10).

**Pre-processing (Hybrid stage)**

1. Denoise: apply FFDNet to reduce sensor noise.
2. Enhance: apply Zero-DCE or DALE to recover brightness and contrast.

- Save the enhanced outputs in a separate dataset folder (e.g. `data/enhanced/`) and keep original images for comparisons and ablation studies.

**Detection training (Strategy)**

- Model: YOLOv8-Medium (`yolov8m`) — good balance of speed and accuracy.
- Training strategy: Memory Anchor (layer freezing of early backbone layers) + Mosaic augmentation.
- Recommended freeze: `freeze=9` to keep the backbone's pre-trained COCO features stable when fine-tuning on low-light images.

**Environment**

Install dependencies (example):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install ultralytics
```

Ensure your enhancement model weights (FFDNet / Zero-DCE / DALE) are available locally and referenced by your denoiser/enhancer scripts.

**Execution (training)**

Download base weights (optional):

```bash
# Download weights manually (or let Ultralytics fetch them)
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt
```

Run training with the Ultralytics CLI:

Default command to start training
```bash
yolo task=detect mode=train model=yolov8m.pt data=custom_exdark.yaml epochs=100 imgsz=640 batch=16 freeze=9
```

Our optimize command below (adjust the batch accordingly based on your gpu memory):

```bash
yolo task=detect mode=train \model=yolov8m.pt \data=original_data.yaml \epochs=100 patience=10 \batch=32 imgsz=640 workers=4 \optimizer=AdamW cos_lr=true \freeze=9 lr0=0.0005 momentum=0.95 weight_decay=0.0008 warmup_epochs=5.0 \box=9.0 cls=0.3 kobj=1.5 \hsv_h=0.015 hsv_s=0.7 hsv_v=0.8 \degrees=10.0 \translate=0.2 \scale=0.9 \mixup=0.4 \copy_paste=0.3 \mosaic=1.0 close_mosaic=10 \cache=ram amp=true
```

- `data=custom_exdark.yaml` should point to your dataset YAML describing paths and class names.
- Mosaic augmentation is enabled by default in the Ultralytics training pipeline — keep it on for low-light occlusion robustness.

Tips:

- Start with frozen backbone (`freeze=9`) to preserve learned low-level features. After initial convergence, consider unfreezing and fine-tuning with a reduced learning rate.
- Use early stopping, model checkpointing, and learning-rate schedules for stable training.
- Track metrics (Precision, Recall, mAP) on the validation split and visualize detections for human-in-the-loop verification.

**Key results**

| Configuration | Precision | Recall | mAP@50 |
|---|---:|---:|---:|
| Baseline (untuned) | 0.770 | 0.624 | 0.707 |
| Ours (Freeze + Mosaic) | 0.814 | 0.755 | 0.837 |

**References**

- Ultralytics YOLO CLI docs: https://docs.ultralytics.com/

