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
