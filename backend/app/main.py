import os
import sys
import shutil
import tempfile
import base64
import io
import json
from pathlib import Path
from typing import Optional, Dict, List

import torch
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# Get the directory of this file and resolve paths relative to it
app_dir = Path(__file__).parent  # /app/app/
backend_dir = app_dir.parent     # /app
root_dir = backend_dir            # /app - CORRECT!

# Add paths for denoiser and enhancer imports
sys.path.insert(0, str(backend_dir / 'denoiser' / 'ffdnet'))
sys.path.insert(0, str(backend_dir / 'enhancer' / 'zerodce'))

def log_debug(msg):
    """Print debug message with flush"""
    print(f"[BACKEND DEBUG] {msg}", flush=True)
    sys.stderr.flush()
    sys.stdout.flush()

# Initialize FastAPI app
app = FastAPI(title="YOLO Processing Pipeline")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models (loaded at startup)
yolo_model = None
temp_base_dir = root_dir / 'temp'

# ==================== Model Loading ====================

@app.on_event("startup")
async def load_models():
    """Load models at startup"""
    global yolo_model
    log_debug("Loading YOLO model...")
    yolo_model = YOLO(str(root_dir / 'models' / 'best.pt'))
    log_debug("YOLO model loaded successfully!")

# ==================== Utility Functions ====================

def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        log_debug(f"Converting to base64: {image_path}")
        with open(image_path, 'rb') as img_file:
            data = base64.b64encode(img_file.read()).decode('utf-8')
            log_debug(f"Base64 conversion successful, length: {len(data)}")
            return data
    except Exception as e:
        log_debug(f"Base64 conversion failed: {str(e)}")
        raise

def cleanup_temp_folder(folder_path: str):
    """Delete a folder and its contents"""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

# ==================== Wrapper Functions ====================

def process_denoise(input_image_path: str, output_folder: str) -> Dict[str, str]:
    """
    Denoise image using FFDNet via wrapper script
    
    Input: Image file path
    Output: Returns dict with original and denoised image paths
    """
    import subprocess
    
    try:
        log_debug(f"Starting denoise process for: {input_image_path}")
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Prepare output path
        image_name = os.path.basename(input_image_path)
        output_path = os.path.join(output_folder, image_name)
        
        log_debug(f"Output will be saved to: {output_path}")
        
        # Call denoise wrapper script
        wrapper_script = str(backend_dir / 'denoise_wrapper.py')
        python_exe = sys.executable
        
        if not os.path.exists(wrapper_script):
            raise FileNotFoundError(f"Wrapper script not found: {wrapper_script}")
        
        if not os.path.exists(input_image_path):
            raise FileNotFoundError(f"Input image not found: {input_image_path}")
        
        log_debug(f"Calling subprocess: {python_exe} {wrapper_script} {input_image_path} {output_path}")
        
        result = subprocess.run(
            [python_exe, wrapper_script, input_image_path, output_path],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        log_debug(f"Subprocess returned with code: {result.returncode}")
        log_debug(f"Subprocess stdout: {result.stdout}")
        if result.stderr:
            log_debug(f"Subprocess stderr: {result.stderr}")
        
        if result.returncode != 0:
            raise Exception(f"Denoiser failed with return code {result.returncode}: {result.stderr}")
        
        if not os.path.exists(output_path):
            raise Exception(f"Denoised image not found at {output_path}")
        
        log_debug(f"Denoise completed successfully")
        
        return {
            "original": input_image_path,
            "denoised": output_path
        }
        
    except subprocess.TimeoutExpired as e:
        log_debug(f"Denoiser timed out after 120 seconds")
        raise Exception("Denoiser timed out")
    except Exception as e:
        log_debug(f"Error in denoiser: {str(e)}")
        import traceback
        log_debug(traceback.format_exc())
        raise

def process_enhance(input_image_path: str, output_folder: str) -> Dict[str, str]:
    """
    Enhance image using Zero-DCE++
    
    Input: Image file path
    Output: Returns dict with original and enhanced image paths
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Create temp test_data folder for enhancer
    temp_test_data = str(backend_dir / 'enhancer' / 'zerodce' / 'data' / 'test_data')
    os.makedirs(temp_test_data, exist_ok=True)
    
    # Copy input image to temp folder
    image_name = os.path.basename(input_image_path)
    temp_input = os.path.join(temp_test_data, image_name)
    shutil.copy(input_image_path, temp_input)
    
    # Change to enhancer directory and run
    enhancer_dir = str(backend_dir / 'enhancer' / 'zerodce')
    old_cwd = os.getcwd()
    
    try:
        os.chdir(enhancer_dir)
        
        # Import and run lowlight function
        import lowlight_test
        lowlight_test.lowlight(f'data/test_data/{image_name}', output_folder='webapp_enhanced')
        
        # Read enhanced output
        enhanced_output_dir = os.path.join(enhancer_dir, 'data/webapp_enhanced')
        enhanced_image_path = os.path.join(enhanced_output_dir, image_name)
        
        # Copy to output folder
        output_path = os.path.join(output_folder, image_name)
        shutil.copy(enhanced_image_path, output_path)
        
        # Cleanup temp
        cleanup_temp_folder(temp_test_data)
        cleanup_temp_folder(enhanced_output_dir)
        
        return {
            "original": input_image_path,
            "enhanced": output_path
        }
        
    finally:
        os.chdir(old_cwd)

def process_yolo(input_image_path: str, output_folder: str) -> Dict:
    """
    Run YOLO detection on image
    
    Input: Image file path
    Output: Returns dict with image path and detections
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Run YOLO inference
    results = yolo_model.predict(source=input_image_path, conf=0.5)
    result = results[0]
    
    # Draw boxes on image
    result_image = result.plot()
    
    # Save result
    image_name = os.path.basename(input_image_path)
    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, result_image)
    
    # Extract detections
    detections = []
    for box in result.boxes:
        detections.append({
            "class": int(box.cls.item()),
            "class_name": result.names[int(box.cls.item())],
            "confidence": float(box.conf.item()),
            "coordinates": box.xyxy.tolist()[0]
        })
    
    return {
        "image": output_path,
        "detections": detections,
        "detection_count": len(detections)
    }

# ==================== FastAPI Endpoints ====================

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload and save image to temp/input/"""
    try:
        upload_folder = temp_base_dir / "input"
        os.makedirs(upload_folder, exist_ok=True)
        
        # Save uploaded file
        file_path = upload_folder / file.filename
        content = await file.read()
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "path": str(file_path)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/denoise")
async def denoise_endpoint(image_path: str):
    """Denoise image"""
    try:
        print(f"[DEBUG] Denoising image: {image_path}")
        output_folder = temp_base_dir / "denoised"
        result = process_denoise(image_path, str(output_folder))
        
        print(f"[DEBUG] Denoised paths: {result}")
        print(f"[DEBUG] Denoised file exists: {os.path.exists(result['denoised'])}")
        
        return JSONResponse({
            "status": "success",
            "original_base64": image_to_base64(result["original"]),
            "denoised_base64": image_to_base64(result["denoised"]),
            "paths": result
        })
    except Exception as e:
        print(f"[ERROR] Denoise failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/enhance")
async def enhance_endpoint(image_path: str):
    """Enhance image"""
    try:
        output_folder = temp_base_dir / "enhanced"
        result = process_enhance(image_path, str(output_folder))
        
        return JSONResponse({
            "status": "success",
            "original_base64": image_to_base64(result["original"]),
            "enhanced_base64": image_to_base64(result["enhanced"]),
            "paths": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect")
async def detect_endpoint(image_path: str):
    """Run YOLO detection"""
    try:
        output_folder = temp_base_dir / "detected"
        result = process_yolo(image_path, str(output_folder))
        
        return JSONResponse({
            "status": "success",
            "image_base64": image_to_base64(result["image"]),
            "detections": result["detections"],
            "detection_count": result["detection_count"],
            "paths": {
                "image": result["image"]
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "yolo_model_loaded": yolo_model is not None
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
