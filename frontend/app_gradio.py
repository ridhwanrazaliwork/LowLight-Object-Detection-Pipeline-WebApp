import gradio as gr
import requests
import base64
from PIL import Image
from io import BytesIO
import pandas as pd
import os

# Backend API URL - use env var or default to localhost
API_URL = os.getenv('API_URL', 'http://localhost:8000')

# State to store image paths between steps
state = {
    "uploaded_path": None,
    "denoised_path": None,
    "enhanced_path": None
}

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image"""
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))

def upload_image(image):
    """Upload image to backend"""
    if image is None:
        return None, "Please upload an image first"
    
    # Convert PIL image to bytes
    img_bytes = BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Upload to backend
    files = {'file': ('uploaded_image.png', img_bytes.getvalue(), 'image/png')}
    try:
        response = requests.post(f"{API_URL}/upload", files=files, timeout=30)
        if response.status_code == 200:
            data = response.json()
            state["uploaded_path"] = data["path"]
            return image, f"Uploaded: {data['filename']}"
        else:
            return None, f"Upload failed: {response.text}"
    except Exception as e:
        return None, f"Error: {str(e)}"

def denoise_image():
    """Denoise the uploaded image"""
    if state["uploaded_path"] is None:
        return None, None, "Please upload an image first"
    
    try:
        response = requests.post(
            f"{API_URL}/denoise",
            params={"image_path": state["uploaded_path"]},
            timeout=180
        )
        
        if response.status_code == 200:
            data = response.json()
            state["denoised_path"] = data["paths"]["denoised"]
            
            original_img = base64_to_image(data["original_base64"])
            denoised_img = base64_to_image(data["denoised_base64"])
            
            return original_img, denoised_img, "Denoising completed!"
        else:
            return None, None, f"Error: {response.text}"
    except Exception as e:
        return None, None, f"Error: {str(e)}"

def enhance_image():
    """Enhance the denoised image"""
    if state["denoised_path"] is None:
        return None, None, "Please run denoiser first"
    
    try:
        response = requests.post(
            f"{API_URL}/enhance",
            params={"image_path": state["denoised_path"]},
            timeout=180
        )
        
        if response.status_code == 200:
            data = response.json()
            state["enhanced_path"] = data["paths"]["enhanced"]
            
            original_img = base64_to_image(data["original_base64"])
            enhanced_img = base64_to_image(data["enhanced_base64"])
            
            return original_img, enhanced_img, "Enhancement completed!"
        else:
            return None, None, f"Error: {response.text}"
    except Exception as e:
        return None, None, f"Error: {str(e)}"

def detect_objects():
    """Run YOLO detection on enhanced image"""
    if state["enhanced_path"] is None:
        return None, None, "Please run enhancer first"
    
    try:
        response = requests.post(
            f"{API_URL}/detect",
            params={"image_path": state["enhanced_path"]},
            timeout=180
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Get detected image
            detected_img = base64_to_image(data["image_base64"])
            
            # Create detections table
            if data["detection_count"] > 0:
                detections = data["detections"]
                table_data = []
                for det in detections:
                    coords = det["coordinates"]
                    table_data.append({
                        "Class": det["class_name"],
                        "Confidence": f"{det['confidence']*100:.1f}%",
                        "X1": f"{coords[0]:.0f}",
                        "Y1": f"{coords[1]:.0f}",
                        "X2": f"{coords[2]:.0f}",
                        "Y2": f"{coords[3]:.0f}"
                    })
                df = pd.DataFrame(table_data)
                message = f"Detection completed! Found {data['detection_count']} object(s)"
            else:
                df = pd.DataFrame()
                message = "Detection completed! No objects found"
            
            return detected_img, df, message
        else:
            return None, None, f"Error: {response.text}"
    except Exception as e:
        return None, None, f"Error: {str(e)}"

# ==================== Gradio Interface ====================

with gr.Blocks(title="YOLO Image Ridhwan Pipeline") as demo:
    gr.Markdown("# 🔍 YOLO Image Ridhwan Pipeline")
    gr.Markdown("### Denoise → Enhance → Detect Objects")
    
    with gr.Column():
        # Upload Section
        gr.Markdown("## Step 0: Upload Image")
        with gr.Row():
            input_image = gr.Image(label="Upload Image", type="pil")
            upload_btn = gr.Button("Upload", scale=1)
        upload_status = gr.Textbox(label="Status", interactive=False)
        
        gr.Markdown("---")
        
        # Denoise Section
        gr.Markdown("## Step 1: Denoising (FFDNet)")
        denoise_btn = gr.Button("Run Denoiser", variant="primary")
        with gr.Row():
            denoise_original = gr.Image(label="Before (Original)", type="pil")
            denoise_result = gr.Image(label="After (Denoised)", type="pil")
        denoise_status = gr.Textbox(label="Status", interactive=False)
        
        gr.Markdown("---")
        
        # Enhance Section
        gr.Markdown("## Step 2: Enhancement (Zero-DCE++)")
        enhance_btn = gr.Button("Run Enhancer", variant="primary")
        with gr.Row():
            enhance_original = gr.Image(label="Before (Denoised)", type="pil")
            enhance_result = gr.Image(label="After (Enhanced)", type="pil")
        enhance_status = gr.Textbox(label="Status", interactive=False)
        
        gr.Markdown("---")
        
        # YOLO Section
        gr.Markdown("## Step 3: Object Detection (YOLO)")
        detect_btn = gr.Button("Run YOLO Detection", variant="primary")
        detect_image_output = gr.Image(label="Detection Results", type="pil")
        detect_table = gr.Dataframe(label="Detections", interactive=False)
        detect_status = gr.Textbox(label="Status", interactive=False)
    
    # Button interactions
    upload_btn.click(
        upload_image,
        inputs=[input_image],
        outputs=[input_image, upload_status]
    )
    
    denoise_btn.click(
        denoise_image,
        outputs=[denoise_original, denoise_result, denoise_status]
    )
    
    enhance_btn.click(
        enhance_image,
        outputs=[enhance_original, enhance_result, enhance_status]
    )
    
    detect_btn.click(
        detect_objects,
        outputs=[detect_image_output, detect_table, detect_status]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
