import base64
import gc
from io import BytesIO
import threading

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from runtime_config import MAX_MODEL_SIDE
from detection import AI_LOCK, detect_objects, draw_trash_boxes
from enhancement import enhance_image, get_device
from segmentation import segment_image
from explainability import generate_explainability_heatmap
from analytics import compute_comprehensive_analytics
from auth import router as auth_router

app = FastAPI(title="Aqua Scope API")
app.include_router(auth_router)

# Explicit origins are used instead of "*" with credentials.
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://capstone67.vercel.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 15 * 1024 * 1024


@app.get("/")
def home():
    return {
        "message": "Aqua Scope backend is running",
        "models": {
            "enhancement": "NSGA-Net / NU2Net",
            "detection": "YOLOv8",
            "segmentation": "YOLO segmentation",
            "explainability": "Multi-scale YOLO feature activation heatmaps",
        },
        "features": [
            "Image Enhancement", "Pollution Detection", "Trash Segmentation",
            "Grad-CAM Attention Heatmaps", "Total Object Count & Census",
            "Marine Pollution Severity Index (PSI)",
            "Water Quality & Clarity Gain Assessment",
            "Debris Surface Area Coverage Percentage",
        ],
    }


@app.get("/health")
def health():
    return {"status": "healthy", "app": "Aqua Scope", "device": get_device()}


async def read_image(file: UploadFile):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")
    contents = await file.read(MAX_FILE_SIZE + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image size must not exceed 15 MB.")

    image_array = np.frombuffer(contents, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    del image_array, contents
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Could not read this image. Try a JPG or PNG file.")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    del image_bgr
    return image_rgb


def image_to_data_url(image_rgb):
    if image_rgb is None:
        return None
    # Keep PNG data URLs for frontend compatibility.
    buffer = BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG", optimize=False)
    encoded = base64.b64encode(buffer.getbuffer()).decode("ascii")
    buffer.close()
    return f"data:image/png;base64,{encoded}"


def run_segmentation_if_needed(image_rgb, trash_detections):
    if not trash_detections:
        return {
            "segmentation_image": None,
            "segments": [],
            "segmentation_status": "skipped_no_trash_detected",
            "segmentation_error": None,
        }
    try:
        overlay, segments = segment_image(image_rgb)
        data_url = image_to_data_url(overlay)
        del overlay
        gc.collect()
        return {
            "segmentation_image": data_url,
            "segments": segments,
            "segmentation_status": "completed",
            "segmentation_error": None,
        }
    except Exception as exc:
        return {
            "segmentation_image": None,
            "segments": [],
            "segmentation_status": "failed",
            "segmentation_error": str(exc),
        }


def run_explainability(image_rgb, trash_detections):
    try:
        overlay, raw_heat, metrics = generate_explainability_heatmap(image_rgb, trash_detections)
        overlay_url = image_to_data_url(overlay)
        raw_url = image_to_data_url(raw_heat)
        del overlay, raw_heat
        gc.collect()
        return {
            "explainability_heatmap": overlay_url,
            "explainability_raw_heatmap": raw_url,
            "explainability_metrics": metrics,
            "explainability_status": "completed",
            "explainability_error": None,
        }
    except Exception as exc:
        return {
            "explainability_heatmap": None,
            "explainability_raw_heatmap": None,
            "explainability_metrics": None,
            "explainability_status": "failed",
            "explainability_error": str(exc),
        }


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    image_rgb = await read_image(file)
    h, w = image_rgb.shape[:2]
    return {
        "success": True, "message": "Image received successfully.",
        "filename": file.filename, "content_type": file.content_type,
        "size_bytes": file.size, "width": int(w), "height": int(h),
    }


@app.post("/enhance")
async def enhance_endpoint(file: UploadFile = File(...)):
    image_rgb = await read_image(file)
    try:
        with AI_LOCK:
            enhanced_rgb = enhance_image(image_rgb)
        result = {
            "success": True, "message": "Image enhancement completed.",
            "original_image": image_to_data_url(image_rgb),
            "enhanced_image": image_to_data_url(enhanced_rgb),
            "device": get_device(),
        }
        del enhanced_rgb, image_rgb
        gc.collect()
        return result
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Image enhancement failed: {exc}") from exc


@app.post("/detect")
async def detect_endpoint(file: UploadFile = File(...)):
    image_rgb = await read_image(file)
    try:
        with AI_LOCK:
            trash, non_trash = detect_objects(image_rgb)
            detection_image = draw_trash_boxes(image_rgb, trash)
        result = {
            "success": True,
            "message": "Trash detected." if trash else "No trash detected by the model.",
            "trash_detections": trash, "non_trash_objects": non_trash,
            "trash_detected": bool(trash),
            "detection_image": image_to_data_url(detection_image),
        }
        del image_rgb, detection_image
        gc.collect()
        return result
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Pollution detection failed: {exc}") from exc


@app.post("/segment")
async def segment_endpoint(file: UploadFile = File(...)):
    image_rgb = await read_image(file)
    try:
        with AI_LOCK:
            overlay, segments = segment_image(image_rgb)
        result = {
            "success": True, "message": "Segmentation completed.",
            "segmentation_image": image_to_data_url(overlay), "segments": segments,
        }
        del image_rgb, overlay
        gc.collect()
        return result
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Segmentation failed: {exc}") from exc


@app.post("/explain")
async def explain_endpoint(file: UploadFile = File(...)):
    image_rgb = await read_image(file)
    try:
        with AI_LOCK:
            trash, non_trash = detect_objects(image_rgb)
            explainability_result = run_explainability(image_rgb, trash)
        result = {
            "success": True,
            "message": (
                "Visual explanation generated using Grad-CAM."
                if explainability_result["explainability_status"] == "completed"
                else "Visual explanation could not be generated."
            ),
            "trash_detections": trash, "non_trash_objects": non_trash,
            "trash_detected": bool(trash), **explainability_result,
        }
        del image_rgb
        gc.collect()
        return result
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Explainability failed: {exc}") from exc


@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    image_rgb = await read_image(file)

    try:
        # Serialize the complete pipeline. One request at a time avoids overlapping
        # NU2Net/YOLO activation peaks on a 512 MB instance.
        with AI_LOCK:
            enhanced_rgb = enhance_image(image_rgb)
            trash, non_trash = detect_objects(image_rgb)
            original_detection = draw_trash_boxes(image_rgb, trash)
            enhanced_detection = draw_trash_boxes(enhanced_rgb, trash)

            segmentation_result = run_segmentation_if_needed(image_rgb, trash)
            gc.collect()

            explainability_result = run_explainability(image_rgb, trash)
            gc.collect()

            analytics_result = compute_comprehensive_analytics(
                image_rgb, enhanced_rgb, trash, non_trash,
                segmentation_result.get("segments", []),
            )

        # Encode and return the original-resolution outputs.
        result = {
            "success": True,
            "message": (
                "Enhancement, detection, segmentation, visual explanation, and analytics completed."
                if segmentation_result["segmentation_status"] == "completed"
                and explainability_result["explainability_status"] == "completed"
                else "Image analysis completed. Check each stage status for details."
            ),
            "original_image": image_to_data_url(image_rgb),
            "enhanced_image": image_to_data_url(enhanced_rgb),
            "original_detection_image": image_to_data_url(original_detection),
            "enhanced_detection_image": image_to_data_url(enhanced_detection),
            "trash_detections": trash,
            "non_trash_objects": non_trash,
            "trash_detected": bool(trash),
            "device": get_device(),
            "analytics": analytics_result,
            **segmentation_result,
            **explainability_result,
        }

        del image_rgb, enhanced_rgb, original_detection, enhanced_detection
        gc.collect()
        return result

    except Exception as exc:
        gc.collect()
        raise HTTPException(
            status_code=503,
            detail=f"Image analysis failed. Check model files and dependencies. Details: {exc}",
        ) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
