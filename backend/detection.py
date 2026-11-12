from pathlib import Path
import threading

import cv2
import numpy as np
from ultralytics import YOLO

from runtime_config import INFERENCE_DEVICE, MAX_MODEL_SIDE, YOLO_IMAGE_SIZE

MODEL_PATH = Path(__file__).resolve().parent / "models" / "best.pt"

TRASH_CLASSES = {
    "trash_clothing", "trash_pipe", "trash_bottle", "trash_bag",
    "trash_snack_wrapper", "trash_can", "trash_cup", "trash_container",
    "trash_unknown_instance", "trash_branch", "trash_wreckage",
    "trash_tarp", "trash_rope", "trash_net",
}
NON_TRASH_CLASSES = {
    "rov", "plant", "animal_fish", "animal_starfish", "animal_shells",
    "animal_crab", "animal_eel", "animal_etc",
}

# One model instance is shared by detection, segmentation and explainability.
_yolo_model = None
YOLO_LOCK = threading.RLock()
AI_LOCK = threading.RLock()


def resize_for_model(image_rgb, max_side=MAX_MODEL_SIDE):
    h, w = image_rgb.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale >= 1.0:
        return image_rgb, 1.0
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_AREA), scale


def get_detection_model():
    global _yolo_model
    with YOLO_LOCK:
        if _yolo_model is None:
            if not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size == 0:
                raise FileNotFoundError(f"YOLO model weights not found or empty: {MODEL_PATH}")
            model = YOLO(str(MODEL_PATH))
            if model.task != "segment":
                raise ValueError(
                    f"backend/models/best.pt must be a YOLO segmentation checkpoint; found task={model.task!r}"
                )
            _yolo_model = model
        return _yolo_model


def detect_objects(image_rgb):
    model_input, scale = resize_for_model(image_rgb)
    model = get_detection_model()

    with YOLO_LOCK:
        results = model.predict(
            source=model_input,
            imgsz=YOLO_IMAGE_SIZE,
            conf=0.10,
            verbose=False,
            device=INFERENCE_DEVICE,
            half=False,
            max_det=100,
        )

    trash_detections = []
    non_trash_names = []
    inv_scale = 1.0 / scale

    try:
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = model.names[class_id]

                if class_name in TRASH_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    # Return coordinates in the original uploaded image space.
                    x1 = max(0, min(image_rgb.shape[1], int(round(x1 * inv_scale))))
                    x2 = max(0, min(image_rgb.shape[1], int(round(x2 * inv_scale))))
                    y1 = max(0, min(image_rgb.shape[0], int(round(y1 * inv_scale))))
                    y2 = max(0, min(image_rgb.shape[0], int(round(y2 * inv_scale))))
                    trash_detections.append({
                        "class_name": class_name,
                        "confidence": confidence,
                        "box": (x1, y1, x2, y2),
                    })
                elif class_name in NON_TRASH_CLASSES and class_name not in non_trash_names:
                    non_trash_names.append(class_name)
    finally:
        # Release Ultralytics result tensors promptly.
        del results

    return trash_detections, non_trash_names


def draw_trash_boxes(image_rgb, detections):
    output = image_rgb.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 255), 3)
        label = f"TRASH: {class_name} {confidence:.2f}"
        cv2.putText(
            output, label, (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA
        )
    return output
