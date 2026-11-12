import cv2
import numpy as np

from detection import (
    TRASH_CLASSES,
    YOLO_LOCK,
    get_detection_model,
    resize_for_model,
)
from runtime_config import INFERENCE_DEVICE, YOLO_IMAGE_SIZE


def get_segmentation_model():
    # best.pt was verified locally to report model.task == "segment".
    # Reuse the exact same loaded model object as detection/explainability.
    model = get_detection_model()
    if model.task != "segment":
        raise ValueError(f"Expected a YOLO segmentation model, got: {model.task}")
    return model


def segment_image(image_rgb):
    if (
        not isinstance(image_rgb, np.ndarray)
        or image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
        or image_rgb.size == 0
    ):
        raise ValueError("Expected a valid RGB image.")

    model_input, scale = resize_for_model(image_rgb)
    model = get_segmentation_model()

    with YOLO_LOCK:
        result = model.predict(
            source=model_input,
            imgsz=YOLO_IMAGE_SIZE,
            conf=0.25,
            device=INFERENCE_DEVICE,
            verbose=False,
            half=False,
            max_det=100,
        )[0]

    segmented_rgb = image_rgb.copy()
    segments = []

    if result.masks is None or result.boxes is None:
        del result
        return segmented_rgb, segments

    try:
        masks = result.masks.data.cpu().numpy()
        for index, mask in enumerate(masks):
            class_id = int(result.boxes.cls[index].item())
            class_name = model.names[class_id]
            confidence = float(result.boxes.conf[index].item())
            if class_name not in TRASH_CLASSES:
                continue

            # Masks are in model-input space; restore original image dimensions.
            if scale != 1.0:
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (image_rgb.shape[1], image_rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            elif mask.shape[:2] != image_rgb.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (image_rgb.shape[1], image_rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            binary_mask = mask > 0.5
            if not np.any(binary_mask):
                continue

            # Blend directly into the output; avoid a second full-size overlay copy.
            segmented_rgb[binary_mask] = (
                segmented_rgb[binary_mask].astype(np.float32) * 0.6
                + np.array([255, 0, 0], dtype=np.float32) * 0.4
            ).astype(np.uint8)

            contours, _ = cv2.findContours(
                binary_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(segmented_rgb, contours, -1, (0, 0, 0), 2)

            segments.append({
                "class_name": class_name,
                "confidence": round(confidence, 4),
            })
        del masks
    finally:
        del result

    return segmented_rgb, segments
