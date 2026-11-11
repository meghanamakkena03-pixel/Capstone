
import cv2
import numpy as np


def generate_gradcam(image_rgb, detections=None):
    """
    Generate a visual explanation of detected trash objects.

    Note:
        This creates a detection-based visual explanation using YOLO
        bounding boxes. It is not a true Grad-CAM activation heatmap.
    """

    if image_rgb is None:
        raise ValueError("Input image is empty.")

    image_rgb = np.asarray(image_rgb)

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with 3 channels.")

    explanation = image_rgb.copy()
    height, width = explanation.shape[:2]

    if not detections:
        return explanation

    for detection in detections:
        if not isinstance(detection, dict):
            continue

        box = detection.get("box")

        if box is None or len(box) != 4:
            continue

        x1, y1, x2, y2 = map(int, box)

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        class_name = str(
            detection.get("class_name", "trash_object")
        )
        confidence = float(detection.get("confidence", 0.0))

        readable_name = (
            class_name.replace("trash_", "")
            .replace("_", " ")
            .title()
        )

        label = f"{readable_name} {confidence:.0%}"

        # Add a translucent highlight over the detected region.
        overlay = explanation.copy()

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (255, 40, 40),
            thickness=-1,
        )

        explanation = cv2.addWeighted(
            overlay,
            0.22,
            explanation,
            0.78,
            0,
        )

        # Draw the bounding box.
        cv2.rectangle(
            explanation,
            (x1, y1),
            (x2, y2),
            (255, 40, 40),
            thickness=3,
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness,
        )

        label_top = max(0, y1 - text_height - baseline - 10)
        label_right = min(width - 1, x1 + text_width + 10)

        cv2.rectangle(
            explanation,
            (x1, label_top),
            (label_right, label_top + text_height + baseline + 8),
            (255, 40, 40),
            thickness=-1,
        )

        cv2.putText(
            explanation,
            label,
            (x1 + 5, label_top + text_height + 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return explanation
