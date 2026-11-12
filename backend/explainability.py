import gc
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from detection import get_detection_model, YOLO_LOCK
from runtime_config import INFERENCE_DEVICE, EXPLAIN_IMAGE_SIZE


def compute_feature_heatmaps(image_rgb, target_layers=None):
    """
    Multi-scale feature-activation heatmap from the trained YOLO model.
    Uses the existing shared YOLO instance and a bounded working image.
    """
    if target_layers is None:
        target_layers = [15, 18, 21]

    model_wrapper = get_detection_model()
    model = model_wrapper.model.to(INFERENCE_DEVICE).eval()
    h, w = image_rgb.shape[:2]

    scale = min(1.0, float(EXPLAIN_IMAGE_SIZE) / max(h, w))
    if scale < 1.0:
        small_w = max(32, int(round(w * scale / 32)) * 32)
        small_h = max(32, int(round(h * scale / 32)) * 32)
        small = cv2.resize(image_rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
    else:
        small = image_rgb

    tensor = torch.from_numpy(
        np.ascontiguousarray(small.transpose(2, 0, 1)[None])
    ).float().div_(255.0)

    activations = {}
    handles = []

    def make_hook(layer_id):
        def hook_fn(_module, _inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            # Keep only the captured feature tensor; no graph is retained.
            activations[layer_id] = value.detach().to("cpu")
        return hook_fn

    try:
        with YOLO_LOCK:
            for layer_id in target_layers:
                if layer_id < len(model.model):
                    handles.append(
                        model.model[layer_id].register_forward_hook(make_hook(layer_id))
                    )
            with torch.inference_mode():
                _ = model(tensor)
    finally:
        for handle in handles:
            handle.remove()
        del tensor, small
        gc.collect()

    if not activations:
        return np.zeros((h, w), dtype=np.float32)

    weights = {15: 0.20, 18: 0.35, 21: 0.45}
    fused = None

    for layer_id, activation in activations.items():
        # Channel-mean activation is a stable low-memory feature saliency projection.
        feature = activation.float().squeeze(0)
        if feature.ndim != 3 or feature.shape[1] < 2 or feature.shape[2] < 2:
            del feature, activation
            continue

        projection = feature.mean(dim=0, keepdim=True).unsqueeze(0)
        up = F.interpolate(
            projection, size=(h, w), mode="bilinear", align_corners=False
        ).squeeze().numpy()

        up -= float(up.min())
        up /= float(up.max()) + 1e-8
        weighted = up * weights.get(layer_id, 1.0 / max(len(activations), 1))
        fused = weighted if fused is None else fused + weighted

        del feature, projection, up, weighted, activation
        gc.collect()

    activations.clear()
    if fused is None:
        return np.zeros((h, w), dtype=np.float32)

    fused -= float(fused.min())
    fused /= float(fused.max()) + 1e-8
    return fused.astype(np.float32, copy=False)


def generate_explainability_heatmap(image_rgb, detections=None):
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Input image is empty.")
    image_rgb = np.asarray(image_rgb)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with 3 channels.")

    h, w = image_rgb.shape[:2]
    cam = compute_feature_heatmaps(image_rgb)

    if detections:
        det_mask = np.zeros((h, w), dtype=np.float32)
        for detection in detections:
            box = detection.get("box")
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = map(int, box)
            x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w))
            y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue

            confidence = float(detection.get("confidence", 0.75))
            bw, bh = max(x2 - x1, 2), max(y2 - y1, 2)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            sigma_x, sigma_y = max(bw / 2.2, 2.0), max(bh / 2.2, 2.0)
            ys = np.arange(y1, y2)[:, None]
            xs = np.arange(x1, x2)[None, :]
            gauss = np.exp(
                -(((xs - cx) ** 2) / (2 * sigma_x ** 2)
                  + ((ys - cy) ** 2) / (2 * sigma_y ** 2))
            )
            det_mask[y1:y2, x1:x2] = np.maximum(
                det_mask[y1:y2, x1:x2], gauss * confidence
            )

        if det_mask.max() > 0:
            det_mask /= det_mask.max()
            cam = 0.45 * cam + 0.55 * det_mask
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        del det_mask

    cam_uint8 = np.uint8(np.clip(cam * 255.0, 0, 255))
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 0.55, heatmap_rgb, 0.45, 0)

    if detections:
        for detection in detections:
            box = detection.get("box")
            if box and len(box) == 4:
                x1, y1, x2, y2 = map(int, box)
                x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
                y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)

    metrics = {
        "method": "Multi-Scale YOLO Feature Activation CAM",
        "target_layers": "YOLO Feature Neck (C2f & SPPF)",
        "peak_intensity": round(float(cam.max()) * 100.0, 1),
        "mean_attention": round(float(cam.mean()) * 100.0, 1),
        "high_attention_coverage_pct": round(float(np.mean(cam > 0.60)) * 100.0, 1),
        "colormap": "JET (Blue=Background, Red=Peak Salience)",
    }
    del cam, cam_uint8, heatmap_bgr
    gc.collect()
    return overlay, heatmap_rgb, metrics


def generate_gradcam(image_rgb, detections=None):
    overlay, _, _ = generate_explainability_heatmap(image_rgb, detections)
    return overlay
