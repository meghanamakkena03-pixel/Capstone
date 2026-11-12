from pathlib import Path
import gc
import sys

import cv2
import numpy as np
import torch

from runtime_config import INFERENCE_DEVICE, MAX_MODEL_SIDE

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
NU2_ROOT = MODELS_DIR / "NU2Net_UIE"
NU2_CHECKPOINT = MODELS_DIR / "NU2Net.ckpt"
DEVICE = torch.device(INFERENCE_DEVICE)
_nu2_model = None


def get_device():
    return str(DEVICE)


def load_enhancement_model():
    global _nu2_model
    if _nu2_model is not None:
        return _nu2_model

    if not NU2_ROOT.is_dir():
        raise FileNotFoundError(f"NU2Net source folder not found: {NU2_ROOT}")
    if not NU2_CHECKPOINT.is_file():
        raise FileNotFoundError(f"NU2Net checkpoint not found: {NU2_CHECKPOINT}")

    if str(NU2_ROOT) not in sys.path:
        sys.path.insert(0, str(NU2_ROOT))

    try:
        from model.NU2Net import NU2Net
    except ImportError as exc:
        raise RuntimeError(
            "Could not import NU2Net. Check backend/models/NU2Net_UIE/model/NU2Net.py."
        ) from exc

    model = NU2Net()
    checkpoint = torch.load(
        NU2_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Invalid NU2Net checkpoint: expected a dictionary.")

    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise RuntimeError("Invalid NU2Net checkpoint: state_dict is missing or invalid.")

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module.model."):
            cleaned[key[len("module.model."):]] = value
        elif key.startswith("model."):
            cleaned[key[len("model."):]] = value
        elif key.startswith("module."):
            cleaned[key[len("module."):]] = value
        elif key.startswith(("encoder.", "decoder.")):
            cleaned[key] = value

    if not cleaned:
        raise RuntimeError("No NU2Net model weights were found in the checkpoint.")

    model.load_state_dict(cleaned, strict=True)
    model.to(DEVICE)
    model.eval()
    _nu2_model = model

    del checkpoint, state_dict, cleaned
    gc.collect()
    print(f"NU2Net loaded successfully on {DEVICE}")
    return _nu2_model


def enhance_image(image_rgb):
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("The input image is empty.")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with three channels.")

    model = load_enhancement_model()
    original_h, original_w = image_rgb.shape[:2]

    # Bound NU2Net activation memory without changing its architecture or weights.
    scale = min(1.0, float(MAX_MODEL_SIDE) / max(original_h, original_w))
    if scale < 1.0:
        work_w = max(16, int(round(original_w * scale / 16)) * 16)
        work_h = max(16, int(round(original_h * scale / 16)) * 16)
        work_rgb = cv2.resize(image_rgb, (work_w, work_h), interpolation=cv2.INTER_AREA)
    else:
        work_rgb = image_rgb

    height, width = work_rgb.shape[:2]
    pad_h = (16 - height % 16) % 16
    pad_w = (16 - width % 16) % 16
    padded = cv2.copyMakeBorder(
        work_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
    )

    input_tensor = torch.from_numpy(
        np.ascontiguousarray(padded.transpose(2, 0, 1)[None]).astype(np.float32) / 255.0
    )

    with torch.inference_mode():
        output = model(input_tensor)

    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise RuntimeError("NU2Net did not return a tensor.")

    output_np = (
        output.squeeze(0).cpu().permute(1, 2, 0).numpy()
    )
    enhanced_small = (np.clip(output_np[:height, :width], 0.0, 1.0) * 255.0).astype(np.uint8)

    del input_tensor, output, output_np, padded, work_rgb
    gc.collect()

    if enhanced_small.shape[:2] != (original_h, original_w):
        enhanced_rgb = cv2.resize(
            enhanced_small, (original_w, original_h), interpolation=cv2.INTER_LINEAR
        )
        del enhanced_small
        return enhanced_rgb
    return enhanced_small
