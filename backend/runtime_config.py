"""
Aqua Scope low-memory runtime settings.
This does not change any trained model architecture or checkpoint.
"""
import os

# Set before importing torch where possible.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    # PyTorch may already have initialized its thread pools.
    pass

# Inference is intentionally CPU-only on the Render 512 MB instance.
# Change only if deploying on a machine with a supported GPU.
INFERENCE_DEVICE = "cpu"

# Keep inference working dimensions bounded. Original-resolution images
# are returned to the frontend; only model inference uses these dimensions.
MAX_MODEL_SIDE = int(os.getenv("AQUASCOPE_MAX_MODEL_SIDE", "512"))
YOLO_IMAGE_SIZE = int(os.getenv("AQUASCOPE_YOLO_IMAGE_SIZE", "416"))
EXPLAIN_IMAGE_SIZE = int(os.getenv("AQUASCOPE_EXPLAIN_IMAGE_SIZE", "320"))
