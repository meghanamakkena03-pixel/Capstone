AQUA SCOPE - RENDER MEMORY OPTIMIZATION PATCH

IMPORTANT
- This patch does NOT change NU2Net architecture or checkpoint.
- It does NOT change best.pt weights.
- Your local command confirmed best.pt task == "segment".
- Detection and segmentation now share one loaded YOLO instance.
- Detection boxes are scaled back to original image coordinates.
- Image model inference is capped to 512 px maximum side (configurable).
- YOLO inference uses imgsz=416 (configurable).
- Explainability uses a bounded 320 px working image and a low-memory
  channel-mean feature activation projection. The output remains an attention
  heatmap, but its numerical method is not the old PCA Eigen-CAM implementation.
- All AI endpoints share a lock, so only one model pipeline runs at a time.
- API routes and response keys from the supplied main.py are retained.

INSTALL
Copy these files into your existing backend folder, replacing same-named files:
  runtime_config.py
  detection.py
  segmentation.py
  enhancement.py
  explainability.py
  main.py

DO NOT replace:
  analytics.py
  auth.py
  requirements.txt
  models/
  web/

Your backend folder must still contain:
  backend/models/best.pt
  backend/models/NU2Net.ckpt
  backend/models/NU2Net_UIE/model/NU2Net.py

The Python file called segmentation.pt that you previously shared is only a
path-check script, not a weights checkpoint. It is not used by this patch.

LOCAL RUN
Terminal 1:
  cd "C:\Users\megha\OneDrive\Documents\Desktop\capstone-folder\ezyZip (1)\backend"
  python -m uvicorn main:app --host 127.0.0.1 --port 8000

Terminal 2:
  cd "C:\Users\megha\OneDrive\Documents\Desktop\capstone-folder\ezyZip (1)\web"
  npm run dev

Open:
  http://localhost:5173
API docs:
  http://127.0.0.1:8000/docs

LOCAL FRONTEND API URL
Your web frontend must point to http://127.0.0.1:8000 during local testing.
For production, it must point to https://capstone-backend-dtau.onrender.com.
Do not commit a local-only URL as the production URL. Prefer a Vite environment
variable (VITE_API_BASE_URL) if your frontend is already structured for it.

DEPLOY
After local tests pass:
  git status
  git add backend/main.py backend/runtime_config.py backend/detection.py backend/segmentation.py backend/enhancement.py backend/explainability.py
  git commit -m "Optimize backend memory for Render"
  git push origin main

Render settings remain:
  Root Directory: backend
  Build Command: pip install -r requirements.txt
  Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT

MEMORY LIMIT
These changes reduce peak memory but cannot guarantee the complete application
fits Render Free's 512 MB limit. PyTorch, OpenCV, Ultralytics, loaded model
weights, model activations, and encoded image responses all consume RAM.
If Render still OOMs, preserving all models/features requires a higher-memory
instance or a different host. Do not claim a successful fix until /analyze
completes and Render Events show no OOM.

NOTE ABOUT EXPLAINABILITY
The supplied implementation described its method as Grad-CAM/Eigen-CAM, but
it did not calculate gradient-based Grad-CAM; it used feature activations and
PCA. This patch retains the attention heatmap feature while using a less
memory-intensive channel-mean feature projection. This is a methodological
change to the heatmap calculation, not a model architecture change.
