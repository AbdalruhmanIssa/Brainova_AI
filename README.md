# Brainova AI — Brain Tumor MRI Classifier

Brainova AI is a deep-learning microservice that classifies brain-MRI images into one of four categories — **glioma**, **meningioma**, **no-tumor**, or **pituitary** — and returns an explainable **Grad-CAM heatmap** highlighting the region that drove the decision.

It is built with **FastAPI**, runs a **TensorFlow / EfficientNet** model, and is deployed as a containerized service on **Google Cloud Run**.

> ⚠️ **Medical disclaimer:** This is a decision-support and educational tool, **not** a substitute for a qualified radiologist. Predictions should always be reviewed by a medical professional.

---

## 🔗 Live demo & resources

| Resource | Link |
|---|---|
| **Live API (interactive Swagger docs)** | https://brainova-ai-1031567223264.europe-west1.run.app/docs |
| **Model training notebook (Google Colab)** | https://colab.research.google.com/drive/1zL-2A7jRuvODVOPz61kRNEKX1aACmbQH?usp=sharing |
| **Dataset (Kaggle — Brain Tumor MRI Dataset)** | https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset |

You can try the classifier right now: open the **/docs** link above, expand `POST /predict`, click **Try it out**, upload a brain-MRI image (JPG/PNG), and execute.

---

## Table of contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [How prediction works (step by step)](#how-prediction-works-step-by-step)
- [API reference](#api-reference)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Running locally](#running-locally)
- [Deploying to Cloud Run](#deploying-to-cloud-run)
- [The model & training](#the-model--training)
- [Grad-CAM explainability](#grad-cam-explainability)
- [Input validation](#input-validation)
- [License & credits](#license--credits)

---

## What it does

1. Accepts a brain-MRI image via a REST endpoint.
2. **Validates** that the upload actually looks like a grayscale brain MRI (rejects color photos, selfies, and non-brain images).
3. **Preprocesses** the image the same way the model was trained — crops the brain region and resizes to 240×240.
4. Runs **inference** through a trained EfficientNet classifier (with light test-time augmentation for stability).
5. Generates a **Grad-CAM heatmap** overlaid on the original MRI, showing which pixels most influenced the prediction.
6. Returns JSON containing the predicted **label**, per-class **probabilities**, and the **heatmap image** (base64-encoded).

The four output classes are: `glioma`, `meningioma`, `notumor`, `pituitary`.

---

## Architecture

Brainova AI is the **AI inference microservice** in a larger system. It is intentionally kept separate from the main application backend:

```
┌──────────┐      ┌───────────────────┐      ┌──────────────────────────┐
│ Frontend │ ───▶ │  ASP.NET backend  │ ───▶ │  This service (Cloud Run) │
│  (UI)    │ ◀─── │   (main API)      │ ◀─── │   FastAPI + TensorFlow    │
└──────────┘      └───────────────────┘      └──────────────────────────┘
                                                         │
                                                         ▼
                                              ┌────────────────────────┐
                                              │  Google Cloud Storage   │
                                              │  brainova_model.keras   │
                                              └────────────────────────┘
```

The frontend never calls this service directly. The ASP.NET backend receives the user's upload and calls this Python service's `/predict` endpoint, then passes the result back to the frontend.

**Why a separate microservice?** The ML stack (TensorFlow, OpenCV, the model weights) is heavy and Python-based. Isolating it lets the main backend stay lightweight, lets the AI component scale independently on Cloud Run, and keeps ML dependencies decoupled from the rest of the app.

**Model loading.** The model is **not** bundled inside the container image. On the first prediction request it is downloaded from Google Cloud Storage (via the `MODEL_GCS_URI` environment variable) into `/tmp`, validated as a real `.keras` archive, and cached in memory (lazy loading). This keeps the container image small and startup fast, and lets the `/health` endpoint respond even before the model is loaded. A thread lock ensures the model is loaded only once even under concurrent requests.

---

## How prediction works (step by step)

1. **Receive** — image arrives at `POST /predict`.
2. **Check file type** — only `image/jpeg`, `image/jpg`, and `image/png` are accepted.
3. **Decode** — bytes → image array via OpenCV.
4. **Validate MRI-likeness** — size, colorfulness, and presence of a brain-like region.
5. **Preprocess** — crop the brain region, resize to 240×240, convert BGR→RGB.
6. **Load model (lazy)** — download from Cloud Storage on first request, then cache.
7. **Predict** — run the model with 2-view test-time augmentation (original + horizontal flip, averaged).
8. **Grad-CAM** — build a heatmap of the most influential pixels and overlay it on the original image.
9. **Respond** — return JSON: label, probabilities, and base64 heatmap.

---

## API reference

Base URL (production): `https://brainova-ai-1031567223264.europe-west1.run.app`

### `POST /predict`

Classify a brain-MRI image and return a Grad-CAM overlay.

**Request:** `multipart/form-data` with a single file field named `file` (JPG or PNG).

```bash
curl -X POST \
  "https://brainova-ai-1031567223264.europe-west1.run.app/predict" \
  -F "file=@brain_scan.jpg"
```

**Success response (`200`):**

```json
{
  "success": true,
  "message": "Prediction completed successfully.",
  "label": "glioma",
  "probabilities": [0.92, 0.04, 0.02, 0.02],
  "gradcam_image_base64": "<base64-encoded JPEG>",
  "heatmap_available": true,
  "validation": { "colorfulness": 3.1, "area_ratio": 0.42 }
}
```

`probabilities` are ordered as `[glioma, meningioma, notumor, pituitary]`.

**Error responses:** `400` for invalid input (wrong file type or a non-MRI image), `500` for server-side errors. Both return `{ "success": false, "message": "..." }`.

### `GET /health`

Lightweight liveness/readiness check. Returns whether the service is up and whether the model has loaded — without triggering a prediction.

```json
{ "ok": true, "model_ready": false, "model_error": null }
```

### `GET /versions`

Returns the TensorFlow and Keras versions the service is running (useful for debugging environment mismatches).

---

## Tech stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI |
| ASGI server | Uvicorn |
| ML framework | TensorFlow 2.20 / Keras |
| Model architecture | EfficientNet (transfer learning) |
| Image processing | OpenCV (headless), Pillow, NumPy |
| Explainability | Grad-CAM (HiRes/Layer-CAM variant) |
| Model storage | Google Cloud Storage (`google-cloud-storage`) |
| Containerization | Docker (`python:3.11-slim` base) |
| Hosting | Google Cloud Run (serverless containers) |
| Runtime | Python 3.11 |

---

## Project structure

```
Brainova_AI/
├── main.py            # FastAPI app: /health, /versions, /predict
│                      #   - lazy-downloads the model from GCS
│                      #   - MRI validation, crop + resize preprocessing
│                      #   - EfficientNet inference with 2-view TTA
│                      #   - full-picture Grad-CAM overlay
├── Dockerfile         # Container build (python:3.11-slim, uvicorn on $PORT)
├── .dockerignore      # Excludes venv, caches, and model weights from the image
├── requirements.txt   # Python dependencies
├── runtime.txt        # Python version pin (3.11.9)
└── README.md
```

The model weights (`*.keras`) are intentionally **not** in the repo — they are large and are streamed from Google Cloud Storage at runtime (see [Model loading](#architecture)). `.dockerignore` also keeps them out of the container image.

---

## Running locally

Requires **Python 3.11** and access to a GCS bucket holding `brainova_model.keras` (the service loads the model from GCS, not from disk).

```bash
# 1. Clone
git clone https://github.com/AbdalruhmanIssa/Brainova_AI.git
cd Brainova_AI

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Authenticate to Google Cloud (so the app can read the model from GCS)
gcloud auth application-default login

# 5. Point the app at your model and run
export MODEL_GCS_URI="gs://<your-bucket>/brainova_model.keras"
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

Then open http://localhost:8080/docs for the interactive Swagger UI.

### Run with Docker

```bash
docker build -t brainova-ai .
docker run -p 8080:8080 \
  -e PORT=8080 \
  -e MODEL_GCS_URI="gs://<your-bucket>/brainova_model.keras" \
  -v $HOME/.config/gcloud:/root/.config/gcloud \
  brainova-ai
```

---

## Deploying to Cloud Run

The production service is built from the included `Dockerfile` and deployed to Cloud Run in `europe-west1`.

```bash
# 1. Upload the model to a GCS bucket
gsutil cp brainova_model.keras gs://<your-bucket>/brainova_model.keras

# 2. Deploy from source (Cloud Run builds the Dockerfile automatically)
gcloud run deploy brainova-ai \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --set-env-vars MODEL_GCS_URI="gs://<your-bucket>/brainova_model.keras"
```

The container's service account needs **read** access to the bucket. Cloud Run injects the `PORT` env var (the `Dockerfile` binds uvicorn to it), and its writable disk is `/tmp`, where the model is cached after download.

---

## The model & training

The classifier was trained in Google Colab on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (glioma, meningioma, no-tumor, pituitary).

Key training details (mirrored exactly at inference time so input matches training):

- **Brain-region cropping** — each MRI is cropped to the bounding box of the largest bright contour to remove black borders and center the brain.
- **Resize to 240×240** — the fixed input resolution of the network.
- **EfficientNet backbone** — transfer learning with a built-in rescaling layer (inputs stay in the 0–255 range rather than being manually normalized).
- **4-class softmax head** — outputs one probability per tumor class.

The full pipeline — data loading, augmentation, model definition, training loop, and evaluation — lives in the [Colab notebook](https://colab.research.google.com/drive/1zL-2A7jRuvODVOPz61kRNEKX1aACmbQH?usp=sharing).

---

## Grad-CAM explainability

Instead of returning a black-box answer, the service overlays a **Grad-CAM** heatmap on the original MRI so a clinician can see *where* the model looked. Warm colors mark high-influence regions.

Implementation notes:

- Uses a **HiRes/Layer-CAM** style per-pixel weighting (summing ReLU-gated gradient × activation across channels) rather than plain channel-averaged Grad-CAM, for a sharper map.
- The heatmap is computed on the 240×240 cropped input, then **remapped back onto the full original image** using the saved crop bounding box — so the highlight appears on the exact picture that was uploaded.
- For the **`notumor`** class the heatmap is skipped (a flat, cold overlay is returned) since there is no meaningful tumor region to highlight; `heatmap_available` is `false` in that case.

---

## Input validation

Before running inference, uploads are screened so users can't get a fake "diagnosis" from an unrelated image:

- **Minimum size** — images smaller than 128×128 are rejected.
- **Colorfulness** — real MRIs are near-grayscale; overly colorful images are rejected.
- **Brain-region presence** — a connected-components check confirms a substantial bright region exists.

If validation fails, the API returns `400` with an explanatory message rather than a prediction.

---

## License & credits

Developed as a graduation project by the Brainova team.

- Dataset: Masoud Nickparvar — [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle).
- Built with FastAPI, TensorFlow, OpenCV, and Google Cloud Run.
