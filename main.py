import base64
import os
import re
import threading

import cv2
import numpy as np
import requests
import tensorflow as tf
import tensorflow.keras as K
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array

# ==========================
# Config
# ==========================
MODEL_PATH = "model/brainova_model.keras"
MODEL_FILE_ID = os.getenv("MODEL_FILE_ID", "19OrsrlSqZ-pHYWSm_QE9hyF7ykkG0Q1")

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# Lazy model state
_model = None
_model_ready = False
_model_error = None
_model_lock = threading.Lock()

app = FastAPI()


# ==========================
# Google Drive download (robust)
# ==========================
def download_from_drive(file_id: str, destination: str):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    session = requests.Session()

    def save_stream(resp):
        resp.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    # Strategy 1: Try direct uc download with confirm=t (often works for large files)
    url1 = "https://drive.google.com/uc"
    r = session.get(url1, params={"id": file_id, "export": "download", "confirm": "t"}, stream=True)
    if r.status_code == 200 and "text/html" not in r.headers.get("Content-Type", ""):
        save_stream(r)
        return

    # Strategy 2: Do the normal flow but capture token from HTML/cookies
    r = session.get(url1, params={"id": file_id, "export": "download"}, stream=True)
    r.raise_for_status()

    token = None
    for k, v in r.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break

    if token is None and "text/html" in r.headers.get("Content-Type", ""):
        html = r.text
        m = re.search(r'confirm=([0-9A-Za-z_]+)', html)
        if m:
            token = m.group(1)

    if token:
        r2 = session.get(url1, params={"id": file_id, "export": "download", "confirm": token}, stream=True)
        # Sometimes Google redirects to drive.usercontent; allow it, but only if it becomes a real file
        if r2.status_code == 200 and "text/html" not in r2.headers.get("Content-Type", ""):
            save_stream(r2)
            return

    # Strategy 3: Last fallback — use the "download?export=download" endpoint
    url2 = "https://drive.google.com/uc?export=download&id=" + file_id
    r3 = session.get(url2, stream=True)
    if r3.status_code == 200 and "text/html" not in r3.headers.get("Content-Type", ""):
        save_stream(r3)
        return

    raise RuntimeError("Failed to download model from Google Drive on this host (Drive returned HTML/404).")


def ensure_model_file():
    # If file exists and is > 1MB, assume it’s real (prevents partial downloads)
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1024 * 1024:
        return

    print("⬇️ Downloading model from Google Drive...")
    download_from_drive(MODEL_FILE_ID, MODEL_PATH)

    size = os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0
    print(f"✅ Model downloaded: {MODEL_PATH} ({size} bytes)")


def get_model():
    """
    Lazy-load the model the first time /predict is called.
    Keeps Render alive even if the model download fails.
    """
    global _model, _model_ready, _model_error

    if _model_ready and _model is not None:
        return _model

    with _model_lock:
        # double-check inside lock
        if _model_ready and _model is not None:
            return _model

        try:
            ensure_model_file()
            print("🧠 Loading model...")
            _model = tf.keras.models.load_model(MODEL_PATH)
            _model_ready = True
            _model_error = None
            print("✅ Model loaded successfully.")
            return _model
        except Exception as e:
            _model_ready = False
            _model_error = str(e)
            print("❌ Model load failed:", _model_error)
            raise


# ==========================
# Health endpoint
# ==========================
@app.get("/health")
def health():
    return {"ok": True, "model_ready": _model_ready, "model_error": _model_error}


# ==========================
# Preprocess (like your Colab intent)
# ==========================
def preprocess_like_colab(file_bytes: bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR
    if img_bgr is None:
        raise ValueError("Could not decode image. Make sure it's a valid JPG/PNG.")

    img_bgr = cv2.resize(img_bgr, (240, 240))
    arr = img_to_array(img_bgr)  # float32 0..255
    return arr


# ==========================
# Grad-CAM
# ==========================
def VizGradCAM_API(model, image, interpolant=0.5):
    assert 0 < interpolant < 1, "Heatmap Interpolation Must Be Between 0 - 1"

    last_conv_layer = next(x for x in model.layers[::-1] if isinstance(x, K.layers.Conv2D))
    target_layer = model.get_layer(last_conv_layer.name)

    original_img = image
    img = np.expand_dims(original_img, axis=0)

    prediction = model.predict(img, verbose=0)
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    prediction_idx = int(np.argmax(prediction))

    with tf.GradientTape() as tape:
        gradient_model = Model([model.inputs], [target_layer.output, model.output])
        conv2d_out, pred2 = gradient_model(img, training=False)

        if isinstance(pred2, (list, tuple)):
            pred2 = pred2[0]

        loss = pred2[:, prediction_idx]

    gradients = tape.gradient(loss, conv2d_out)
    output = conv2d_out[0]
    weights = tf.reduce_mean(gradients[0], axis=(0, 1))

    activation_map = np.zeros(output.shape[0:2], dtype=np.float32)
    for idx, weight in enumerate(weights):
        activation_map += float(weight) * output[:, :, idx].numpy()

    activation_map = cv2.resize(activation_map, (original_img.shape[1], original_img.shape[0]))
    activation_map = np.maximum(activation_map, 0)

    activation_map = (activation_map - activation_map.min()) / (
        activation_map.max() - activation_map.min() + 1e-8
    )
    activation_map = np.uint8(255 * activation_map)

    heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)

    original_img_norm = np.uint8(
        (original_img - original_img.min())
        / (original_img.max() - original_img.min() + 1e-8)
        * 255
    )

    cvt_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay_rgb = np.uint8(original_img_norm * interpolant + cvt_heatmap * (1 - interpolant))

    probs = prediction[0].tolist() if prediction.ndim == 2 else prediction.tolist()
    return overlay_rgb, probs, prediction_idx


# ==========================
# Predict endpoint
# ==========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_arr = preprocess_like_colab(contents)

        m = get_model()  # <-- lazy load here
        overlay_rgb, probs, idx = VizGradCAM_API(m, img_arr, interpolant=0.5)

        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", overlay_bgr)
        if not ok:
            return JSONResponse({"success": False, "message": "Failed to encode overlay"}, status_code=500)

        overlay_b64 = base64.b64encode(buf).decode("utf-8")

        return {
            "label": CLASS_NAMES[idx],
            "probabilities": probs,
            "gradcam_image_base64": overlay_b64,
        }

    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)