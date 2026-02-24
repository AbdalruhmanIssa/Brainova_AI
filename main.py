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
MODEL_URL = os.getenv("MODEL_URL")  # GitHub Release direct link

# Example (correct pattern):
# https://github.com/<user>/<repo>/releases/download/<tag>/brainova_model.keras

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
MODEL_URL = os.getenv("MODEL_URL")  # required on Render

def download_model(destination: str):
    if not MODEL_URL:
        raise RuntimeError("MODEL_URL env var is not set.")

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    tmp_path = destination + ".tmp"

    headers = {
        "User-Agent": "brainova-ai/1.0",
        "Accept": "application/octet-stream",
    }

    r = requests.get(MODEL_URL, stream=True, timeout=180, allow_redirects=True, headers=headers)
    r.raise_for_status()

    ct = (r.headers.get("content-type") or "").lower()
    if "text/html" in ct:
        # This happens when you use a blob/page link instead of release asset link
        raise RuntimeError(f"MODEL_URL returned HTML (content-type={ct}). Use the GitHub Releases *asset* URL.")

    with open(tmp_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    # atomic replace
    os.replace(tmp_path, destination)
def validate_keras_zip(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")

    size = os.path.getsize(path)
    if size < 5 * 1024 * 1024:
        raise RuntimeError(f"Model file too small ({size} bytes). Likely HTML/failed download.")

    with open(path, "rb") as f:
        head = f.read(4)

    # .keras is a ZIP file => starts with PK
    if head != b"PK\x03\x04":
        # show first bytes in hex to debug
        raise RuntimeError(f"Not a valid .keras zip. Header bytes: {head.hex()} (wrong URL or HTML saved).")
def ensure_model_file():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1024 * 1024:
        # still validate it (in case it’s an HTML file saved earlier)
        validate_keras_zip(MODEL_PATH)
        return

    print("⬇️ Downloading model from GitHub Releases MODEL_URL...")
    download_model(MODEL_PATH)
    validate_keras_zip(MODEL_PATH)
    print("✅ Model ready:", MODEL_PATH, "size:", os.path.getsize(MODEL_PATH))


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