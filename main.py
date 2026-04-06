import base64
import os
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
from google.cloud import storage

# ==========================
# Config
# ==========================
MODEL_PATH = "/tmp/brainova_model.keras"
MODEL_GCS_URI = os.getenv("MODEL_GCS_URI")  # gs://bucket/brainova_model.keras


CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# Lazy model state
_model = None
_model_ready = False
_model_error = None
_model_lock = threading.Lock()

app = FastAPI()


# ==========================
# Helpers: Download + Validate .keras
# ==========================


def download_model_from_gcs(gcs_uri: str, destination: str):
    if not gcs_uri or not gcs_uri.startswith("gs://"):
        raise RuntimeError("MODEL_GCS_URI must be set like: gs://bucket/object")

    # parse gs://bucket/object
    no_scheme = gcs_uri.replace("gs://", "", 1)
    bucket_name, blob_name = no_scheme.split("/", 1)

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    tmp_path = destination + ".tmp"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.download_to_filename(tmp_path)
    os.replace(tmp_path, destination)

def ensure_model_file():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1024 * 1024:
        validate_keras_zip(MODEL_PATH)
        return

    print("⬇️ Downloading model from GCS ...")
    download_model_from_gcs(MODEL_GCS_URI, MODEL_PATH)
    validate_keras_zip(MODEL_PATH)
    print(f"✅ Model downloaded: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)} bytes)")

def validate_keras_zip(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")

    size = os.path.getsize(path)
    if size < 5 * 1024 * 1024:
        raise RuntimeError(f"Model file too small ({size} bytes). Likely failed download.")

    with open(path, "rb") as f:
        head = f.read(4)

    # .keras is a ZIP file => starts with PK\x03\x04
    if head != b"PK\x03\x04":
        raise RuntimeError(
            f"Not a valid .keras ZIP. Header bytes: {head.hex()} "
            f"(wrong URL or HTML saved)."
        )





def get_model():
    """
    Lazy-load the model the first time /predict is called.
    This keeps /health working even if the model fails.
    """
    global _model, _model_ready, _model_error

    if _model_ready and _model is not None:
        return _model

    with _model_lock:
        if _model_ready and _model is not None:
            return _model

        try:
            ensure_model_file()
            print("🧠 Loading model...")
            _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
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
# Debug endpoints
# ==========================
@app.get("/health")
def health():
    return {"ok": True, "model_ready": _model_ready, "model_error": _model_error}


@app.get("/versions")
def versions():
    return {
        "tf_version": tf.__version__,
        "keras_version": getattr(tf.keras, "__version__", "unknown"),
    }


# ==========================
# Preprocess
# ==========================
def preprocess_like_colab(file_bytes: bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR
    if img_bgr is None:
        raise ValueError("Could not decode image. Make sure it's a valid JPG/PNG.")

    img_bgr = cv2.resize(img_bgr, (240, 240))
    arr = img_to_array(img_bgr)  # float32 0..255 (keeps BGR order)
    return arr
def decode_image(file_bytes: bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image. Make sure it's a valid JPG/PNG.")
    return img_bgr


def preprocess_like_colab_from_bgr(img_bgr: np.ndarray):
    img_bgr = cv2.resize(img_bgr, (240, 240))
    arr = img_to_array(img_bgr)  # float32 0..255 (keeps BGR order)
    return arr


def validate_brain_mri_like(img_bgr: np.ndarray):
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if h < 128 or w < 128:
        return False, "Image too small for MRI.", {}

    # MRI images are usually close to grayscale
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    colorfulness = np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b))
    if colorfulness > 25:
        return False, "Image is too colorful to be a brain MRI.", {
            "colorfulness": float(colorfulness)
        }

    # MRI scans usually have dark background around the head
    border = 20
    border_pixels = np.concatenate([
        gray[:border, :].flatten(),
        gray[-border:, :].flatten(),
        gray[:, :border].flatten(),
        gray[:, -border:].flatten()
    ])
    dark_ratio = np.mean(border_pixels < 40)
    if dark_ratio < 0.5:
        return False, "Missing MRI-style dark background.", {
            "dark_ratio": float(dark_ratio)
        }

    # Detect main object area
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
    if num_labels <= 1:
        return False, "No brain-like region detected.", {}

    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    area = stats[largest_idx, cv2.CC_STAT_AREA]
    area_ratio = area / float(h * w)

    

    return True, "Looks like MRI", {
        "colorfulness": float(colorfulness),
        "dark_ratio": float(dark_ratio),
        "area_ratio": float(area_ratio)
    }

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

    activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
    activation_map = np.uint8(255 * activation_map)

    heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)

    original_img_norm = np.uint8(
        (original_img - original_img.min()) / (original_img.max() - original_img.min() + 1e-8) * 255
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

        allowed_types = {"image/jpeg", "image/png", "image/jpg"}
        if file.content_type not in allowed_types:
            return JSONResponse(
                {"success": False, "message": "Only JPG and PNG images are allowed."},
                status_code=400
            )

        img_bgr = decode_image(contents)

        is_valid, reason, debug = validate_brain_mri_like(img_bgr)
        if not is_valid:
            return JSONResponse(
                {
                    "success": False,
                    "message": reason,
                    "validation": debug
                },
                status_code=400
            )

        img_arr = preprocess_like_colab_from_bgr(img_bgr)

        m = get_model()  # lazy-load stays the same
        overlay_rgb, probs, idx = VizGradCAM_API(m, img_arr, interpolant=0.5)

        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", overlay_bgr)
        if not ok:
            return JSONResponse(
                {"success": False, "message": "Failed to encode overlay"},
                status_code=500
            )

        overlay_b64 = base64.b64encode(buf).decode("utf-8")

        return {
            "success": True,
            "message": "Prediction completed successfully.",
            "label": CLASS_NAMES[idx],
            "probabilities": probs,
            "gradcam_image_base64": overlay_b64,
            "validation": debug
        }

    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)
