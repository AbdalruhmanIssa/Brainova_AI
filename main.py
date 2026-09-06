import base64
import os
import threading
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array

# ==========================
# Config
# ==========================
MODEL_PATH = os.getenv("MODEL_CACHE_PATH", "/tmp/brainova_model.keras")

# Option A (recommended) - blob URL + the container app's managed identity:
#   https://<account>.blob.core.windows.net/models/brainova_model.keras
MODEL_BLOB_URL = os.getenv("MODEL_BLOB_URL")

# Option B (fallback) - the same URL with a SAS token appended. No Azure auth
# needed, but the token expires. Handy for a quick local test.
MODEL_SAS_URL = os.getenv("MODEL_SAS_URL")
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
INPUT_SIZE = (240, 240)  # (W, H) — must match training

# Lazy model state
_model = None
_model_ready = False
_model_error = None
_model_lock = threading.Lock()

app = FastAPI()


# ==========================
# Helpers: Download from Azure Blob Storage + Validate .keras
# ==========================
def download_model_from_azure(destination: str):
    """
    Download the .keras model from Azure Blob Storage into `destination`.

    Writes to a .tmp file first and only renames on success, so a half-finished
    download can never be mistaken for a valid cached model.
    """
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    tmp_path = destination + ".tmp"

    if MODEL_SAS_URL:
        import requests
        with requests.get(MODEL_SAS_URL, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
    elif MODEL_BLOB_URL:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobClient
        blob = BlobClient.from_blob_url(
            MODEL_BLOB_URL, credential=DefaultAzureCredential()
        )
        with open(tmp_path, "wb") as f:
            blob.download_blob(max_concurrency=4).readinto(f)
    else:
        raise RuntimeError(
            "No model source configured. Set MODEL_BLOB_URL (managed identity) "
            "or MODEL_SAS_URL (SAS token)."
        )

    os.replace(tmp_path, destination)


def ensure_model_file():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1024 * 1024:
        validate_keras_zip(MODEL_PATH)
        return
    print("⬇️ Downloading model from Azure Blob Storage ...")
    download_model_from_azure(MODEL_PATH)
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
def decode_image(file_bytes: bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image. Make sure it's a valid JPG/PNG.")
    return img_bgr


def crop_brain_region_with_bbox(img_bgr: np.ndarray):
    """
    Same training-time crop logic as main_cloudrun.py, but ALSO returns
    the bounding box (x1, y1, x2, y2) in original-image coordinates so
    the heatmap can be remapped onto the full uploaded MRI for display.

    The model still consumes the cropped+resized 240x240 region. The
    bbox is only used by the visualization path.

    Falls back to a bbox covering the whole image if contour search
    fails — in that case the cropped image already IS the original.
    """
    H, W = img_bgr.shape[:2]
    full_bbox = (0, 0, W, H)
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return img_bgr, full_bbox

        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 100:
            return img_bgr, full_bbox

        ext_left = tuple(c[c[:, :, 0].argmin()][0])
        ext_right = tuple(c[c[:, :, 0].argmax()][0])
        ext_top = tuple(c[c[:, :, 1].argmin()][0])
        ext_bottom = tuple(c[c[:, :, 1].argmax()][0])

        x1, y1, x2, y2 = ext_left[0], ext_top[1], ext_right[0], ext_bottom[1]
        cropped = img_bgr[y1:y2, x1:x2]
        if cropped.size == 0 or cropped.shape[0] < 5 or cropped.shape[1] < 5:
            return img_bgr, full_bbox
        return cropped, (x1, y1, x2, y2)
    except Exception:
        return img_bgr, full_bbox


def preprocess_for_model(img_bgr: np.ndarray):
    """
    Full-pic variant.

    Training pipeline (unchanged for model input):
      1. crop_brain_region     (cropped MRI region)
      2. cv2.resize to 240x240
      3. BGR -> RGB
      4. float32 in 0..255 (EfficientNet Rescaling layer inside)

    Display image (different from main_cloudrun.py):
      • The ORIGINAL full uploaded MRI, BGR -> RGB only.
      • NO crop, NO resize. The returned Grad-CAM overlay will have
        the same dimensions as the file the caller uploaded.

    Returns:
      original_rgb_uint8    : (H, W, 3) uint8   — full original MRI as RGB
      model_input_rgb_f32   : (240,240,3) float32 — cropped + resized
      crop_bbox             : (x1, y1, x2, y2) in original coords
    """
    cropped_bgr, crop_bbox = crop_brain_region_with_bbox(img_bgr)
    resized_bgr = cv2.resize(cropped_bgr, INPUT_SIZE)
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    model_input_rgb_f32 = img_to_array(resized_rgb).astype(np.float32)

    original_rgb_uint8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return original_rgb_uint8, model_input_rgb_f32, crop_bbox


def validate_brain_mri_like(img_bgr: np.ndarray):
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if h < 128 or w < 128:
        return False, "Image too small for MRI.", {}

    b, g, r = cv2.split(img_bgr.astype(np.float32))
    colorfulness = np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b))

    if colorfulness > 25:
        return False, "Image is too colorful to be a brain MRI.", {
            "colorfulness": float(colorfulness)
        }

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
        "area_ratio": float(area_ratio)
    }


# ==========================
# Grad-CAM
# ==========================
def _get_last_feature_map_layer(model):
    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
        except Exception:
            continue
        if shape is not None and len(shape) == 4:
            return layer
    return next(x for x in model.layers[::-1] if isinstance(x, K.layers.Conv2D))


def VizGradCAM_API(model, display_rgb, model_input_rgb, crop_bbox,
                   interpolant=0.5, skip_classes=(), override_probs=None):
    """
    Full-pic Grad-CAM.

    display_rgb     : (H, W, 3) uint8     — FULL original MRI (RGB)
    model_input_rgb : (240, 240, 3) f32   — cropped+resized model input
    crop_bbox       : (x1, y1, x2, y2)    — bbox in display_rgb coords
                                            where the cropped region came
                                            from. Used to remap the
                                            heatmap back onto the full
                                            original image.

    Heatmap is computed at the model's feature-map resolution, then:
      1. resized to (bbox_w, bbox_h) — the cropped region's size in
         original coordinates;
      2. pasted into a zero canvas the size of display_rgb at the bbox;
      3. colormapped (JET) and blended with display_rgb inside the bbox.

    A soft mask keeps the overlay confined to the brain region; pixels
    outside the bbox stay as the original MRI rather than turning JET-
    zero blue.

    Returns (overlay_rgb, probs, prediction_idx, heatmap_available).
    """
    assert 0 < interpolant < 1, "Heatmap interpolation must be between 0 and 1"
    skip_classes = set(skip_classes)

    target_layer = _get_last_feature_map_layer(model)

    last_dense = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            last_dense = layer
            break

    img = np.expand_dims(model_input_rgb, axis=0)

    if override_probs is not None:
        prediction = np.asarray(override_probs, dtype=np.float32)
        if prediction.ndim == 1:
            prediction = prediction[np.newaxis, :]
    else:
        prediction = model.predict(img, verbose=0)
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]
    prediction_idx = int(np.argmax(prediction))

    H, W = display_rgb.shape[:2]
    x1, y1, x2, y2 = crop_bbox
    x1 = max(0, min(W, x1))
    x2 = max(0, min(W, x2))
    y1 = max(0, min(H, y1))
    y2 = max(0, min(H, y2))
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)

    # Short-circuit for notumor: flat cold-blue overlay over the FULL
    # original MRI (not the cropped region).
    if prediction_idx in skip_classes:
        probs = prediction[0].tolist() if prediction.ndim == 2 else prediction.tolist()
        flat = np.zeros((H, W), dtype=np.uint8)
        cold_bgr = cv2.applyColorMap(flat, cv2.COLORMAP_JET)
        cold_rgb = cv2.cvtColor(cold_bgr, cv2.COLOR_BGR2RGB)
        flat_overlay = np.uint8(
            display_rgb.astype(np.float32) * interpolant
            + cold_rgb.astype(np.float32) * (1 - interpolant)
        )
        return flat_overlay, probs, prediction_idx, False

    if last_dense is not None:
        gradient_model = Model(
            [model.inputs], [target_layer.output, last_dense.input]
        )
        with tf.GradientTape() as tape:
            conv2d_out, dense_in = gradient_model(img, training=False)
            logits = tf.matmul(dense_in, last_dense.kernel) + last_dense.bias
            loss = logits[:, prediction_idx]
    else:
        gradient_model = Model(
            [model.inputs], [target_layer.output, model.output]
        )
        with tf.GradientTape() as tape:
            conv2d_out, pred2 = gradient_model(img, training=False)
            if isinstance(pred2, (list, tuple)):
                pred2 = pred2[0]
            loss = pred2[:, prediction_idx]

    gradients = tape.gradient(loss, conv2d_out)

    # HiResCAM / Layer-CAM per-pixel weighting (no channel averaging).
    A = conv2d_out[0]                                              # (h, w, C)
    dY = gradients[0]                                              # (h, w, C)
    relu_grads = tf.maximum(dY, 0.0)
    activation_map_small = tf.reduce_sum(relu_grads * A, axis=-1)  # (h, w)
    activation_map_small = tf.maximum(activation_map_small, 0.0).numpy().astype(np.float32)

    # Resize the activation map to the bbox's size in original coords.
    amap_bbox = cv2.resize(activation_map_small, (bbox_w, bbox_h))
    amap_bbox = np.maximum(amap_bbox, 0)

    # Paste into a full-image canvas at the bbox location.
    activation_map = np.zeros((H, W), dtype=np.float32)
    activation_map[y1:y2, x1:x2] = amap_bbox

    blur_sigma = max(2.0, bbox_h / 30.0)
    activation_map = cv2.GaussianBlur(
        activation_map, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma
    )

    amap_max = float(activation_map.max())
    probs = prediction[0].tolist() if prediction.ndim == 2 else prediction.tolist()

    if amap_max <= 1e-6:
        return display_rgb.copy(), probs, prediction_idx, False

    activation_map = (activation_map - activation_map.min()) / (
        amap_max - activation_map.min() + 1e-8
    )
    activation_map_u8 = np.uint8(255 * activation_map)

    heatmap_bgr = cv2.applyColorMap(activation_map_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend the heatmap over the FULL original MRI uniformly. Inside the
    # bbox the colormap carries the actual Grad-CAM signal (warm =
    # important); outside the bbox the activation is zero, so JET maps
    # it to its cold-blue endpoint. The whole image picks up a uniform
    # blue tint, with the warm region only where the model actually
    # looked. No mask, no fade — matches the visual style of a regular
    # Grad-CAM overlay applied to the full picture.
    overlay_rgb = np.uint8(
        display_rgb.astype(np.float32) * interpolant
        + heatmap_rgb.astype(np.float32) * (1 - interpolant)
    )

    return overlay_rgb, probs, prediction_idx, True


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

        display_rgb, model_input_rgb, crop_bbox = preprocess_for_model(img_bgr)

        m = get_model()  # lazy-load stays the same

        # 2-view TTA (original + horizontal flip) on the 240x240 input.
        img_batch = np.expand_dims(model_input_rgb, axis=0)
        img_flip = img_batch[:, :, ::-1, :]
        probs_orig = m.predict(img_batch, verbose=0)[0]
        probs_flip = m.predict(img_flip, verbose=0)[0]
        probs_tta = (probs_orig + probs_flip) / 2.0

        skip = {CLASS_NAMES.index("notumor")} if "notumor" in CLASS_NAMES else set()
        overlay_rgb, probs, idx, heatmap_available = VizGradCAM_API(
            m, display_rgb, model_input_rgb, crop_bbox,
            interpolant=0.5, skip_classes=skip, override_probs=probs_tta,
        )

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
            "heatmap_available": heatmap_available,
            "validation": debug
        }

    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)
