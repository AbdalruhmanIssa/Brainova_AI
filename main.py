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
INPUT_SIZE = (240, 240)  # (W, H) — must match training

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
def decode_image(file_bytes: bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image. Make sure it's a valid JPG/PNG.")
    return img_bgr


def crop_brain_region(img_bgr: np.ndarray) -> np.ndarray:
    """
    Port of the training-time crop_image() from the Colab notebook.
    Finds the bounding box of the largest bright contour (the brain)
    and crops the surrounding black background.

    Training used this step on EVERY image before resizing to 240x240.
    We must reproduce it at inference, otherwise the model sees a
    zoomed-out brain at different pixel coordinates than it learned
    during training, and Grad-CAM heatmaps land on the wrong spot.

    Falls back to the original image if the contour search fails.
    """
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # cv2 4.x returns (contours, hierarchy) directly — no need for imutils.
        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return img_bgr

        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 100:
            return img_bgr

        ext_left = tuple(c[c[:, :, 0].argmin()][0])
        ext_right = tuple(c[c[:, :, 0].argmax()][0])
        ext_top = tuple(c[c[:, :, 1].argmin()][0])
        ext_bottom = tuple(c[c[:, :, 1].argmax()][0])

        cropped = img_bgr[ext_top[1]:ext_bottom[1], ext_left[0]:ext_right[0]]
        if cropped.size == 0 or cropped.shape[0] < 5 or cropped.shape[1] < 5:
            return img_bgr
        return cropped
    except Exception:
        return img_bgr


def preprocess_for_model(img_bgr: np.ndarray):
    """
    Reproduces the training pipeline exactly:
      1. crop_brain_region     (same as notebook)
      2. cv2.resize to 240x240 (same as notebook)
      3. BGR -> RGB            (training loaded via PIL which gives RGB)
      4. float32 in 0..255     (EfficientNet has a Rescaling layer baked in;
                                do NOT divide by 255)

    Returns:
      display_rgb_uint8    : (240,240,3) uint8   — for building the overlay
      model_input_rgb_f32  : (240,240,3) float32 — feed to the model
    """
    cropped_bgr = crop_brain_region(img_bgr)
    resized_bgr = cv2.resize(cropped_bgr, INPUT_SIZE)
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    display_rgb_uint8 = resized_rgb.copy()
    model_input_rgb_f32 = img_to_array(resized_rgb).astype(np.float32)
    return display_rgb_uint8, model_input_rgb_f32


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
    """
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
        """

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
     #   "dark_ratio": float(dark_ratio),
        "area_ratio": float(area_ratio)
    }


# ==========================
# Grad-CAM
# ==========================
def _get_last_feature_map_layer(model):
    """
    Return the last layer whose output is a 4D feature map (N, H, W, C).

    This is the correct Grad-CAM target — not the last Conv2D. In
    EfficientNet the last Conv2D is 'top_conv', whose output is
    pre-BatchNorm and pre-swish. Those raw values can be uniformly
    negative for the class-relevant channels, so after the ReLU step
    in Grad-CAM the whole heatmap collapses to zero and the JET
    colormap renders as solid dark blue.

    For EfficientNet the layer we want is 'top_activation'
    (post-BN, post-swish), whose features are non-negative and
    behave well for Grad-CAM.
    """
    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
        except Exception:
            continue
        if shape is not None and len(shape) == 4:
            return layer
    # Fallback — shouldn't happen in practice.
    return next(x for x in model.layers[::-1] if isinstance(x, K.layers.Conv2D))


def VizGradCAM_API(model, display_rgb, model_input_rgb, interpolant=0.5, skip_classes=()):
    """
    Grad-CAM over the last spatial feature map. Both inputs are RGB:
      display_rgb      : (H,W,3) uint8    — image the heatmap is drawn on
      model_input_rgb  : (H,W,3) float32  — tensor fed to the model
      skip_classes     : iterable of class indices for which Grad-CAM
                         should be skipped entirely. Used for 'notumor':
                         the model classifies that case by ABSENCE of
                         tumor features, so there's nothing positive
                         for Grad-CAM to highlight, and the result is
                         a misleading dead-blue overlay.
    Returns (overlay_rgb, probs, prediction_idx, heatmap_available).
    """
    assert 0 < interpolant < 1, "Heatmap interpolation must be between 0 and 1"
    skip_classes = set(skip_classes)

    target_layer = _get_last_feature_map_layer(model)

    # Grad-CAM must take gradients w.r.t. the PRE-softmax logits, not
    # the post-softmax probabilities. With softmax the derivative is
    # p*(1-p), which collapses to zero when the top class sits at
    # p ≈ 1.0 — exactly what happens for confident tumor predictions.
    # We sidestep that by re-computing the logits manually from the
    # last Dense layer's input and weights, inside the tape.
    last_dense = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            last_dense = layer
            break

    img = np.expand_dims(model_input_rgb, axis=0)

    prediction = model.predict(img, verbose=0)
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    prediction_idx = int(np.argmax(prediction))

    # Short-circuit for classes where Grad-CAM is meaningless (e.g. notumor).
    # Render a flat "cold" overlay (uniform JET-zero blue blended with the
    # MRI) so the result is visually consistent with normal heatmap output
    # but clearly conveys "no positive evidence found".
    if prediction_idx in skip_classes:
        probs = prediction[0].tolist() if prediction.ndim == 2 else prediction.tolist()
        flat = np.zeros((display_rgb.shape[0], display_rgb.shape[1]), dtype=np.uint8)
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

    output = conv2d_out[0]
    weights = tf.reduce_mean(gradients[0], axis=(0, 1))

    activation_map = np.zeros(output.shape[0:2], dtype=np.float32)
    for idx, weight in enumerate(weights):
        activation_map += float(weight) * output[:, :, idx].numpy()

    activation_map = cv2.resize(
        activation_map, (display_rgb.shape[1], display_rgb.shape[0])
    )
    activation_map = np.maximum(activation_map, 0)

    amap_max = float(activation_map.max())
    probs = prediction[0].tolist() if prediction.ndim == 2 else prediction.tolist()

    # If the ReLU zeroed everything out, the heatmap carries no signal.
    # Return the plain image rather than a misleading solid-blue overlay.
    if amap_max <= 1e-6:
        return display_rgb.copy(), probs, prediction_idx, False

    activation_map = (activation_map - activation_map.min()) / (
        amap_max - activation_map.min() + 1e-8
    )
    activation_map = np.uint8(255 * activation_map)

    # applyColorMap returns BGR — convert to RGB so it blends with display_rgb.
    heatmap_bgr = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

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

        # MRI-like validation runs on the ORIGINAL uploaded image — the
        # dark-border check would fail after cropping.
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

        display_rgb, model_input_rgb = preprocess_for_model(img_bgr)

        m = get_model()  # lazy-load stays the same
        # Skip Grad-CAM for "notumor" — there are no positive features
        # to highlight, so the heatmap collapses into a misleading blue.
        skip = {CLASS_NAMES.index("notumor")} if "notumor" in CLASS_NAMES else set()
        overlay_rgb, probs, idx, heatmap_available = VizGradCAM_API(
            m, display_rgb, model_input_rgb, interpolant=0.5,
            skip_classes=skip,
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
