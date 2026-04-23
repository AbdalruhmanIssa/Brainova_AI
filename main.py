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
        "area_ratio": float(area_ratio)
    }


# ==========================
# Grad-CAM
# ==========================
def _get_last_feature_map_layer(model):
    """
    Return the last layer whose output is a 4D feature map (N, H, W, C).

    For EfficientNetB1 this resolves to 'top_activation' (post-BN,
    post-swish). Using the deepest 4D layer maximises class-
    discriminativeness — the heatmap highlights tumor-class features,
    not generic edges or textures that earlier layers respond to.

    Spatial coarseness (8x8 for this model) is handled by the
    Grad-CAM++ weighting in VizGradCAM_API, which concentrates the
    activation on the strongest positive-gradient locations instead
    of averaging over the whole feature map.
    """
    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
        except Exception:
            continue
        if shape is not None and len(shape) == 4:
            return layer
    return next(x for x in model.layers[::-1] if isinstance(x, K.layers.Conv2D))


def VizGradCAM_API(model, display_rgb, model_input_rgb, interpolant=0.5,
                   skip_classes=(), override_probs=None):
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
      override_probs   : optional precomputed softmax probabilities to
                         use in place of model.predict() inside this
                         function. Used for Test-Time Augmentation: the
                         caller runs the model on original+flipped
                         inputs, averages the softmax outputs, and
                         passes the averaged vector in here. Grad-CAM
                         itself still runs on the ORIGINAL (unflipped)
                         input, so the heatmap stays anatomically
                         correct while the classification benefits from
                         TTA stability.
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

    if override_probs is not None:
        # TTA path: use the caller's pre-averaged softmax vector.
        prediction = np.asarray(override_probs, dtype=np.float32)
        if prediction.ndim == 1:
            prediction = prediction[np.newaxis, :]
    else:
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

    # ---- HiResCAM / Layer-CAM weighting ----
    # Grad-CAM and Grad-CAM++ both aggregate gradients across space to
    # produce per-channel SCALAR weights, then multiply each channel's
    # full activation map by that scalar. That channel-level step
    # blurs spatial detail — two pixels in the same channel get the
    # same weight regardless of where they fall, so the heatmap peak
    # drifts toward the channel's spatial center of mass.
    #
    # HiResCAM / Layer-CAM skips channel-level aggregation entirely.
    # Each pixel's contribution is computed directly from its own
    # gradient and activation at that exact location:
    #
    #   cam_ij = ReLU( sum_k ReLU(grads_ij,k) * A_ij,k )
    #
    # No averaging across space. Preserves the spatial detail that
    # earlier CAM variants lose. For small or off-center tumors, the
    # pixels that contribute are exactly those where gradient AND
    # activation are both positive at the tumor location — the heatmap
    # locks onto that region instead of spreading out or drifting.
    A = conv2d_out[0]                                              # (H, W, C)
    dY = gradients[0]                                              # (H, W, C)

    relu_grads = tf.maximum(dY, 0.0)                               # (H, W, C)
    activation_map = tf.reduce_sum(relu_grads * A, axis=-1)        # (H, W)
    activation_map = tf.maximum(activation_map, 0.0).numpy().astype(np.float32)

    activation_map = cv2.resize(
        activation_map, (display_rgb.shape[1], display_rgb.shape[0])
    )
    activation_map = np.maximum(activation_map, 0)

    # HiResCAM's per-pixel weighting produces a very tight peak —
    # correct location, but often only covers the center of a tumor
    # rather than its full extent. A mild Gaussian blur widens the
    # warm region around the peak without moving it, so the heatmap
    # visually covers the tumor instead of just marking its middle.
    # sigma scales with image size so coverage stays consistent across
    # different input resolutions. Bump the divisor (30 → 20) for
    # wider spread, lower it (30 → 40) for a tighter peak.
    blur_sigma = max(2.0, display_rgb.shape[0] / 30.0)
    activation_map = cv2.GaussianBlur(
        activation_map, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma
    )

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

        # Test-Time Augmentation: run the model on the original AND a
        # horizontally-flipped copy, then average the softmax outputs.
        # Training used horizontal_flip=True augmentation, so the model
        # is flip-invariant by design. Averaging two views pulls the
        # final prediction toward the consistent class signal and away
        # from noise — in particular, this helps ring-enhancing glioma
        # cases that a single-view pass sometimes mis-routes to notumor.
        img_batch = np.expand_dims(model_input_rgb, axis=0)
        img_flip = img_batch[:, :, ::-1, :]
        probs_orig = m.predict(img_batch, verbose=0)[0]
        probs_flip = m.predict(img_flip, verbose=0)[0]
        probs_tta = (probs_orig + probs_flip) / 2.0

        # Skip Grad-CAM for "notumor" — there are no positive features
        # to highlight, so the heatmap collapses into a misleading blue.
        skip = {CLASS_NAMES.index("notumor")} if "notumor" in CLASS_NAMES else set()
        overlay_rgb, probs, idx, heatmap_available = VizGradCAM_API(
            m, display_rgb, model_input_rgb, interpolant=0.5,
            skip_classes=skip, override_probs=probs_tta,
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
