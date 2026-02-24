import io
import base64
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse,StreamingResponse
import os, re
import requests
import tensorflow as tf

MODEL_PATH = "model/brainova_model.keras"
MODEL_FILE_ID = os.getenv("MODEL_FILE_ID", "19OrsrlSqZ-pHYWSm_QE9hyF7ykkG0Q1")

def download_from_drive(file_id: str, destination: str):
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    session = requests.Session()
    base_url = "https://drive.google.com/uc"

    r = session.get(base_url, params={"export": "download", "id": file_id}, stream=True)
    r.raise_for_status()

    token = None
    for k, v in r.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break

    if token is None:
        content_type = r.headers.get("Content-Type", "")
        if "text/html" in content_type:
            html = r.text
            m = re.search(r'confirm=([0-9A-Za-z_]+)', html)
            if m:
                token = m.group(1)

    if token:
        r = session.get(base_url, params={"export": "download", "id": file_id, "confirm": token}, stream=True)
        r.raise_for_status()

    if "text/html" in r.headers.get("Content-Type", ""):
        raise RuntimeError("Google Drive returned HTML instead of the file (sharing/permission blocked).")

    with open(destination, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1024 * 1024:
        print("✅ Model already exists:", MODEL_PATH)
        return
    print("⬇️ Downloading model from Google Drive...")
    download_from_drive(MODEL_FILE_ID, MODEL_PATH)
    print("✅ Model downloaded:", MODEL_PATH, "size:", os.path.getsize(MODEL_PATH))

ensure_model()

# Load model once
model = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]  # confirmed by your class_indices




app = FastAPI()


# ==========================
# Preprocess EXACTLY like Colab:
# cv2.imread -> BGR -> img_to_array -> float32 0..255
# ==========================
def preprocess_like_colab(file_bytes: bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR, like cv2.imread
    if img_bgr is None:
        raise ValueError("Could not decode image. Make sure it's a valid JPG/PNG.")

    # Safe resize (your model expects 240x240)
    img_bgr = cv2.resize(img_bgr, (240, 240))

    # img_to_array keeps channel order (BGR here) and gives float32
    arr = img_to_array(img_bgr)  # (240,240,3) float32, values 0..255
    return arr


# ==========================
# Your Grad-CAM (API version)
# - removed matplotlib plotting
# - returns overlay image + probs + label index
# ==========================
def VizGradCAM_API(model, image, interpolant=0.5):
    assert 0 < interpolant < 1, "Heatmap Interpolation Must Be Between 0 - 1"

    last_conv_layer = next(
        x for x in model.layers[::-1] if isinstance(x, K.layers.Conv2D)
    )
    target_layer = model.get_layer(last_conv_layer.name)

    original_img = image
    img = np.expand_dims(original_img, axis=0)

    prediction = model.predict(img, verbose=0)
    # handle list/tuple outputs
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

    activation_map = cv2.resize(
        activation_map, (original_img.shape[1], original_img.shape[0])
    )
    activation_map = np.maximum(activation_map, 0)

    activation_map = (activation_map - activation_map.min()) / (
        activation_map.max() - activation_map.min() + 1e-8
    )
    activation_map = np.uint8(255 * activation_map)

    heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)

    # Same normalization style as your code
    original_img_norm = np.uint8(
        (original_img - original_img.min())
        / (original_img.max() - original_img.min() + 1e-8)
        * 255
    )

    cvt_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay (same formula)
    overlay_rgb = np.uint8(original_img_norm * interpolant + cvt_heatmap * (1 - interpolant))

    probs = prediction[0].tolist() if prediction.ndim == 2 else prediction.tolist()
    return overlay_rgb, probs, prediction_idx


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        img_arr = preprocess_like_colab(contents)

        overlay_rgb, probs, idx = VizGradCAM_API(model, img_arr, interpolant=0.5)

        # Encode overlay as JPG base64
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", overlay_bgr)
        if not ok:
            return JSONResponse({"success": False, "message": "Failed to encode overlay"}, status_code=500)

        overlay_b64 = base64.b64encode(buf).decode("utf-8")

        return {
            "label": CLASS_NAMES[idx],
            "probabilities": probs,
            "gradcam_image_base64": overlay_b64
        }

    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)
