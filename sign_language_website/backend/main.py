import os
import io
import base64
import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

# ===============================
# FASTAPI APP
# ===============================
app = FastAPI(title="Sign Language Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Netlify + local
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# CONFIG
# ===============================
SEQ_LEN = 30
IMG_SIZE = 224

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

STATIC_MODEL_PATH = os.path.join(MODELS_DIR, "static_model.keras")
DYNAMIC_MODEL_PATH = os.path.join(MODELS_DIR, "dynamic_model.keras")

STATIC_LABELS_PATH = os.path.join(MODELS_DIR, "labels_static.npy")
DYNAMIC_LABELS_PATH = os.path.join(MODELS_DIR, "labels_dynamic.npy")

# ===============================
# LOAD MODELS
# ===============================
try:
    static_model = tf.keras.models.load_model(STATIC_MODEL_PATH)
    dynamic_model = tf.keras.models.load_model(DYNAMIC_MODEL_PATH)

    static_labels = np.load(STATIC_LABELS_PATH, allow_pickle=True)
    dynamic_labels = np.load(DYNAMIC_LABELS_PATH, allow_pickle=True)

    print("✅ Models loaded successfully")
    print(f"✅ Static labels: {len(static_labels)}")
    print(f"✅ Dynamic labels: {len(dynamic_labels)}")

except Exception as e:
    print("❌ Failed to load models:", e)
    static_model = None
    dynamic_model = None
    static_labels = []
    dynamic_labels = []

# ===============================
# REQUEST SCHEMA
# ===============================
class PredictRequest(BaseModel):
    image: str
    mode: str = "static"   # static | dynamic | auto

# ===============================
# UTILS
# ===============================
def decode_image(base64_str: str):
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        image_bytes = base64.b64decode(base64_str)
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image.astype("float32") / 255.0

        return image

    except Exception:
        return None

# ===============================
# ROUTES
# ===============================
@app.get("/")
def root():
    return {"message": "Sign Language API is running"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "static_loaded": static_model is not None,
        "dynamic_loaded": dynamic_model is not None,
        "static_labels_count": len(static_labels),
        "dynamic_labels_count": len(dynamic_labels),
    }

@app.get("/gestures")
def gestures():
    return {
        "static": static_labels.tolist(),
        "dynamic": dynamic_labels.tolist(),
    }

@app.post("/predict")
def predict(req: PredictRequest):
    if static_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    image = decode_image(req.image)
    if image is None:
        return {
            "success": False,
            "prediction": "No hand detected",
            "confidence": 0.0,
        }

    image = np.expand_dims(image, axis=0)

    # Static prediction (default)
    preds = static_model.predict(image, verbose=0)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    label = str(static_labels[idx])

    return {
        "success": True,
        "prediction": label,
        "confidence": confidence,
    }

# ===============================
# RENDER ENTRYPOINT (CRITICAL)
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
