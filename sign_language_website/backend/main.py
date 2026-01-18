import os
import base64
import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import mediapipe as mp


# ===============================
# FASTAPI APP
# ===============================
app = FastAPI(title="Sign Language Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,
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
if not os.path.isdir(MODELS_DIR):
    MODELS_DIR = BASE_DIR

STATIC_MODEL_PATH = os.path.join(MODELS_DIR, "static_model.keras")
DYNAMIC_MODEL_PATH = os.path.join(MODELS_DIR, "dynamic_model.keras")
STATIC_LABELS_PATH = os.path.join(MODELS_DIR, "labels_static.npy")
DYNAMIC_LABELS_PATH = os.path.join(MODELS_DIR, "labels_dynamic.npy")

# =============
# LOAD MODELS 
# =============
static_model = None
dynamic_model = None
static_labels = np.array([])
dynamic_labels = np.array([])

try:
    # 1. Use compile=False to bypass the version conflict error
    static_model = load_model(STATIC_MODEL_PATH, compile=False)
    dynamic_model = load_model(DYNAMIC_MODEL_PATH, compile=False)

    # 2. Re-compile the models so they are ready for real-time predictions
    static_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    dynamic_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 3. Load .npy label files
    static_labels = np.load(STATIC_LABELS_PATH, allow_pickle=True)
    dynamic_labels = np.load(DYNAMIC_LABELS_PATH, allow_pickle=True)

    print("✅ Models loaded successfully with Compatibility Fix")
    print(f"✅ Static model input: {static_model.input_shape}")
    print(f"✅ Dynamic model input: {dynamic_model.input_shape}")
    
except Exception as e:
    print(f"❌ Failed to load models: {e}")

# ===============================
# MEDIAPIPE INIT
# ===============================
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ===============================
# REQUEST SCHEMAS
# ===============================
class PredictRequest(BaseModel):
    image: str
    mode: str = "static" 


class PredictDynamicRequest(BaseModel):
    frames: list[str] 


# ===============================
# UTILS
# ===============================
def decode_image_to_rgb_float(base64_str: str):
    """
    Returns RGB float image [0..1], shape (H,W,3) resized to IMG_SIZE.
    """
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        image_bytes = base64.b64decode(base64_str)
        image_np = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if bgr is None:
            return None

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        rgb = rgb.astype("float32") / 255.0
        return rgb
    except Exception:
        return None


def extract_hand_landmarks_vector(base64_str: str, target_len: int):
    """
    Extracts MediaPipe hand landmarks, returns a 1D vector of length target_len.
    If no hand found -> returns None.
    """
    rgb = decode_image_to_rgb_float(base64_str)
    if rgb is None:
        return None

    # MediaPipe expects uint8 RGB
    rgb_u8 = (rgb * 255).astype(np.uint8)
    results = hands_detector.process(rgb_u8)

    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]
    coords = []
    for lm in hand.landmark:
        coords.extend([lm.x, lm.y, lm.z])  # 21 * 3 = 63 values

    vec = np.array(coords, dtype=np.float32)

    # pad / truncate to match model input size
    if vec.shape[0] < target_len:
        vec = np.pad(vec, (0, target_len - vec.shape[0]), mode="constant")
    elif vec.shape[0] > target_len:
        vec = vec[:target_len]

    return vec


def model_expects_image(model) -> bool:
    """
    If input shape looks like (None, H, W, C) => image model.
    If input shape looks like (None, N) or (None, T, N) => feature model.
    """
    shp = model.input_shape
    # shp can be tuple or list of tuples
    if isinstance(shp, list):
        shp = shp[0]
    return isinstance(shp, tuple) and len(shp) == 4


def get_static_feature_len():
    shp = static_model.input_shape
    if isinstance(shp, list):
        shp = shp[0]
    # (None, N)
    return int(shp[-1])


def get_dynamic_feature_len_and_seq():
    shp = dynamic_model.input_shape
    if isinstance(shp, list):
        shp = shp[0]
    seq = int(shp[1])
    feat = int(shp[2])
    return seq, feat


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
        "static_input_shape": getattr(static_model, "input_shape", None),
        "dynamic_input_shape": getattr(dynamic_model, "input_shape", None),
        "static_labels_count": int(len(static_labels)) if static_labels is not None else 0,
        "dynamic_labels_count": int(len(dynamic_labels)) if dynamic_labels is not None else 0,
    }


@app.get("/gestures")
def gestures():
    return {
        "static": static_labels.tolist() if static_labels is not None else [],
        "dynamic": dynamic_labels.tolist() if dynamic_labels is not None else [],
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if static_model is None:
        raise HTTPException(status_code=500, detail="Static model not loaded")

    try:
        # Decide pipeline based on model input type
        if model_expects_image(static_model):
            img = decode_image_to_rgb_float(req.image)
            if img is None:
                return {"success": False, "prediction": "Invalid image", "confidence": 0.0}

            x = np.expand_dims(img, axis=0)  # (1,224,224,3)

        else:
            feat_len = get_static_feature_len()
            vec = extract_hand_landmarks_vector(req.image, feat_len)
            if vec is None:
                return {"success": False, "prediction": "No hand detected", "confidence": 0.0}

            x = np.expand_dims(vec, axis=0)  # (1,N)

        preds = static_model.predict(x, verbose=0)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])
        label = str(static_labels[idx]) if idx < len(static_labels) else str(idx)

        return {"success": True, "prediction": label, "confidence": confidence}

    except Exception as e:
        print("❌ /predict error:", repr(e))
        raise HTTPException(status_code=500, detail=f"Predict failed: {e}")


@app.post("/predict/dynamic")
def predict_dynamic(req: PredictDynamicRequest):
    if dynamic_model is None:
        raise HTTPException(status_code=500, detail="Dynamic model not loaded")

    try:
        # If dynamic model expects images sequence: (None,T,H,W,C)
        shp = dynamic_model.input_shape
        if isinstance(shp, list):
            shp = shp[0]

        if len(shp) == 5:
            # image-sequence model
            seq_len = int(shp[1])
            frames = req.frames[:seq_len]
            if len(frames) < seq_len:
                # pad with last frame if missing
                if not frames:
                    return {"success": False, "prediction": "No frames", "confidence": 0.0}
                frames = frames + [frames[-1]] * (seq_len - len(frames))

            imgs = []
            for f in frames:
                img = decode_image_to_rgb_float(f)
                if img is None:
                    return {"success": False, "prediction": "Invalid frame", "confidence": 0.0}
                imgs.append(img)

            x = np.expand_dims(np.array(imgs, dtype=np.float32), axis=0)  # (1,T,224,224,3)

        else:
            # feature-sequence model: (None,T,N)
            seq_len, feat_len = get_dynamic_feature_len_and_seq()
            frames = req.frames[:seq_len]
            if len(frames) < seq_len:
                if not frames:
                    return {"success": False, "prediction": "No frames", "confidence": 0.0}
                frames = frames + [frames[-1]] * (seq_len - len(frames))

            seq = []
            for f in frames:
                vec = extract_hand_landmarks_vector(f, feat_len)
                if vec is None:
                    # if a frame has no hand
                    vec = np.zeros((feat_len,), dtype=np.float32)
                seq.append(vec)

            x = np.expand_dims(np.array(seq, dtype=np.float32), axis=0)  # (1,T,N)

        preds = dynamic_model.predict(x, verbose=0)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])
        label = str(dynamic_labels[idx]) if idx < len(dynamic_labels) else str(idx)

        return {"success": True, "prediction": label, "confidence": confidence}

    except Exception as e:
        print("❌ /predict/dynamic error:", repr(e))
        raise HTTPException(status_code=500, detail=f"Dynamic predict failed: {e}")


# ===============================
# RENDER ENTRYPOINT
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

