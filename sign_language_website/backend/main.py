import base64
from pathlib import Path
from typing import List, Literal, Optional

import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Paths 
# -----------------------
BASE_DIR = Path(__file__).resolve().parent                  
MODELS_DIR = BASE_DIR / "models"                            

STATIC_MODEL_PATH = MODELS_DIR / "static_model.keras"
DYNAMIC_MODEL_PATH = MODELS_DIR / "dynamic_model.keras"
STATIC_LABELS_PATH = MODELS_DIR / "labels_static.npy"
DYNAMIC_LABELS_PATH = MODELS_DIR / "labels_dynamic.npy"

print("BASE_DIR:", BASE_DIR)
print("MODELS_DIR:", MODELS_DIR)
print("STATIC_MODEL exists:", STATIC_MODEL_PATH.exists(), STATIC_MODEL_PATH)
print("DYNAMIC_MODEL exists:", DYNAMIC_MODEL_PATH.exists(), DYNAMIC_MODEL_PATH)
print("STATIC_LABELS exists:", STATIC_LABELS_PATH.exists(), STATIC_LABELS_PATH)
print("DYNAMIC_LABELS exists:", DYNAMIC_LABELS_PATH.exists(), DYNAMIC_LABELS_PATH)

# -----------------------
# Load models + labels
# -----------------------
static_model = load_model(str(STATIC_MODEL_PATH))
dynamic_model = load_model(str(DYNAMIC_MODEL_PATH))
static_labels = np.load(str(STATIC_LABELS_PATH), allow_pickle=True)
dynamic_labels = np.load(str(DYNAMIC_LABELS_PATH), allow_pickle=True)

# -----------------------
# Mediapipe (two configs = more reliable)
# -----------------------
mp_hands = mp.solutions.hands

hands_static = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)

hands_dynamic = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

class GestureRequest(BaseModel):
    frames: List[str]
    mode: Optional[Literal["static", "dynamic", "auto"]] = "auto"

@app.get("/health")
def health():
    return {"ok": True}

def _decode_data_url(data_url: str):
    """data:image/jpeg;base64,... -> OpenCV BGR image"""
    if not data_url:
        return None
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    try:
        raw = base64.b64decode(data_url)
        nparr = np.frombuffer(raw, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None

def extract_landmarks(data_url: str, use_static: bool):
    img = _decode_data_url(data_url)
    if img is None:
        return None

    # IMPORTANT:
    # Frontend will show mirrored video to the user,
    # but we capture frames normally. We flip ONCE here for model consistency.
    img = cv2.flip(img, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hands = hands_static if use_static else hands_dynamic
    res = hands.process(img_rgb)

    if not res.multi_hand_landmarks:
        return None

    out = []
    for lm in res.multi_hand_landmarks[0].landmark:
        out.extend([lm.x, lm.y, lm.z])
    return out

@app.post("/predict")
async def predict(data: GestureRequest):
    frames = data.frames or []
    mode = (data.mode or "auto").lower()

    if not frames:
        return {"success": False, "prediction": "No frames received", "confidence": 0.0}

    # AUTO routing:
    # - static if only 1 frame
    # - dynamic if >1 frames
    if mode == "static" or (mode == "auto" and len(frames) == 1):
        lms = extract_landmarks(frames[0], use_static=True)
        if not lms:
            return {"success": False, "prediction": "No hand detected", "confidence": 0.0}

        pred = static_model.predict(np.array([lms], dtype=np.float32), verbose=0)
        idx = int(np.argmax(pred))
        return {
            "success": True,
            "prediction": str(static_labels[idx]),
            "confidence": float(np.max(pred)),
        }

    # Dynamic
    seq = []
    for f in frames[-30:]:
        lms = extract_landmarks(f, use_static=False)
        if lms is not None:
            seq.append(lms)

    if len(seq) < 20:
        return {"success": False, "prediction": "Incomplete motion", "confidence": 0.0}

    while len(seq) < 30:
        seq.append(seq[-1])

    arr = np.array([seq], dtype=np.float32)  # (1,30,63)
    pred = dynamic_model.predict(arr, verbose=0)
    idx = int(np.argmax(pred))
    return {
        "success": True,
        "prediction": str(dynamic_labels[idx]),
        "confidence": float(np.max(pred)),
    }
