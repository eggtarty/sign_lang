import base64
import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from typing import List, Literal, Optional

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Load models + labels
# -----------------------
static_model = load_model("model_static.h5")
dynamic_model = load_model("model_dynamic.h5")
static_labels = np.load("labels_static.npy", allow_pickle=True)
dynamic_labels = np.load("labels_dynamic.npy", allow_pickle=True)

# -----------------------
# Mediapipe Setup
# -----------------------
mp_hands = mp.solutions.hands

# Better for single images
hands_static = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)

# Better for continuous stream
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

def _decode_base64_image(b64_string: str):
    """
    Accepts 'data:image/jpeg;base64,...' and returns OpenCV BGR image.
    """
    if not b64_string:
        return None

    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    try:
        nparr = np.frombuffer(base64.b64decode(b64_string), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def extract_landmarks_from_frame(b64_string: str, use_static: bool):
    """
    Returns a flat list of 63 floats (21 landmarks * x,y,z) or None.
    """
    img = _decode_base64_image(b64_string)
    if img is None:
        return None

    # Flip ONCE on backend so the model sees the orientation it was trained on.
    # (Frontend mirrors for display only, not for capture.)
    img = cv2.flip(img, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hands = hands_static if use_static else hands_dynamic
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None

    lms = []
    for lm in results.multi_hand_landmarks[0].landmark:
        lms.extend([lm.x, lm.y, lm.z])
    return lms

@app.post("/predict")
async def predict(data: GestureRequest):
    frames = data.frames or []
    mode = (data.mode or "auto").lower()

    if len(frames) == 0:
        return {"success": False, "prediction": "No frames received", "confidence": 0.0}

    # Decide route
    # - explicit static => static
    # - explicit dynamic => dynamic
    # - auto => if 1 frame => static else dynamic
    if mode == "static" or (mode == "auto" and len(frames) == 1):
        lms = extract_landmarks_from_frame(frames[0], use_static=True)
        if not lms:
            return {"success": False, "prediction": "No hand detected", "confidence": 0.0}

        pred = static_model.predict(np.array([lms], dtype=np.float32), verbose=0)
        idx = int(np.argmax(pred))
        conf = float(np.max(pred))
        label = str(static_labels[idx])
        return {"success": True, "prediction": label, "confidence": conf}

    # Dynamic path
    seq = []
    # Use last 30 frames
    for f in frames[-30:]:
        lms = extract_landmarks_from_frame(f, use_static=False)
        if lms is not None:
            seq.append(lms)

    if len(seq) < 20:
        return {"success": False, "prediction": "Incomplete motion (need clearer movement)", "confidence": 0.0}

    # Pad to 30
    while len(seq) < 30:
        seq.append(seq[-1])

    arr = np.array([seq], dtype=np.float32)  # shape (1, 30, 63)
    pred = dynamic_model.predict(arr, verbose=0)
    idx = int(np.argmax(pred))
    conf = float(np.max(pred))
    label = str(dynamic_labels[idx])
    return {"success": True, "prediction": label, "confidence": conf}
