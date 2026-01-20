import os
import base64
import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

# =========================
# App
# =========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

STATIC_MODEL_PATH = os.path.join(MODELS_DIR, "static_model.keras")
DYNAMIC_MODEL_PATH = os.path.join(MODELS_DIR, "dynamic_model.keras")
STATIC_LABELS_PATH = os.path.join(MODELS_DIR, "labels_static.npy")
DYNAMIC_LABELS_PATH = os.path.join(MODELS_DIR, "labels_dynamic.npy")

SEQ_LEN = 30
MIN_DYNAMIC_FRAMES = 20

# =========================
# Load models
# =========================
static_model = load_model(STATIC_MODEL_PATH)
dynamic_model = load_model(DYNAMIC_MODEL_PATH)
static_labels = np.load(STATIC_LABELS_PATH, allow_pickle=True)
dynamic_labels = np.load(DYNAMIC_LABELS_PATH, allow_pickle=True)

# =========================
# Mediapipe
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =========================
# Request schema (frontend sends {frames: [...]})
# =========================
class GestureRequest(BaseModel):
    frames: list[str]

@app.get("/health")
def health():
    return {"ok": True}

# =========================
# Helpers
# =========================
def decode_frame(b64_string: str):
    encoded_data = b64_string.split(",")[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def extract_coords(img_bgr) -> np.ndarray | None:
    """
    IMPORTANT:
    Your ORIGINAL frontend already mirrors the canvas before sending frames.
    app.py flips the webcam frame before mediapipe.
    So here we DO NOT flip again (avoid double-mirror).
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None
    h_lms = results.multi_hand_landmarks[0]
    coords = np.array([[lm.x, lm.y, lm.z] for lm in h_lms.landmark], dtype=np.float32)
    return coords

def get_120_features(coords_seq: list[np.ndarray]) -> np.ndarray:
    """
    Same idea as app.py:
    - normalize by wrist
    - scale normalize
    - bones(60) + velocity(60) = 120 per frame
    """
    all_frame_feats = []
    for frame in coords_seq:
        wrist = frame[0]
        norm_frame = frame - wrist
        scale = np.max(np.linalg.norm(norm_frame, axis=1))
        if scale > 0:
            norm_frame /= scale
        bones = np.array([norm_frame[i] - norm_frame[0] for i in range(1, 21)], dtype=np.float32).flatten()
        all_frame_feats.append(bones)

    all_frame_feats = np.array(all_frame_feats, dtype=np.float32)  # (T, 60)

    if all_frame_feats.shape[0] < 2:
        velocity = np.zeros_like(all_frame_feats)
    else:
        velocity = np.diff(all_frame_feats, axis=0)
        velocity = np.vstack([velocity, np.zeros((1, 60), dtype=np.float32)])

    return np.concatenate([all_frame_feats, velocity], axis=1)  # (T, 120)

def pad_to_len(seq: list[np.ndarray], target_len: int) -> list[np.ndarray]:
    if len(seq) >= target_len:
        return seq[-target_len:]
    if len(seq) == 0:
        return seq
    return seq + [seq[-1]] * (target_len - len(seq))

def motion_intensity(features_120: np.ndarray, window_size=5) -> float:
    if len(features_120) < window_size:
        return 0.0
    vel = features_120[-window_size:, 60:]  # last velocities
    mags = np.linalg.norm(vel, axis=1)
    return float(np.mean(mags))

@app.post("/predict")
async def predict(data: GestureRequest):
    # Decode frames -> coords
    coords_seq = []
    for f in data.frames[-SEQ_LEN:]:
        try:
            img = decode_frame(f)
            coords = extract_coords(img)
            if coords is not None:
                coords_seq.append(coords)
        except:
            pass

    if len(coords_seq) == 0:
        return {"success": False, "prediction": "No Hand", "confidence": 0.0, "mode": "NoHand"}

    # If frontend sends exactly 1 frame -> force static
    if len(data.frames) == 1:
        feats = get_120_features(coords_seq)
        x = feats[-1].reshape(1, 120)
        pred = static_model.predict(x, verbose=0)
        idx = int(np.argmax(pred))
        conf = float(np.max(pred))
        return {
            "success": True,
            "prediction": str(static_labels[idx]),
            "confidence": conf,
            "mode": "Static"
        }

    # Otherwise (auto/dynamic/static buffer):
    feats_120 = get_120_features(coords_seq)
    mot = motion_intensity(feats_120, window_size=5)

    # Auto switch: if motion small => treat as static, else dynamic
    MOTION_THRESHOLD = 0.15

    if mot < MOTION_THRESHOLD:
        # static prediction from the latest frame features
        x = feats_120[-1].reshape(1, 120)
        pred = static_model.predict(x, verbose=0)
        idx = int(np.argmax(pred))
        conf = float(np.max(pred))
        return {
            "success": True,
            "prediction": str(static_labels[idx]),
            "confidence": conf,
            "mode": "Static"
        }

    # dynamic prediction needs enough frames
    if len(coords_seq) < MIN_DYNAMIC_FRAMES:
        return {"success": False, "prediction": "Incomplete Motion", "confidence": 0.0, "mode": "Dynamic"}

    coords_seq = pad_to_len(coords_seq, SEQ_LEN)
    feats_120_dyn = get_120_features(coords_seq)
    x = feats_120_dyn.reshape(1, SEQ_LEN, 120)

    pred = dynamic_model.predict(x, verbose=0)
    idx = int(np.argmax(pred))
    conf = float(np.max(pred))
    return {
        "success": True,
        "prediction": str(dynamic_labels[idx]),
        "confidence": conf,
        "mode": "Dynamic"
    }
