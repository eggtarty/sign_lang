import os
import time
import base64
import uuid
from collections import deque
from typing import Optional, Dict

import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

# ======================
# PATHS
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

SEQ_LEN = 30
MIN_DYNAMIC_FRAMES = 20

STATIC_MODEL_PATH = os.path.join(MODELS_DIR, "static_model.keras")
DYNAMIC_MODEL_PATH = os.path.join(MODELS_DIR, "dynamic_model.keras")
STATIC_LABELS_PATH = os.path.join(MODELS_DIR, "labels_static.npy")
DYNAMIC_LABELS_PATH = os.path.join(MODELS_DIR, "labels_dynamic.npy")

# ======================
# FASTAPI
# ======================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# LOAD MODELS + LABELS
# ======================
static_model = load_model(STATIC_MODEL_PATH)
dynamic_model = load_model(DYNAMIC_MODEL_PATH)
static_labels = np.load(STATIC_LABELS_PATH, allow_pickle=True)
dynamic_labels = np.load(DYNAMIC_LABELS_PATH, allow_pickle=True)

# ======================
# MEDIAPIPE
# ======================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ======================
# FEATURE PIPELINE 
# ======================
def get_120_features(coords_seq):
    all_frame_feats = []
    for frame in coords_seq:
        wrist = frame[0]
        norm_frame = frame - wrist
        scale = np.max(np.linalg.norm(norm_frame, axis=1))
        if scale > 0:
            norm_frame /= scale
        bones = np.array([norm_frame[i] - norm_frame[0] for i in range(1, 21)]).flatten()
        all_frame_feats.append(bones)

    all_frame_feats = np.array(all_frame_feats)

    if all_frame_feats.shape[0] < 2:
        velocity = np.zeros_like(all_frame_feats)
    else:
        velocity = np.diff(all_frame_feats, axis=0)
        velocity = np.vstack([velocity, np.zeros((1, 60))])

    return np.concatenate([all_frame_feats, velocity], axis=1)  # (T, 120)

def calculate_motion_intensity(features_120, window_size=3):
    if len(features_120) < window_size:
        return 0.0
    velocities = features_120[-window_size:, 60:]
    motion_magnitudes = np.linalg.norm(velocities, axis=1)
    return float(np.mean(motion_magnitudes))

def pad_sequence_to_length(sequence, target_length):
    if len(sequence) >= target_length:
        return sequence[-target_length:]
    padding_needed = target_length - len(sequence)
    padding = [sequence[-1]] * padding_needed
    return sequence + padding

# ==============================================
# SERVER-SIDE STATE MACHINE 
# ==============================================
class GestureDetector:
    def __init__(self, seq_len=30, min_dynamic_frames=20):
        self.seq_len = seq_len
        self.min_dynamic_frames = min_dynamic_frames
        self.coord_buffer = deque(maxlen=seq_len)
        self.last_move_time = time.time()
        self.last_prediction_time = 0.0
        self.current_mode = "Initializing"
        self.prediction_cooldown = 0.8
        self.motion_history = deque(maxlen=10)
        self.is_dynamic_active = False

    def update(self, coords, motion_intensity, motion_threshold=0.15):
        current_time = time.time()

        if coords is not None:
            self.coord_buffer.append(coords)

        if motion_intensity > motion_threshold:
            self.last_move_time = current_time
            self.motion_history.append(True)
            if not self.is_dynamic_active and len(self.coord_buffer) >= self.min_dynamic_frames:
                self.is_dynamic_active = True
        else:
            self.motion_history.append(False)

        still_duration = current_time - self.last_move_time

        if self.is_dynamic_active:
            if still_duration > 0.5:
                self.is_dynamic_active = False
                self.current_mode = "Static"
            else:
                self.current_mode = "Dynamic"
        elif still_duration > 1.6:
            self.current_mode = "Static"
        else:
            self.current_mode = "Transition"

        if self.current_mode == "Static" and len(self.motion_history) > 0 and any(self.motion_history):
            self.coord_buffer.clear()

        return self.current_mode, still_duration

    def can_predict_dynamic(self):
        return (
            self.is_dynamic_active
            and len(self.coord_buffer) >= self.min_dynamic_frames
            and (time.time() - self.last_prediction_time) > self.prediction_cooldown
        )

    def can_predict_static(self):
        still_duration = time.time() - self.last_move_time
        return (
            self.current_mode == "Static"
            and still_duration > 1.6
            and len(self.coord_buffer) > 0
            and (time.time() - self.last_prediction_time) > self.prediction_cooldown
        )

    def record_prediction(self):
        self.last_prediction_time = time.time()

SESSIONS: Dict[str, GestureDetector] = {}
SESSIONS_LAST_SEEN: Dict[str, float] = {}
SESSION_TTL_SECONDS = 120.0

def get_session(session_id: str) -> GestureDetector:
    now = time.time()
    # cleanup
    for sid, ts in list(SESSIONS_LAST_SEEN.items()):
        if now - ts > SESSION_TTL_SECONDS:
            SESSIONS.pop(sid, None)
            SESSIONS_LAST_SEEN.pop(sid, None)

    if session_id not in SESSIONS:
        SESSIONS[session_id] = GestureDetector(seq_len=SEQ_LEN, min_dynamic_frames=MIN_DYNAMIC_FRAMES)

    SESSIONS_LAST_SEEN[session_id] = now
    return SESSIONS[session_id]

# ======================
# REQUEST / RESPONSE
# ======================
class FrameRequest(BaseModel):
    session_id: Optional[str] = None
    frame_b64: str
    motion_threshold: float = 0.15
    static_confidence: float = 0.70
    dynamic_confidence: float = 0.60
    force_mode: str = "auto"  # auto | static | dynamic

def decode_frame(b64_string: str) -> Optional[np.ndarray]:
    try:
        encoded_data = b64_string.split(",")[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except:
        return None

def extract_coords(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    img_bgr = cv2.flip(img_bgr, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None
    h_lms = results.multi_hand_landmarks[0]
    coords = np.array([[lm.x, lm.y, lm.z] for lm in h_lms.landmark], dtype=np.float32)
    return coords

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict_frame")
def predict_frame(req: FrameRequest):
    session_id = req.session_id or str(uuid.uuid4())
    detector = get_session(session_id)

    img = decode_frame(req.frame_b64)
    if img is None:
        return {"success": False, "session_id": session_id, "error": "Bad frame"}

    coords = extract_coords(img)
    if coords is None:
        detector.coord_buffer.clear()
        detector.is_dynamic_active = False
        return {
            "success": True,
            "session_id": session_id,
            "hand": False,
            "mode": "NoHand",
            "label": "No Hand",
            "confidence": 0.0,
            "motion": 0.0,
            "buffer": 0
        }

    # compute motion intensity using your 120-feature velocity logic
    if len(detector.coord_buffer) > 0:
        buffer_list = list(detector.coord_buffer) + [coords]
        feats_120 = get_120_features(buffer_list)
        motion_intensity = calculate_motion_intensity(feats_120, window_size=3)
    else:
        motion_intensity = 0.0
        feats_120 = None

    current_mode, still_duration = detector.update(coords, motion_intensity, req.motion_threshold)

    # Decide mode
    if req.force_mode == "static":
        chosen = "Static"
    elif req.force_mode == "dynamic":
        chosen = "Dynamic"
    else:
        chosen = current_mode

    label = "Ready..."
    confidence = 0.0
    predicted_mode = chosen
    did_predict = False

    # Predict dynamic
    if (chosen == "Dynamic" and detector.can_predict_dynamic()):
        buffer_list = list(detector.coord_buffer)
        if len(buffer_list) < SEQ_LEN:
            buffer_list = pad_sequence_to_length(buffer_list, SEQ_LEN)
        feats_120_dynamic = get_120_features(buffer_list)
        x = feats_120_dynamic.reshape(1, SEQ_LEN, 120)

        pred = dynamic_model.predict(x, verbose=0)
        idx = int(np.argmax(pred))
        confidence = float(np.max(pred))

        if confidence >= req.dynamic_confidence:
            label = str(dynamic_labels[idx])
            did_predict = True
            detector.record_prediction()
        else:
            label = "Low Confidence"

    # Predict static
    elif (chosen == "Static" and detector.can_predict_static() and feats_120 is not None):
        x = feats_120[-1].reshape(1, 120)
        pred = static_model.predict(x, verbose=0)
        idx = int(np.argmax(pred))
        confidence = float(np.max(pred))

        if confidence >= req.static_confidence:
            label = str(static_labels[idx])
            did_predict = True
            detector.record_prediction()
        else:
            label = "Uncertain"

    else:
        if chosen == "Transition":
            if len(detector.coord_buffer) < MIN_DYNAMIC_FRAMES:
                label = f"Move ({len(detector.coord_buffer)}/{MIN_DYNAMIC_FRAMES})"
            else:
                label = "Perform Gesture"
        elif chosen == "Static":
            label = f"Hold still ({still_duration:.1f}s)"

    return {
        "success": True,
        "session_id": session_id,
        "hand": True,
        "mode": predicted_mode,
        "label": label,
        "confidence": confidence,
        "motion": float(motion_intensity),
        "buffer": int(len(detector.coord_buffer)),
        "predicted": did_predict
    }
