import os
import base64
import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.models import load_model

import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
SEQ_LEN = 30
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# --- Load Assets ---
static_model = load_model(os.path.join(MODELS_DIR, "static_model.keras"), compile=False)
dynamic_model = load_model(os.path.join(MODELS_DIR, "dynamic_model.keras"), compile=False)
static_labels = np.load(os.path.join(MODELS_DIR, "labels_static.npy"), allow_pickle=True)
dynamic_labels = np.load(os.path.join(MODELS_DIR, "labels_dynamic.npy"), allow_pickle=True)

print("✅ Models and Labels loaded successfully")

# --- Initialize Detector ---
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

class PredictRequest(BaseModel):
    frames: list[str]

def get_120_features(coords_seq):
    all_frame_feats = []
    for frame in coords_seq:
        wrist = frame[0]
        norm_frame = frame - wrist
        scale = np.max(np.linalg.norm(norm_frame, axis=1))
        if scale > 0: norm_frame /= scale
        bones = np.array([norm_frame[i] - norm_frame[0] for i in range(1, 21)]).flatten()
        all_frame_feats.append(bones)
    
    all_frame_feats = np.array(all_frame_feats)
    if all_frame_feats.shape[0] < 2:
        velocity = np.zeros_like(all_frame_feats)
    else:
        velocity = np.diff(all_frame_feats, axis=0)
        velocity = np.vstack([velocity, np.zeros((1, 60))])
    return np.concatenate([all_frame_feats, velocity], axis=1)

def process_base64_to_coords(base64_str: str):
    try:
        if "," in base64_str: base64_str = base64_str.split(",")[1]
        img_bytes = base64.b64decode(base64_str)
        np_img = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb)
        if results.multi_hand_landmarks:
            return np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
    except: return None
    return None

@app.post("/predict")
async def predict(req: PredictRequest):
    if not req.frames:
        return {"success": False, "prediction": "No data", "confidence": 0.0}

    # Static Logic (1 Frame)
    if len(req.frames) == 1:
        coords = process_base64_to_coords(req.frames[0])
        if coords is None: return {"success": False, "prediction": "No hand", "confidence": 0.0}
        x = get_120_features([coords])[0].reshape(1, 120)
        preds = static_model.predict(x, verbose=0)[0]
        return {"success": True, "prediction": str(static_labels[np.argmax(preds)]), "confidence": float(np.max(preds))}

    # Dynamic Logic (30 Frames)
    coord_sequence = []
    for f in req.frames:
        c = process_base64_to_coords(f)
        if c is not None: coord_sequence.append(c)
    
    if len(coord_sequence) < 5:
        return {"success": False, "prediction": "Scanning...", "confidence": 0.0}

    if len(coord_sequence) < SEQ_LEN:
        coord_sequence += [coord_sequence[-1]] * (SEQ_LEN - len(coord_sequence))
    else:
        coord_sequence = coord_sequence[-SEQ_LEN:]

    x = get_120_features(coord_sequence).reshape(1, SEQ_LEN, 120)
    preds = dynamic_model.predict(x, verbose=0)[0]
    return {"success": True, "prediction": str(dynamic_labels[np.argmax(preds)]), "confidence": float(np.max(preds))}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
