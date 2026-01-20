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
try:
    static_model = load_model(os.path.join(MODELS_DIR, "static_model.keras"), compile=False)
    dynamic_model = load_model(os.path.join(MODELS_DIR, "dynamic_model.keras"), compile=False)
    static_labels = np.load(os.path.join(MODELS_DIR, "labels_static.npy"), allow_pickle=True)
    dynamic_labels = np.load(os.path.join(MODELS_DIR, "labels_dynamic.npy"), allow_pickle=True)
    print("✅ Models and Labels loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# --- Initialize Detector ---
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

class PredictRequest(BaseModel):
    image: str 

@app.get("/health")
async def health():
    return {"status": "online"}

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

@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        header, encoded = req.image.split(",", 1)
        data = base64.b64decode(encoded)
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands_detector.process(rgb)
        if not results.multi_hand_landmarks:
            return {"success": False, "prediction": "No hand detected", "confidence": 0.0}

        coords = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
        
        # Single frame static prediction logic
        features = get_120_features([coords])[0].reshape(1, 120)
        preds = static_model.predict(features, verbose=0)[0]
        
        idx = np.argmax(preds)
        return {
            "success": True, 
            "prediction": str(static_labels[idx]), 
            "confidence": float(preds[idx])
        }
    except Exception as e:
        return {"success": False, "prediction": "Error", "error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
