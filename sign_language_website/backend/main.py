import os
import base64
import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.models import load_model
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
# CONFIG & PATHS
# ===============================
SEQ_LEN = 30
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

STATIC_MODEL_PATH = os.path.join(MODELS_DIR, "static_model.keras")
DYNAMIC_MODEL_PATH = os.path.join(MODELS_DIR, "dynamic_model.keras")
STATIC_LABELS_PATH = os.path.join(MODELS_DIR, "labels_static.npy")
DYNAMIC_LABELS_PATH = os.path.join(MODELS_DIR, "labels_dynamic.npy")

# =============
# LOAD ASSETS 
# =============
static_model = None
dynamic_model = None
static_labels = np.array([])
dynamic_labels = np.array([])

try:
    # Load with compile=False to handle potential Keras version mismatches
    static_model = load_model(STATIC_MODEL_PATH, compile=False)
    dynamic_model = load_model(DYNAMIC_MODEL_PATH, compile=False)

    static_labels = np.load(STATIC_LABELS_PATH, allow_pickle=True)
    dynamic_labels = np.load(DYNAMIC_LABELS_PATH, allow_pickle=True)

    print("✅ Models and Labels loaded successfully (Feature Sync Mode)")
except Exception as e:
    print(f"❌ Failed to load assets: {e}")

# ===============================
# MEDIAPIPE INIT
# ===============================
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# ===============================
# REQUEST SCHEMAS
# ===============================
class PredictRequest(BaseModel):
    image: str # Base64 string

class PredictDynamicRequest(BaseModel):
    frames: list[str] # List of Base64 strings

# ===============================
# CORE FEATURE EXTRACTION (The Fix)
# ===============================
def get_120_features(coords_seq):
    """
    EXACT SYNC WITH LOCAL APP.PY:
    Normalizes hand coordinates relative to the wrist and calculates velocity.
    """
    all_frame_feats = []
    for frame in coords_seq:
        # frame is (21, 3)
        wrist = frame[0]
        norm_frame = frame - wrist
        
        # Scaling normalization
        scale = np.max(np.linalg.norm(norm_frame, axis=1))
        if scale > 0:
            norm_frame /= scale
            
        # Extract bone relative positions (20 points * 3 dims = 60 features)
        bones = np.array([norm_frame[i] - norm_frame[0] for i in range(1, 21)]).flatten()
        all_frame_feats.append(bones)

    all_frame_feats = np.array(all_frame_feats) # (T, 60)

    # Calculate Velocity (The remaining 60 features)
    if all_frame_feats.shape[0] < 2:
        velocity = np.zeros_like(all_frame_feats)
    else:
        velocity = np.diff(all_frame_feats, axis=0)
        velocity = np.vstack([velocity, np.zeros((1, 60))])

    # Combine: 60 bones + 60 velocity = 120 features
    return np.concatenate([all_frame_feats, velocity], axis=1)

def process_base64_to_coords(base64_str: str):
    """Converts base64 image to MediaPipe landmark coordinates (21, 3)."""
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_bytes = base64.b64decode(base64_str)
        np_img = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if frame is None: return None
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb)
        
        if results.multi_hand_landmarks:
            h_lms = results.multi_hand_landmarks[0]
            return np.array([[lm.x, lm.y, lm.z] for lm in h_lms.landmark])
        return None
    except:
        return None

# ===============================
# ROUTES
# ===============================
@app.get("/health")
def health():
    return {"status": "online", "backend": "Render Fixed"}

@app.post("/predict")
def predict_static(req: PredictRequest):
    coords = process_base64_to_coords(req.image)
    if coords is None:
        return {"success": False, "prediction": "No hand detected", "confidence": 0.0}

    # Static prediction needs 120 features for the single frame
    feats = get_120_features([coords])
    x = feats[0].reshape(1, 120)
    
    preds = static_model.predict(x, verbose=0)[0]
    idx = np.argmax(preds)
    conf = float(preds[idx])
    
    return {
        "success": True, 
        "prediction": str(static_labels[idx]), 
        "confidence": conf
    }

@app.post("/predict/dynamic")
def predict_dynamic(req: PredictDynamicRequest):
    coord_sequence = []
    for f in req.frames:
        c = process_base64_to_coords(f)
        if c is not None:
            coord_sequence.append(c)
    
    if len(coord_sequence) < 5: # Minimum frames to consider a gesture
        return {"success": False, "prediction": "Too few frames", "confidence": 0.0}

    # Pad sequence to match SEQ_LEN (30)
    if len(coord_sequence) < SEQ_LEN:
        padding = [coord_sequence[-1]] * (SEQ_LEN - len(coord_sequence))
        coord_sequence.extend(padding)
    else:
        coord_sequence = coord_sequence[-SEQ_LEN:]

    # Transform to 120 features
    feats_120 = get_120_features(coord_sequence)
    x = feats_120.reshape(1, SEQ_LEN, 120)
    
    preds = dynamic_model.predict(x, verbose=0)[0]
    idx = np.argmax(preds)
    conf = float(preds[idx])
    
    return {
        "success": True, 
        "prediction": str(dynamic_labels[idx]), 
        "confidence": conf
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
