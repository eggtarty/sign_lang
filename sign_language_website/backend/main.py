# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import base64
import mediapipe as mp
from typing import Dict, Any
import os

# Initialize FastAPI
app = FastAPI(title="Sign Language Recognition API")

# Allow all origins for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe once
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# Load models
MODEL_DIR = "models"
try:
    static_model = load_model(os.path.join(MODEL_DIR, "static_model.keras"))
    dynamic_model = load_model(os.path.join(MODEL_DIR, "dynamic_model.keras"))
    static_labels = np.load(os.path.join(MODEL_DIR, "labels_static.npy"), allow_pickle=True)
    dynamic_labels = np.load(os.path.join(MODEL_DIR, "labels_dynamic.npy"), allow_pickle=True)
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    static_model = dynamic_model = None
    static_labels = dynamic_labels = []

def extract_features_from_image(image: np.ndarray) -> np.ndarray:
    """Extract 120 features from hand landmarks"""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    # Get landmarks
    landmarks = results.multi_hand_landmarks[0]
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    
    # Normalize
    wrist = coords[0]
    norm_coords = coords - wrist
    scale = np.max(np.linalg.norm(norm_coords, axis=1))
    if scale > 0:
        norm_coords /= scale
    
    # Extract features (60 bones + 60 velocity zeros for single frame)
    bones = np.array([norm_coords[i] - norm_coords[0] for i in range(1, 21)]).flatten()
    
    # Add velocity (zeros for static)
    features = np.concatenate([bones, np.zeros(60)])
    
    return features

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Sign Language Recognition API",
        "endpoints": {
            "/health": "Check API health",
            "/predict": "POST with base64 image"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": static_model is not None,
        "static_labels_count": len(static_labels) if static_labels is not None else 0,
        "dynamic_labels_count": len(dynamic_labels) if dynamic_labels is not None else 0
    }

@app.post("/predict")
async def predict(data: Dict[str, Any]):
    """Predict sign language gesture from base64 image"""
    try:
        # Get base64 image
        image_base64 = data.get("image", "")
        if not image_base64:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Remove data URL prefix if present
        if "base64," in image_base64:
            image_base64 = image_base64.split("base64,")[1]
        
        # Decode base64 to image
        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Extract features
        features = extract_features_from_image(image)
        if features is None:
            return JSONResponse({
                "success": False,
                "prediction": "No hand detected",
                "confidence": 0.0
            })
        
        # Reshape for model (1, 120)
        features = features.reshape(1, 120)
        
        # Predict
        prediction = static_model.predict(features, verbose=0)
        label_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        label = str(static_labels[label_idx]) if static_labels is not None else "Unknown"
        
        return {
            "success": True,
            "prediction": label,
            "confidence": confidence,
            "label_index": int(label_idx),
            "all_predictions": prediction[0].tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gestures")
def get_gestures():
    """Get list of available gestures"""
    return {
        "static_gestures": static_labels.tolist() if static_labels is not None else [],
        "dynamic_gestures": dynamic_labels.tolist() if dynamic_labels is not None else []
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)