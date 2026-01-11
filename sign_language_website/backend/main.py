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
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Load models
MODEL_DIR = "models"
try:
    static_model = load_model(os.path.join(MODEL_DIR, "static_model.keras"))
    dynamic_model = load_model(os.path.join(MODEL_DIR, "dynamic_model.keras"))
    static_labels = np.load(os.path.join(MODEL_DIR, "labels_static.npy"), allow_pickle=True)
    dynamic_labels = np.load(os.path.join(MODEL_DIR, "labels_dynamic.npy"), allow_pickle=True)
    print("✅ Models loaded successfully")
    print(f"Static labels: {static_labels}")
    print(f"Dynamic labels: {dynamic_labels}")
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
    
    # For static prediction, we need 120 features (60 bones + 60 velocity)
    features = np.concatenate([bones, np.zeros(60)])
    
    return features

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Sign Language Recognition API",
        "endpoints": {
            "/health": "Check API health",
            "/predict": "POST with base64 image",
            "/gestures": "Get available gestures"
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
        
        # Check if model is loaded
        if static_model is None:
            raise HTTPException(status_code=500, detail="Static model not loaded")
        
        # Predict
        prediction = static_model.predict(features, verbose=0)
        label_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        # Get label from static_labels
        if static_labels is not None and len(static_labels) > label_idx:
            label = str(static_labels[label_idx])
        else:
            label = f"Label_{label_idx}"
        
        return {
            "success": True,
            "prediction": label,
            "confidence": confidence,
            "label_index": int(label_idx),
            "all_predictions": prediction[0].tolist()
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gestures")
def get_gestures():
    """Get list of available gestures"""
    return {
        "static_gestures": static_labels.tolist() if static_labels is not None else [],
        "dynamic_gestures": dynamic_labels.tolist() if dynamic_labels is not None else []
    }

# Add dynamic gesture prediction endpoint
@app.post("/predict/dynamic")
async def predict_dynamic(data: Dict[str, Any]):
    """Predict dynamic gesture from sequence of images"""
    try:
        images_base64 = data.get("images", [])
        if not images_base64 or len(images_base64) < 20:  # Need at least 20 frames
            raise HTTPException(status_code=400, detail="Need at least 20 frames for dynamic gesture")
        
        # Process each image
        all_coords = []
        for img_base64 in images_base64[:30]:  # Limit to 30 frames like SEQ_LEN
            # Remove data URL prefix if present
            if "base64," in img_base64:
                img_base64 = img_base64.split("base64,")[1]
            
            # Decode image
            image_bytes = base64.b64decode(img_base64)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                continue
            
            # Extract coordinates
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
                all_coords.append(coords)
        
        if len(all_coords) < 20:
            return JSONResponse({
                "success": False,
                "prediction": "Not enough hand data collected",
                "confidence": 0.0
            })
        
        return {
            "success": True,
            "message": "Dynamic prediction endpoint - implement feature extraction",
            "frames_collected": len(all_coords)
        }
        
    except Exception as e:
        print(f"Dynamic prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
