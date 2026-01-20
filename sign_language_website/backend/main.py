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

# Import mediapipe specifically for server stability
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands

app = FastAPI()

# Enable CORS so your frontend can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration for models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

try:
    static_model = load_model(os.path.join(MODELS_DIR, "static_model.keras"), compile=False)
    static_labels = np.load(os.path.join(MODELS_DIR, "labels_static.npy"), allow_pickle=True)
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error: {e}")

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

class PredictRequest(BaseModel):
    image: str

@app.get("/health")
async def health():
    return {"status": "online"}

@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        # Decode base64 image from frontend
        encoded = req.image.split(",", 1)[1]
        data = base64.b64decode(encoded)
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands_detector.process(rgb)
        if not results.multi_hand_landmarks:
            return {"success": False, "prediction": "No hand detected", "confidence": 0.0}

        # Simplified prediction logic
        coords = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]).flatten()
        preds = static_model.predict(coords.reshape(1, -1), verbose=0)[0]
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
