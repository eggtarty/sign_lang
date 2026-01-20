import os
import base64
import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
static_model = load_model('model_static.h5')
dynamic_model = load_model('model_dynamic.h5')
static_labels = np.load('labels_static.npy')
dynamic_labels = np.load('labels_dynamic.npy')

# Mediapipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

class GestureRequest(BaseModel):
    frames: list[str]

def process_frame(b64_string):
    try:
        encoded_data = b64_string.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            landmarks = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            return landmarks
    except:
        pass
    return None

@app.post("/predict")
async def predict(data: GestureRequest):
    # If only 1 frame is sent, treat as Static
    if len(data.frames) == 1:
        lms = process_frame(data.frames[0])
        if not lms: return {"success": False, "prediction": "No Hand"}
        pred = static_model.predict(np.array([lms]), verbose=0)
        return {"success": True, "prediction": str(static_labels[np.argmax(pred)]), "confidence": float(np.max(pred))}

    # Otherwise, treat as Dynamic (Sequence)
    sequence = []
    for f in data.frames[-30:]: # Take last 30 frames
        lms = process_frame(f)
        if lms: sequence.append(lms)
    
    if len(sequence) < 20:
        return {"success": False, "prediction": "Incomplete Motion"}

    # Pad if sequence < 30
    while len(sequence) < 30:
        sequence.append(sequence[-1])

    pred = dynamic_model.predict(np.array([sequence]), verbose=0)
    return {"success": True, "prediction": str(dynamic_labels[np.argmax(pred)]), "confidence": float(np.max(pred))}
