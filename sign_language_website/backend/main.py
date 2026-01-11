from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import base64
import mediapipe as mp
from typing import Dict, Any, List
import os
from collections import deque

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

SEQ_LEN = 30
MIN_DYNAMIC_FRAMES = 20

# Initialize MediaPipe
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
    print(f"Static labels ({len(static_labels)}): {static_labels}")
    print(f"Dynamic labels ({len(dynamic_labels)}): {dynamic_labels}")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    static_model = dynamic_model = None
    static_labels = dynamic_labels = []

def get_120_features(coords_seq: List[np.ndarray]) -> np.ndarray:
    """
    Extract 120 features from coordinate sequence (60 bones + 60 velocities)
    Same as in Streamlit app
    """
    all_frame_feats = []
    for frame in coords_seq:
        wrist = frame[0]
        norm_frame = frame - wrist
        scale = np.max(np.linalg.norm(norm_frame, axis=1))
        if scale > 0:
            norm_frame /= scale
        
        # Extract bone vectors (frame[i] - frame[0] for i=1..20)
        bones = np.array([norm_frame[i] - norm_frame[0] for i in range(1, 21)]).flatten()
        all_frame_feats.append(bones)
    
    all_frame_feats = np.array(all_frame_feats)
    
    # Calculate velocity
    if all_frame_feats.shape[0] < 2:
        velocity = np.zeros_like(all_frame_feats)
    else:
        velocity = np.diff(all_frame_feats, axis=0)
        velocity = np.vstack([velocity, np.zeros((1, 60))])
    
    # Concatenate bones + velocity
    return np.concatenate([all_frame_feats, velocity], axis=1)

def calculate_motion_intensity(features_120: np.ndarray, window_size: int = 5) -> float:
    """Calculate motion intensity from velocity features"""
    if len(features_120) < window_size:
        return 0.0
    velocities = features_120[-window_size:, 60:]  # Velocity features are last 60 dimensions
    motion_magnitudes = np.linalg.norm(velocities, axis=1)
    return float(np.mean(motion_magnitudes))

def pad_sequence_to_length(sequence: List[np.ndarray], target_length: int) -> List[np.ndarray]:
    """Pad sequence to target length"""
    if len(sequence) >= target_length:
        return sequence[-target_length:]
    padding_needed = target_length - len(sequence)
    padding = [sequence[-1]] * padding_needed if sequence else [np.zeros((21, 3))] * padding_needed
    return sequence + padding

def extract_features_from_image(image: np.ndarray) -> np.ndarray:
    """Extract 120 features from hand landmarks for static prediction"""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    # Get landmarks
    landmarks = results.multi_hand_landmarks[0]
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    
    # Create a single-frame sequence for static prediction
    single_frame_seq = [coords]
    features = get_120_features(single_frame_seq)
    
    # Return features for the single frame
    return features[0] 

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

@app.post("/predict/dynamic")
async def predict_dynamic(data: Dict[str, Any]):
    """Predict dynamic gesture from sequence of base64 images"""
    try:
        # Get images array
        images_base64 = data.get("images", [])
        
        if not images_base64:
            raise HTTPException(status_code=400, detail="No images provided")
        
        print(f"Received {len(images_base64)} frames for dynamic prediction")
        
        # Process each image to get coordinates
        all_coords = []
        successful_frames = 0
        
        for i, img_base64 in enumerate(images_base64[:SEQ_LEN]):  # Limit to SEQ_LEN
            try:
                # Remove data URL prefix if present
                if "base64," in img_base64:
                    img_base64 = img_base64.split("base64,")[1]
                
                # Decode base64 to image
                image_bytes = base64.b64decode(img_base64)
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    print(f"Frame {i}: Failed to decode image")
                    continue
                
                # Process with MediaPipe
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                
                if results.multi_hand_landmarks:
                    landmarks = results.multi_hand_landmarks[0]
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
                    all_coords.append(coords)
                    successful_frames += 1
                else:
                    print(f"Frame {i}: No hand detected")
                    
            except Exception as e:
                print(f"Frame {i} processing error: {e}")
                continue
        
        print(f"Successfully extracted coordinates from {successful_frames}/{len(images_base64)} frames")
        
        # Check if we have enough frames
        if len(all_coords) < MIN_DYNAMIC_FRAMES:
            return JSONResponse({
                "success": False,
                "prediction": f"Insufficient data: {len(all_coords)}/{MIN_DYNAMIC_FRAMES} frames",
                "confidence": 0.0,
                "frames_collected": len(all_coords),
                "min_required": MIN_DYNAMIC_FRAMES
            })
        
        # Pad sequence to SEQ_LEN if needed
        if len(all_coords) < SEQ_LEN:
            print(f"Padding sequence from {len(all_coords)} to {SEQ_LEN} frames")
            all_coords = pad_sequence_to_length(all_coords, SEQ_LEN)
        else:
            # Take the last SEQ_LEN frames
            all_coords = all_coords[-SEQ_LEN:]
        
        # Extract features
        features_120 = get_120_features(all_coords)
        
        # Calculate motion intensity
        motion_intensity = calculate_motion_intensity(features_120, window_size=3)
        print(f"Motion intensity: {motion_intensity:.4f}")
        
        # Reshape for dynamic model: (1, SEQ_LEN, 120)
        input_data = features_120.reshape(1, SEQ_LEN, 120)
        
        # Check if model is loaded
        if dynamic_model is None:
            raise HTTPException(status_code=500, detail="Dynamic model not loaded")
        
        # Make prediction
        prediction = dynamic_model.predict(input_data, verbose=0)
        label_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        # Get label from dynamic_labels
        if dynamic_labels is not None and len(dynamic_labels) > label_idx:
            label = str(dynamic_labels[label_idx])
        else:
            label = f"Dynamic_Label_{label_idx}"
        
        # Optional: Apply confidence threshold
        confidence_threshold = data.get("confidence_threshold", 0.6)
        if confidence < confidence_threshold:
            return JSONResponse({
                "success": True,
                "prediction": label,
                "confidence": confidence,
                "label_index": int(label_idx),
                "warning": f"Low confidence ({confidence:.2%} < {confidence_threshold:.0%})",
                "motion_intensity": motion_intensity,
                "frames_used": len(all_coords)
            })
        
        return {
            "success": True,
            "prediction": label,
            "confidence": confidence,
            "label_index": int(label_idx),
            "motion_intensity": motion_intensity,
            "frames_used": len(all_coords),
            "all_predictions": prediction[0].tolist(),
            "top_3_predictions": [
                {
                    "label": str(dynamic_labels[i]) if i < len(dynamic_labels) else f"Label_{i}",
                    "confidence": float(prediction[0][i])
                }
                for i in np.argsort(prediction[0])[-3:][::-1]
            ]
        }
        
    except Exception as e:
        print(f"Dynamic prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Also add a simplified version for testing
@app.post("/predict/dynamic/analyze")
async def analyze_dynamic_sequence(data: Dict[str, Any]):
    """Analyze motion in a sequence without prediction"""
    try:
        images_base64 = data.get("images", [])
        if not images_base64:
            raise HTTPException(status_code=400, detail="No images provided")
        
        all_coords = []
        for img_base64 in images_base64[:30]:
            try:
                if "base64," in img_base64:
                    img_base64 = img_base64.split("base64,")[1]
                
                image_bytes = base64.b64decode(img_base64)
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    continue
                
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                
                if results.multi_hand_landmarks:
                    landmarks = results.multi_hand_landmarks[0]
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
                    all_coords.append(coords)
                    
            except Exception as e:
                print(f"Frame analysis error: {e}")
                continue
        
        if len(all_coords) < 2:
            return {
                "success": False,
                "message": "Need at least 2 frames with hands",
                "frames": len(all_coords)
            }
        
        # Calculate motion statistics
        features = get_120_features(all_coords)
        motion_intensity = calculate_motion_intensity(features)
        
        # Calculate frame-to-frame motion
        motions = []
        for i in range(1, len(all_coords)):
            diff = np.mean(np.linalg.norm(all_coords[i] - all_coords[i-1], axis=1))
            motions.append(float(diff))
        
        return {
            "success": True,
            "frames_analyzed": len(all_coords),
            "motion_intensity": motion_intensity,
            "avg_frame_motion": float(np.mean(motions)) if motions else 0.0,
            "max_frame_motion": float(np.max(motions)) if motions else 0.0,
            "is_dynamic_ready": len(all_coords) >= MIN_DYNAMIC_FRAMES
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add a test endpoint to verify feature extraction
@app.post("/test/features")
async def test_features(data: Dict[str, Any]):
    """Test feature extraction with sample data"""
    try:
        # Test with 5 dummy coordinate frames
        dummy_frames = []
        for i in range(5):
            # Create dummy coordinates similar to real hand data
            coords = np.random.randn(21, 3) * 0.1 + np.array([0.5, 0.5, 0])
            dummy_frames.append(coords)
        
        features = get_120_features(dummy_frames)
        
        return {
            "success": True,
            "feature_shape": features.shape,
            "sample_feature": features[0].tolist() if len(features) > 0 else [],
            "velocity_sample": features[0, 60:65].tolist() if len(features) > 0 else []
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
