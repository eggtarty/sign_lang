import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def dbg(path):
    print(
        "MODEL CHECK →",
        path,
        "| exists:",
        os.path.exists(path),
        "| size:",
        os.path.getsize(path) if os.path.exists(path) else "N/A"
    )

dbg(os.path.join(MODEL_DIR, "static_model.keras"))
dbg(os.path.join(MODEL_DIR, "dynamic_model.keras"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import numpy as np
import cv2
import base64
import os
from typing import Dict, Any, List

import mediapipe as mp

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer as TFInputLayer
from tensorflow.keras.mixed_precision import Policy as TFPolicy


# ----------------
# FastAPI setup
# ----------------
app = FastAPI(title="Sign Language Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SEQ_LEN = 30
MIN_DYNAMIC_FRAMES = 20

# -----------------------
# MediaPipe setup
# -----------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# -----------------------
# Keras load compatibility patches
# -----------------------
class PatchedInputLayer(TFInputLayer):
    @classmethod
    def from_config(cls, config):
        # Some saved models store "batch_shape" instead of "batch_input_shape"
        if "batch_shape" in config and "batch_input_shape" not in config:
            config["batch_input_shape"] = config.pop("batch_shape")
        return super().from_config(config)


class DTypePolicy(TFPolicy):
    """
    Compatibility shim for models saved with configs that reference keras.DTypePolicy.
    TF/Keras 2.12 uses mixed_precision.Policy.
    """
    @classmethod
    def from_config(cls, config):
        if isinstance(config, dict):
            name = config.get("name", "float32")
        else:
            name = str(config) if config is not None else "float32"
        return TFPolicy(name)


# -----------------------
# Load models & labels
# -----------------------
MODEL_DIR = "models"

static_model = None
dynamic_model = None
static_labels = np.array([])
dynamic_labels = np.array([])

try:
    custom_objects = {
        "InputLayer": PatchedInputLayer,
        "DTypePolicy": DTypePolicy,
    }

    static_model = load_model(
        os.path.join(MODEL_DIR, "static_model.keras"),
        custom_objects=custom_objects,
        compile=False,
    )
    dynamic_model = load_model(
        os.path.join(MODEL_DIR, "dynamic_model.keras"),
        custom_objects=custom_objects,
        compile=False,
    )

    static_labels = np.load(os.path.join(MODEL_DIR, "labels_static.npy"), allow_pickle=True)
    dynamic_labels = np.load(os.path.join(MODEL_DIR, "labels_dynamic.npy"), allow_pickle=True)

    print("✅ Models loaded successfully")
    print(f"✅ Static labels: {len(static_labels)}")
    print(f"✅ Dynamic labels: {len(dynamic_labels)}")

except Exception as e:
    print(f"❌ Error loading models: {e}")
    static_model = None
    dynamic_model = None
    static_labels = np.array([])
    dynamic_labels = np.array([])


# -----------------------
# Feature extraction helpers
# -----------------------
def get_120_features(coords_seq: List[np.ndarray]) -> np.ndarray:
    """
    Extract 120 features from coordinate sequence:
    - 60 bone vector features (wrist->landmarks 1..20)
    - 60 velocity features (diff of bone features)
    """
    all_frame_feats = []
    for frame in coords_seq:
        wrist = frame[0]
        norm_frame = frame - wrist
        scale = np.max(np.linalg.norm(norm_frame, axis=1))
        if scale > 0:
            norm_frame /= scale

        bones = np.array([norm_frame[i] - norm_frame[0] for i in range(1, 21)]).flatten()  # (60,)
        all_frame_feats.append(bones)

    all_frame_feats = np.array(all_frame_feats)  # (T, 60)

    if all_frame_feats.shape[0] < 2:
        velocity = np.zeros_like(all_frame_feats)
    else:
        velocity = np.diff(all_frame_feats, axis=0)
        velocity = np.vstack([velocity, np.zeros((1, 60))])

    return np.concatenate([all_frame_feats, velocity], axis=1)  # (T, 120)


def calculate_motion_intensity(features_120: np.ndarray, window_size: int = 5) -> float:
    if len(features_120) < window_size:
        return 0.0
    velocities = features_120[-window_size:, 60:]  # last 60 dims
    motion_magnitudes = np.linalg.norm(velocities, axis=1)
    return float(np.mean(motion_magnitudes))


def pad_sequence_to_length(sequence: List[np.ndarray], target_length: int) -> List[np.ndarray]:
    if len(sequence) >= target_length:
        return sequence[-target_length:]
    padding_needed = target_length - len(sequence)
    padding = [sequence[-1]] * padding_needed if sequence else [np.zeros((21, 3))] * padding_needed
    return sequence + padding


def extract_features_from_image(image: np.ndarray) -> np.ndarray | None:
    """Extract 120 features from hand landmarks for static prediction (single frame -> 120)."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    landmarks = results.multi_hand_landmarks[0]
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])  # (21, 3)

    feats = get_120_features([coords])  # (1, 120)
    return feats[0]  # (120,)


# -------------
# API routes
# -------------
@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Sign Language Recognition API",
        "endpoints": {
            "/health": "Check API health",
            "/predict": "POST with base64 image",
            "/predict/dynamic": "POST with base64 image sequence",
            "/gestures": "Get available gestures",
        },
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "static_loaded": static_model is not None,
        "dynamic_loaded": dynamic_model is not None,
        "static_labels_count": int(len(static_labels)),
        "dynamic_labels_count": int(len(dynamic_labels)),
    }


@app.get("/gestures")
def get_gestures():
    return {
        "static_gestures": static_labels.tolist(),
        "dynamic_gestures": dynamic_labels.tolist(),
    }


@app.post("/predict")
async def predict(data: Dict[str, Any]):
    """Predict sign language gesture from a base64 image (static)."""
    try:
        if static_model is None or static_labels is None or len(static_labels) == 0:
            raise HTTPException(status_code=500, detail="Static model/labels not loaded")

        image_base64 = data.get("image", "")
        if not image_base64:
            raise HTTPException(status_code=400, detail="No image provided")

        if "base64," in image_base64:
            image_base64 = image_base64.split("base64,")[1]

        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        features = extract_features_from_image(image)
        if features is None:
            return JSONResponse(
                {
                    "success": False,
                    "prediction": "No hand detected",
                    "confidence": 0.0,
                }
            )

        features = features.reshape(1, 120)

        prediction = static_model.predict(features, verbose=0)
        label_idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        label = str(static_labels[label_idx]) if label_idx < len(static_labels) else f"Label_{label_idx}"

        return {
            "success": True,
            "prediction": label,
            "confidence": confidence,
            "label_index": label_idx,
            "all_predictions": prediction[0].tolist(),
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/dynamic")
async def predict_dynamic(data: Dict[str, Any]):
    """Predict dynamic gesture from a sequence of base64 images."""
    try:
        if dynamic_model is None or dynamic_labels is None or len(dynamic_labels) == 0:
            raise HTTPException(status_code=500, detail="Dynamic model/labels not loaded")

        images_base64 = data.get("images", [])
        if not images_base64:
            raise HTTPException(status_code=400, detail="No images provided")

        all_coords: List[np.ndarray] = []
        for i, img_base64 in enumerate(images_base64[:SEQ_LEN]):
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
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])  # (21,3)
                    all_coords.append(coords)
            except Exception as fe:
                print(f"Frame {i} processing error: {fe}")
                continue

        if len(all_coords) < MIN_DYNAMIC_FRAMES:
            return JSONResponse(
                {
                    "success": False,
                    "prediction": f"Insufficient data: {len(all_coords)}/{MIN_DYNAMIC_FRAMES} frames",
                    "confidence": 0.0,
                    "frames_collected": len(all_coords),
                    "min_required": MIN_DYNAMIC_FRAMES,
                }
            )

        # Pad or crop to SEQ_LEN
        if len(all_coords) < SEQ_LEN:
            all_coords = pad_sequence_to_length(all_coords, SEQ_LEN)
        else:
            all_coords = all_coords[-SEQ_LEN:]

        features_120 = get_120_features(all_coords)  # (SEQ_LEN, 120)
        motion_intensity = calculate_motion_intensity(features_120, window_size=3)

        input_data = features_120.reshape(1, SEQ_LEN, 120)

        prediction = dynamic_model.predict(input_data, verbose=0)
        label_idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        label = (
            str(dynamic_labels[label_idx]) if label_idx < len(dynamic_labels) else f"Dynamic_Label_{label_idx}"
        )

        confidence_threshold = float(data.get("confidence_threshold", 0.6))

        response = {
            "success": True,
            "prediction": label,
            "confidence": confidence,
            "label_index": label_idx,
            "motion_intensity": motion_intensity,
            "frames_used": len(all_coords),
            "all_predictions": prediction[0].tolist(),
            "top_3_predictions": [
                {
                    "label": str(dynamic_labels[i]) if i < len(dynamic_labels) else f"Label_{i}",
                    "confidence": float(prediction[0][i]),
                }
                for i in np.argsort(prediction[0])[-3:][::-1]
            ],
        }

        if confidence < confidence_threshold:
            response["warning"] = f"Low confidence ({confidence:.2%} < {confidence_threshold:.0%})"

        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"Dynamic prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/dynamic/analyze")
async def analyze_dynamic_sequence(data: Dict[str, Any]):
    """Analyze motion in a sequence without prediction."""
    try:
        images_base64 = data.get("images", [])
        if not images_base64:
            raise HTTPException(status_code=400, detail="No images provided")

        all_coords: List[np.ndarray] = []
        for i, img_base64 in enumerate(images_base64[:SEQ_LEN]):
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
            except Exception as fe:
                print(f"Frame analysis error {i}: {fe}")
                continue

        if len(all_coords) < 2:
            return {
                "success": False,
                "message": "Need at least 2 frames with hands",
                "frames": len(all_coords),
            }

        features = get_120_features(all_coords)
        motion_intensity = calculate_motion_intensity(features)

        motions = []
        for i in range(1, len(all_coords)):
            diff = np.mean(np.linalg.norm(all_coords[i] - all_coords[i - 1], axis=1))
            motions.append(float(diff))

        return {
            "success": True,
            "frames_analyzed": len(all_coords),
            "motion_intensity": motion_intensity,
            "avg_frame_motion": float(np.mean(motions)) if motions else 0.0,
            "max_frame_motion": float(np.max(motions)) if motions else 0.0,
            "is_dynamic_ready": len(all_coords) >= MIN_DYNAMIC_FRAMES,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/test/features")
async def test_features():
    """Test feature extraction with dummy data."""
    try:
        dummy_frames = []
        for _ in range(5):
            coords = np.random.randn(21, 3) * 0.1 + np.array([0.5, 0.5, 0.0])
            dummy_frames.append(coords)

        features = get_120_features(dummy_frames)

        return {
            "success": True,
            "feature_shape": list(features.shape),
            "sample_feature": features[0].tolist() if len(features) > 0 else [],
            "velocity_sample": features[0, 60:65].tolist() if len(features) > 0 else [],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----
# Run
# -----
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

