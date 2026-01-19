// Configuration
const BACKEND_URL = "https://signlanguage-detector-pi6d.onrender.com";

// ===== Settings =====
const SEQ_LEN = 30;                 
const DYNAMIC_INTERVAL_MS = 150;    
const TTS_CONFIDENCE_MIN = 0.70;    

// Auto motion detection settings
const MOTION_SAMPLE_MS = 250;       
const MOTION_THRESHOLD = 12;        
const MOTION_DOWNSCALE_W = 64;      

// DOM Elements
const videoElement = document.getElementById('videoElement');
const startCameraBtn = document.getElementById('startCamera');
const captureBtn = document.getElementById('captureBtn');
const autoModeBtn = document.getElementById('autoMode');
const ttsToggleBtn = document.getElementById('ttsToggle');
const modeSelect = document.getElementById('modeSelect');

const cameraStatus = document.getElementById('cameraStatus');
const handStatus = document.getElementById('handStatus');
const apiStatus = document.getElementById('apiStatus');
const gestureText = document.getElementById('gestureText');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceValue = document.getElementById('confidenceValue');
const historyList = document.getElementById('historyList');
const errorMessage = document.getElementById('errorMessage');
const backendUrlElement = document.getElementById('backendUrl');

// State
let stream = null;
let isCameraOn = false;
let isAutoMode = false;
let autoInterval = null;
let predictionHistory = [];

// Dynamic buffer state
let dynamicFrames = [];
let dynamicCaptureTimer = null;
let isCapturingDynamic = false;

// Motion detection state
let motionTimer = null;
let prevMotionGray = null;
let lastMotionScore = 0;

// TTS state
let lastSpoken = "";
let ttsEnabled = true;

// ===============================
// 1. UTILS & UI
// ===============================

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.style.display = 'block';
  setTimeout(() => errorMessage.style.display = 'none', 5000);
}

function applyResultToUI(result) {
  if (result.success) {
    gestureText.textContent = result.prediction;
    const conf = Math.round((result.confidence || 0) * 100);
    confidenceBar.style.width = `${conf}%`;
    confidenceValue.textContent = `${conf}%`;
    
    handStatus.textContent = 'Yes';
    handStatus.className = 'status-value online';

    addToHistory(result.prediction, conf);

    // TTS Logic
    if (ttsEnabled && result.confidence >= TTS_CONFIDENCE_MIN && result.prediction !== lastSpoken) {
        lastSpoken = result.prediction;
        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(result.prediction);
        window.speechSynthesis.speak(utter);
    }
  } else {
    handStatus.textContent = 'No';
    handStatus.className = 'status-value offline';
    gestureText.textContent = result.prediction || "Error";
  }
}

function addToHistory(gesture, conf) {
    const item = { gesture, conf, time: new Date().toLocaleTimeString() };
    predictionHistory.unshift(item);
    if (predictionHistory.length > 5) predictionHistory.pop();
    
    historyList.innerHTML = predictionHistory.map(h => `
        <div class="history-item">
            <span>${h.gesture}</span>
            <span>${h.conf}% (${h.time})</span>
        </div>
    `).join('');
}

// ===============================
// 2. MOTION DETECTION
// ===============================

function computeMotionScore() {
  if (!isCameraOn) return 0;
  const canvas = document.createElement('canvas');
  canvas.width = MOTION_DOWNSCALE_W;
  canvas.height = Math.round(MOTION_DOWNSCALE_W * (videoElement.videoHeight / videoElement.videoWidth));
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

  const img = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const gray = new Uint8Array(canvas.width * canvas.height);
  for (let i = 0, j = 0; i < img.length; i += 4, j++) {
    gray[j] = (img[i] * 0.299 + img[i + 1] * 0.587 + img[i + 2] * 0.114) | 0;
  }

  if (!prevMotionGray) {
    prevMotionGray = gray;
    return 0;
  }

  let diffSum = 0;
  for (let k = 0; k < gray.length; k++) {
    diffSum += Math.abs(gray[k] - prevMotionGray[k]);
  }
  prevMotionGray = gray;
  return diffSum / gray.length;
}

function startMotionDetection() {
  if (motionTimer) clearInterval(motionTimer);
  motionTimer = setInterval(() => {
    lastMotionScore = computeMotionScore();
  }, MOTION_SAMPLE_MS);
}

// ===============================
// 3. API CALLS
// ===============================

async function sendStaticPrediction(imageData) {
  try {
    const response = await fetch(`${BACKEND_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageData })
    });
    return await response.json();
  } catch (error) {
    return { success: false, prediction: 'Error', confidence: 0 };
  }
}

async function sendDynamicPrediction(imagesBase64) {
  try {
    const response = await fetch(`${BACKEND_URL}/predict/dynamic`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frames: imagesBase64 }) 
    });
    return await response.json();
  } catch (error) {
    return { success: false, prediction: 'Error', confidence: 0 };
  }
}

// ===============================
// 4. CORE ENGINE
// ===============================

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    videoElement.srcObject = stream;
    isCameraOn = true;
    cameraStatus.textContent = 'On';
    cameraStatus.className = 'status-value online';
    startCameraBtn.disabled = true;
    captureBtn.disabled = false;
    startMotionDetection();
    checkBackend();
  } catch (error) {
    showError(`Camera Error: ${error.message}`);
  }
}

async function checkBackend() {
    backendUrlElement.textContent = BACKEND_URL;
    try {
        const res = await fetch(`${BACKEND_URL}/health`);
        if (res.ok) {
            apiStatus.textContent = 'Online';
            apiStatus.className = 'status-value online';
        }
    } catch(e) {
        apiStatus.textContent = 'Offline';
        apiStatus.className = 'status-value offline';
    }
}

function captureFrameBase64() {
  if (!isCameraOn) return null;
  const canvas = document.createElement('canvas');
  canvas.width = 480; 
  canvas.height = 360;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.6);
}

async function processStaticOnce() {
    const img = captureFrameBase64();
    if (img) {
        const result = await sendStaticPrediction(img);
        applyResultToUI(result);
    }
}

function startDynamicCapture() {
  if (isCapturingDynamic || !isCameraOn) return;
  isCapturingDynamic = true;
  dynamicFrames = [];
  gestureText.textContent = "Capturing...";

  dynamicCaptureTimer = setInterval(async () => {
    const img = captureFrameBase64();
    if (img) dynamicFrames.push(img);

    if (dynamicFrames.length >= SEQ_LEN) {
      clearInterval(dynamicCaptureTimer);
      const result = await sendDynamicPrediction(dynamicFrames);
      isCapturingDynamic = false;
      applyResultToUI(result);
    }
  }, DYNAMIC_INTERVAL_MS);
}

function toggleAutoMode() {
    isAutoMode = !isAutoMode;
    if (isAutoMode) {
        autoModeBtn.textContent = '🔄 Auto Mode: ON';
        autoModeBtn.className = 'btn btn-success';
        autoInterval = setInterval(() => {
            if (isCapturingDynamic) return;
            const mode = modeSelect.value;
            if (mode === "auto") {
                if (lastMotionScore >= MOTION_THRESHOLD) startDynamicCapture();
                else processStaticOnce();
            } else if (mode === "static") {
                processStaticOnce();
            } else {
                startDynamicCapture();
            }
        }, 3000);
    } else {
        autoModeBtn.textContent = '🔄 Auto Mode: OFF';
        autoModeBtn.className = 'btn btn-secondary';
        clearInterval(autoInterval);
    }
}

// Listeners
startCameraBtn.addEventListener('click', startCamera);
captureBtn.addEventListener('click', () => {
    if (modeSelect.value === "static") processStaticOnce();
    else startDynamicCapture();
});
autoModeBtn.addEventListener('click', toggleAutoMode);
ttsToggleBtn.addEventListener('click', () => {
    ttsEnabled = !ttsEnabled;
    ttsToggleBtn.textContent = ttsEnabled ? "🔊 TTS: ON" : "🔇 TTS: OFF";
});
