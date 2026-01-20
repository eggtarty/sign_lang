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
// 1. UTILS & MOTION 
// ===============================

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.style.display = 'block';
  setTimeout(() => errorMessage.style.display = 'none', 5000);
}

function computeMotionScore() {
  if (!isCameraOn) return 0;
  const canvas = document.createElement('canvas');
  canvas.width = MOTION_DOWNSCALE_W;
  canvas.height = Math.round(MOTION_DOWNSCALE_W * (videoElement.videoHeight / videoElement.videoWidth || 0.75));
  const ctx = canvas.getContext('2d');
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(videoElement, -canvas.width, 0, canvas.width, canvas.height);
  ctx.restore();

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
  prevMotionGray = null;
  motionTimer = setInterval(() => {
    lastMotionScore = computeMotionScore();
  }, MOTION_SAMPLE_MS);
}

// ===============================
// 2. API CALLS 
// ===============================

async function sendStaticPrediction(imageData) {
  try {
    const startTime = Date.now();
    const response = await fetch(`${BACKEND_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageData })
    });
    const result = await response.json();
    return { ...result, responseTime: Date.now() - startTime };
  } catch (error) {
    return { success: false, prediction: 'Error', confidence: 0, error: error.message };
  }
}

async function sendDynamicPrediction(imagesBase64) {
  try {
    const startTime = Date.now();
    const response = await fetch(`${BACKEND_URL}/predict/dynamic`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frames: imagesBase64 }) 
    });
    const result = await response.json();
    return { ...result, responseTime: Date.now() - startTime };
  } catch (error) {
    return { success: false, prediction: 'Error', confidence: 0, error: error.message };
  }
}

// ===============================
// 3. CORE LOGIC
// ===============================

async function startCamera() {
  try {
    const constraints = { video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false };
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    videoElement.srcObject = stream;
    isCameraOn = true;
    cameraStatus.textContent = 'On';
    cameraStatus.className = 'status-value online';
    startCameraBtn.disabled = true;
    captureBtn.disabled = false;

    startMotionDetection(); // Now defined above!
  } catch (error) {
    showError(`Camera Error: ${error.message}`);
  }
}

function captureFrameBase64() {
  if (!isCameraOn) return null;
  const canvas = document.createElement('canvas');
  canvas.width = 480; // Downscale for speed
  canvas.height = 360;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.7);
}

function applyResultToUI(result) {
  if (result.success) {
    gestureText.textContent = result.prediction;
    const conf = Math.round((result.confidence || 0) * 100);
    confidenceBar.style.width = `${conf}%`;
    confidenceValue.textContent = `${conf}%`;
    
    // TTS
    if (ttsEnabled && result.confidence >= TTS_CONFIDENCE_MIN && result.prediction !== lastSpoken) {
        lastSpoken = result.prediction;
        const utter = new SpeechSynthesisUtterance(result.prediction);
        window.speechSynthesis.speak(utter);
    }
  } else {
    gestureText.textContent = result.prediction || "Error";
  }
}

async function processDynamicSequence(frames) {
  const result = await sendDynamicPrediction(frames);
  applyResultToUI(result);
}

function startDynamicCapture() {
  if (isCapturingDynamic || !isCameraOn) return;
  isCapturingDynamic = true;
  dynamicFrames = [];
  gestureText.textContent = "Capturing...";

  dynamicCaptureTimer = setInterval(() => {
    const img = captureFrameBase64();
    if (img) dynamicFrames.push(img);

    if (dynamicFrames.length >= SEQ_LEN) {
      clearInterval(dynamicCaptureTimer);
      isCapturingDynamic = false;
      processDynamicSequence(dynamicFrames);
    }
  }, DYNAMIC_INTERVAL_MS);
}

async function processOnce() {
  if (!isCameraOn) return;
  const mode = modeSelect.value;
  if (mode === "static") {
      const img = captureFrameBase64();
      const res = await sendStaticPrediction(img);
      applyResultToUI(res);
  } else {
      startDynamicCapture();
  }
}

// Listeners
startCameraBtn.addEventListener('click', startCamera);
captureBtn.addEventListener('click', processOnce);
ttsToggleBtn.addEventListener('click', () => {
    ttsEnabled = !ttsEnabled;
    ttsToggleBtn.textContent = ttsEnabled ? "🔊 TTS: ON" : "🔇 TTS: OFF";
});

// Init
startCameraBtn.disabled = false;
console.log("System Ready");
