// Configuration
const BACKEND_URL = "https://signlanguage-detector-pi6d.onrender.com";

// ===== Settings =====
const SEQ_LEN = 30;                 
const DYNAMIC_INTERVAL_MS = 150;    // capture frames every 150ms (~4.5s for 30 frames)
const TTS_CONFIDENCE_MIN = 0.70;    // speak only if >= 70%

// Auto motion detection settings
const MOTION_SAMPLE_MS = 250;       // check motion every 250ms
const MOTION_THRESHOLD = 12;        // tune: higher = needs more movement
const MOTION_DOWNSCALE_W = 64;      // small = faster

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

function speakIfConfident(text, confidence) {
  if (!ttsEnabled) return;
  if (!text) return;

  const conf = Number(confidence || 0);
  if (conf < TTS_CONFIDENCE_MIN) return;

  if (text === "Show Your Hand" || text === "No hand detected" || text === "Error") return;
  if (text === lastSpoken) return;
  lastSpoken = text;

  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(text);
  utter.rate = 1.0;
  utter.pitch = 1.0;
  window.speechSynthesis.speak(utter);
}

// Hide error message
function hideError() { errorMessage.style.display = 'none'; }

// Show error message
function showError(message) {
  errorMessage.textContent = message;
  errorMessage.style.display = 'block';
  setTimeout(hideError, 5000);
}

// Check backend status
async function checkBackendStatus() {
  try {
    const response = await fetch(`${BACKEND_URL}/health`);
    if (response.ok) {
      await response.json();
      apiStatus.textContent = 'Online';
      apiStatus.className = 'status-value online';
      backendUrlElement.textContent = BACKEND_URL;
      return true;
    }
  } catch (error) {
    console.error('Backend offline:', error);
  }

  apiStatus.textContent = 'Offline';
  apiStatus.className = 'status-value offline';
  backendUrlElement.textContent = 'Not connected';
  showError('Backend server is offline. Please try again later.');
  return false;
}

// Start camera
async function startCamera() {
  try {
    const constraints = {
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false
    };

    stream = await navigator.mediaDevices.getUserMedia(constraints);
    videoElement.srcObject = stream;

    isCameraOn = true;
    cameraStatus.textContent = 'On';
    cameraStatus.className = 'status-value online';
    startCameraBtn.disabled = true;
    captureBtn.disabled = false;

    startMotionDetection();

    console.log('Camera started successfully');
  } catch (error) {
    console.error('Error accessing camera:', error);
    showError(`Cannot access camera: ${error.message}. Please check permissions.`);
  }
}

// Capture frame from video (base64 jpg)
function captureFrameBase64() {
  if (!isCameraOn) return null;

  const canvas = document.createElement('canvas');
  canvas.width = videoElement.videoWidth;
  canvas.height = videoElement.videoHeight;
  const ctx = canvas.getContext('2d');

  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(videoElement, -canvas.width, 0, canvas.width, canvas.height);
  ctx.restore();

  return canvas.toDataURL('image/jpeg', 0.8);
}

// ===== Motion detection (simple pixel diff) =====
function computeMotionScore() {
  if (!isCameraOn) return 0;

  const canvas = document.createElement('canvas');
  canvas.width = MOTION_DOWNSCALE_W;
  canvas.height = Math.round(MOTION_DOWNSCALE_W * (videoElement.videoHeight / videoElement.videoWidth || 0.75));
  const ctx = canvas.getContext('2d');

  // draw mirrored small frame
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(videoElement, -canvas.width, 0, canvas.width, canvas.height);
  ctx.restore();

  const img = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

  // grayscale array
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

  // normalized average difference
  return diffSum / gray.length;
}

function startMotionDetection() {
  if (motionTimer) clearInterval(motionTimer);
  prevMotionGray = null;
  lastMotionScore = 0;

  motionTimer = setInterval(() => {
    lastMotionScore = computeMotionScore();
  }, MOTION_SAMPLE_MS);
}

// ===== Backend calls =====
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
    console.error('Static prediction error:', error);
    return { success: false, prediction: 'Error', confidence: 0, error: error.message };
  }
}

async function sendDynamicPrediction(imagesBase64) {
  try {
    const startTime = Date.now();
    const response = await fetch(`${BACKEND_URL}/predict/dynamic`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ images: imagesBase64, confidence_threshold: 0.6 })
    });
    const result = await response.json();
    return { ...result, responseTime: Date.now() - startTime };
  } catch (error) {
    console.error('Dynamic prediction error:', error);
    return { success: false, prediction: 'Error', confidence: 0, error: error.message };
  }
}

// ===== UI =====
function applyResultToUI(result) {
  if (result.success) {
    gestureText.textContent = result.prediction;

    const confidencePercent = Math.round((result.confidence || 0) * 100);
    confidenceBar.style.width = `${confidencePercent}%`;
    confidenceValue.textContent = `${confidencePercent}%`;

    handStatus.textContent = 'Yes';
    handStatus.className = 'status-value online';

    addToHistory(result.prediction, confidencePercent, result.responseTime);

    // speak ONLY if confident enough
    speakIfConfident(result.prediction, result.confidence);
  } else {
    handStatus.textContent = 'No';
    handStatus.className = 'status-value offline';

    if (result.prediction && String(result.prediction).toLowerCase().includes('insufficient data')) {
      gestureText.textContent = 'Keep Moving...';
    } else if (result.prediction === 'No hand detected') {
      gestureText.textContent = 'Show Your Hand';
    } else {
      showError(result.error || result.prediction || 'Prediction failed.');
    }
  }
}

// ===== Prediction flows =====
async function processStaticOnce() {
  const img = captureFrameBase64();
  if (!img) return;
  const result = await sendStaticPrediction(img);
  applyResultToUI(result);
}

function startDynamicCapture() {
  if (isCapturingDynamic || !isCameraOn) return;

  isCapturingDynamic = true;
  dynamicFrames = [];
  gestureText.textContent = "Capturing...";
  confidenceBar.style.width = "0%";
  confidenceValue.textContent = "0%";

  dynamicCaptureTimer = setInterval(() => {
    const img = captureFrameBase64();
    if (!img) return;

    dynamicFrames.push(img);

    if (dynamicFrames.length >= SEQ_LEN) {
      stopDynamicCapture();
      processDynamicSequence(dynamicFrames.slice(-SEQ_LEN));
    }
  }, DYNAMIC_INTERVAL_MS);
}

function stopDynamicCapture() {
  if (dynamicCaptureTimer) clearInterval(dynamicCaptureTimer);
  dynamicCaptureTimer = null;
  isCapturingDynamic = false;
}

async function processDynamicSequence(frames) {
  const result = await sendDynamicPrediction(frames);
  applyResultToUI(result);
}

// Auto decide based on motion
async function processAutoOnce() {
  // If moving a lot -> dynamic, else static
  const moving = lastMotionScore >= MOTION_THRESHOLD;

  if (moving) {
    startDynamicCapture();
  } else {
    await processStaticOnce();
  }
}

// Process once (Capture button)
async function processOnce() {
  if (!isCameraOn) return;

  const mode = modeSelect.value;
  if (mode === "static") return processStaticOnce();
  if (mode === "dynamic") return startDynamicCapture();
  return processAutoOnce(); // auto
}

// Toggle auto mode (repeat predictions)
function toggleAutoMode() {
  isAutoMode = !isAutoMode;

  if (isAutoMode) {
    autoModeBtn.textContent = '🔄 Auto Mode: ON';
    autoModeBtn.className = 'btn btn-success';

    // Run frequently; dynamic capture prevents overlap
    autoInterval = setInterval(() => {
      if (modeSelect.value !== "static" && isCapturingDynamic) return;
      processOnce();
    }, 2000);

  } else {
    autoModeBtn.textContent = '🔄 Auto Mode: OFF';
    autoModeBtn.className = 'btn btn-secondary';
    clearInterval(autoInterval);
    autoInterval = null;
    stopDynamicCapture();
  }
}

// History
function addToHistory(gesture, confidence, responseTime) {
  const timestamp = new Date().toLocaleTimeString();
  predictionHistory.unshift({ gesture, confidence, timestamp, responseTime });
  if (predictionHistory.length > 5) predictionHistory = predictionHistory.slice(0, 5);
  updateHistoryDisplay();
}

function updateHistoryDisplay() {
  historyList.innerHTML = '';

  if (predictionHistory.length === 0) {
    historyList.innerHTML = `
      <div class="history-item">
        <span>No predictions yet</span>
        <span>-</span>
      </div>
    `;
    return;
  }

  predictionHistory.forEach(item => {
    const el = document.createElement('div');
    el.className = 'history-item';
    el.innerHTML = `
      <span>${item.gesture}</span>
      <span>${item.confidence}% (${item.responseTime}ms)</span>
    `;
    historyList.appendChild(el);
  });
}

// TTS toggle
function toggleTTS() {
  ttsEnabled = !ttsEnabled;
  ttsToggleBtn.textContent = ttsEnabled ? "🔊 TTS: ON" : "🔇 TTS: OFF";
  if (!ttsEnabled) window.speechSynthesis.cancel();
  if (ttsEnabled) lastSpoken = "";
}

// If user changes mode while auto is ON, restart loop
function handleModeChange() {
  if (!isAutoMode) return;
  clearInterval(autoInterval);
  autoInterval = null;
  stopDynamicCapture();
  isAutoMode = false;
  toggleAutoMode();
}

// Event Listeners
startCameraBtn.addEventListener('click', startCamera);
captureBtn.addEventListener('click', processOnce);
autoModeBtn.addEventListener('click', toggleAutoMode);
ttsToggleBtn.addEventListener('click', toggleTTS);
modeSelect.addEventListener('change', handleModeChange);

// Initialize
async function initialize() {
  console.log('Initializing Sign Language Translator...');

  // 1. Enable buttons
  startCameraBtn.disabled = false;
  apiStatus.textContent = "Connecting to server...";
  apiStatus.className = "status-tag status-warning";

  // 2. Start checking backend status without awaiting 
  // Allow camera to work even if the backend is slow to wake up
  checkBackendStatus().then(online => {
    if (!online) {
      console.warn('Backend is still waking up...');
      // We don't disable the button here anymore, just show a warning
    }
  });

  // 3. Fetch gestures in the background
  try {
    const response = await fetch(`${BACKEND_URL}/gestures`);
    if (response.ok) {
      const data = await response.json();
      console.log('Available gestures:', data);
    }
  } catch (error) {
    console.error('Initial gesture fetch failed (Backend might be sleeping)');
  }
}

initialize();


