// =====================
// Configuration
// =====================
const BACKEND_URL = "https://signlanguage-detector-pi6d.onrender.com";

// Prediction settings
const STATIC_INTERVAL_MS = 900;   // static prediction cadence
const DYNAMIC_INTERVAL_MS = 1400; // dynamic prediction cadence
const DYNAMIC_BUFFER_MAX = 30;    // backend expects up to 30

// =====================
// DOM Elements
// =====================
const videoElement = document.getElementById("videoElement");
const startCameraBtn = document.getElementById("startCamera");
const captureBtn = document.getElementById("captureBtn");
const autoModeBtn = document.getElementById("autoMode");
const ttsToggleBtn = document.getElementById("ttsToggle");
const modeSelect = document.getElementById("modeSelect");

const handStatus = document.getElementById("handStatus");
const apiStatus = document.getElementById("apiStatus");
const gestureText = document.getElementById("gestureText");
const confidenceBar = document.getElementById("confidenceBar");
const confidenceValue = document.getElementById("confidenceValue");
const historyList = document.getElementById("historyList");
const backendUrlElement = document.getElementById("backendUrl");
const errorMessage = document.getElementById("errorMessage");

// =====================
// State
// =====================
let isCameraOn = false;
let isAutoMode = false;
let isTTS = true;

let lastSpoken = "";
let predictionHistory = [];

let frameBuffer = [];
let lastPredictAt = 0;
let isProcessing = false;

// We display a mirrored camera via CSS (video transform),
// BUT we capture frames un-mirrored for consistency.
// (Backend will flip if needed.)
function ensureMirroredDisplay() {
  // In case CSS wasn’t applied, force it here too:
  videoElement.style.transform = "scaleX(-1)";
  videoElement.style.objectFit = "cover";
}

// =====================
// Utilities
// =====================
function setAPIStatus(ok) {
  apiStatus.textContent = ok ? "Online" : "Offline";
  apiStatus.className = ok ? "status-value online" : "status-value offline";
  backendUrlElement.textContent = ok ? BACKEND_URL : "Error Connecting";
}

function setHandStatus(hasHand) {
  handStatus.textContent = hasHand ? "Yes" : "No";
  handStatus.className = hasHand ? "status-value online" : "status-value offline";
}

function setError(msg) {
  if (!errorMessage) return;
  if (!msg) {
    errorMessage.style.display = "none";
    errorMessage.textContent = "";
    return;
  }
  errorMessage.style.display = "block";
  errorMessage.textContent = msg;
}

function setPredictionText(text) {
  gestureText.textContent = text || "Waiting...";
}

function setConfidence(conf01) {
  const conf = Math.max(0, Math.min(1, Number(conf01) || 0));
  const pct = Math.round(conf * 100);
  confidenceBar.style.width = `${pct}%`;
  confidenceValue.textContent = `${pct}%`;
}

function speak(text) {
  if (!isTTS) return;
  if (!text) return;

  // Don’t speak these:
  const lower = String(text).toLowerCase();
  if (lower.includes("no hand")) return;
  if (lower.includes("incomplete")) return;
  if (lower.includes("error")) return;

  // Avoid repeating the same word over and over
  if (text === lastSpoken) return;

  try {
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utter);
    lastSpoken = text;
  } catch (_) {
    // If TTS fails in browser, just ignore quietly
  }
}

function addHistory(prediction, confidence01) {
  const conf = Math.round((Number(confidence01) || 0) * 100);
  const timestamp = new Date().toLocaleTimeString();
  predictionHistory.unshift({ gesture: prediction, conf, timestamp });
  if (predictionHistory.length > 5) predictionHistory.pop();
  historyList.innerHTML = predictionHistory
    .map(
      (item) =>
        `<div class="history-item"><span>${item.timestamp}</span><span>${item.gesture} (${item.conf}%)</span></div>`
    )
    .join("");
}

// =====================
// Backend status check
// =====================
async function checkBackendStatus() {
  try {
    const res = await fetch(`${BACKEND_URL}/health`, { method: "GET" });
    setAPIStatus(res.ok);
  } catch (e) {
    setAPIStatus(false);
  }
}

// =====================
// Camera
// =====================
async function startCamera() {
  setError("");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });

    videoElement.srcObject = stream;
    await videoElement.play();

    ensureMirroredDisplay();

    isCameraOn = true;
    startCameraBtn.disabled = true;
    captureBtn.disabled = false;

    setHandStatus(false);
  } catch (error) {
    setError("Camera access denied or unavailable: " + error.message);
  }
}

// Capture frame from the video.
// IMPORTANT: do NOT mirror the captured frame here — we only mirror display.
// This keeps backend processing consistent.
function captureFrameBase64Jpeg(quality = 0.75) {
  if (!isCameraOn) return null;
  const w = videoElement.videoWidth || 640;
  const h = videoElement.videoHeight || 480;

  const canvas = document.createElement("canvas");
  canvas.width = 480;   // fixed size for consistent backend inference
  canvas.height = 360;

  const ctx = canvas.getContext("2d");
  // Draw normally (un-mirrored)
  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

  return canvas.toDataURL("image/jpeg", quality);
}

// =====================
// Prediction
// =====================
async function callPredict(frames, mode) {
  const payload = { frames, mode };

  const res = await fetch(`${BACKEND_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  // If backend returns non-JSON, handle
  let data = null;
  try {
    data = await res.json();
  } catch (_) {
    data = { success: false, prediction: "Backend returned non-JSON response." };
  }

  // Attach status for debug
  data._http_ok = res.ok;
  data._status = res.status;
  return data;
}

function getSelectedMode() {
  // Your select values are: auto, static, dynamic
  const v = (modeSelect?.value || "auto").toLowerCase();
  if (v === "static") return "static";
  if (v === "dynamic") return "dynamic";
  return "auto";
}

async function processOnce(manual = false) {
  if (!isCameraOn) {
    setError("Camera is off.");
    return;
  }
  if (isProcessing) return;

  const mode = getSelectedMode();

  // Build frames
  const currentFrame = captureFrameBase64Jpeg(0.75);
  if (!currentFrame) return;

  if (mode === "dynamic") {
    frameBuffer.push(currentFrame);
    if (frameBuffer.length > DYNAMIC_BUFFER_MAX) frameBuffer.shift();
  } else {
    // static or auto -> don’t accumulate endlessly
    frameBuffer = [currentFrame];
  }

  // Throttle
  const now = Date.now();
  const interval = mode === "dynamic" ? DYNAMIC_INTERVAL_MS : STATIC_INTERVAL_MS;
  if (!manual && now - lastPredictAt < interval) return;

  // For dynamic, wait until we have enough frames
  if (mode === "dynamic" && frameBuffer.length < 20) {
    setPredictionText("Keep moving your hand...");
    setHandStatus(false);
    setConfidence(0);
    return;
  }

  isProcessing = true;
  lastPredictAt = now;

  try {
    const result = await callPredict(frameBuffer, mode);

    // Backend reachable but returned an error code
    if (!result._http_ok && result.detail) {
      setError(`Backend error (${result._status}): ${JSON.stringify(result.detail)}`);
      setPredictionText("Backend error");
      setHandStatus(false);
      setConfidence(0);
      isProcessing = false;
      return;
    }

    setError("");

    // Always show what backend says (no silent failures)
    if (result.success) {
      setPredictionText(result.prediction);
      setHandStatus(true);
      setConfidence(result.confidence);
      addHistory(result.prediction, result.confidence);
      speak(result.prediction);
    } else {
      setPredictionText(result.prediction || "No hand detected");
      setHandStatus(false);
      setConfidence(0);
    }
  } catch (e) {
    setError("API Error: " + e.message);
    setPredictionText("API error");
    setHandStatus(false);
    setConfidence(0);
  } finally {
    isProcessing = false;
  }
}

// Auto loop using requestAnimationFrame (smooth + efficient)
function autoLoop() {
  if (!isAutoMode) return;
  processOnce(false);
  requestAnimationFrame(autoLoop);
}

// =====================
// Toggles
// =====================
function toggleAutoMode() {
  isAutoMode = !isAutoMode;
  autoModeBtn.textContent = isAutoMode ? "🔄 Auto: ON" : "🔄 Auto: OFF";
  autoModeBtn.className = isAutoMode ? "btn btn-success" : "btn btn-secondary";

  if (isAutoMode) {
    setError("");
    autoLoop();
  }
}

function toggleTTS() {
  isTTS = !isTTS;
  ttsToggleBtn.textContent = isTTS ? "🔊 TTS: ON" : "🔇 TTS: OFF";
  if (!isTTS) {
    try { window.speechSynthesis.cancel(); } catch (_) {}
  }
}

// =====================
// Event listeners
// =====================
startCameraBtn.onclick = startCamera;
captureBtn.onclick = () => processOnce(true);
autoModeBtn.onclick = toggleAutoMode;
ttsToggleBtn.onclick = toggleTTS;

// If user changes mode, reset buffer to avoid weird transitions
modeSelect?.addEventListener("change", () => {
  frameBuffer = [];
  lastPredictAt = 0;
  setPredictionText("Waiting...");
  setConfidence(0);
  setHandStatus(false);
});

// =====================
// Initial check
// =====================
ensureMirroredDisplay();
checkBackendStatus();
setInterval(checkBackendStatus, 10000);
