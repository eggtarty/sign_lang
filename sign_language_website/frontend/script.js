const BACKEND_URL = "https://signlanguage-detector-pi6d.onrender.com";

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

let isCameraOn = false;
let isAutoMode = false;
let isTTS = true;
let lastSpoken = "";
let predictionHistory = [];

let frameBuffer = [];
let isProcessing = false;
let lastRequestAt = 0;

const STATIC_INTERVAL_MS = 900;
const DYNAMIC_INTERVAL_MS = 1400;

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

function speak(text) {
  if (!isTTS || !text) return;
  const lower = String(text).toLowerCase();
  if (lower.includes("no hand") || lower.includes("incomplete") || lower.includes("error")) return;
  if (text === lastSpoken) return;

  try {
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(u);
    lastSpoken = text;
  } catch (_) {}
}

async function checkBackendStatus() {
  try {
    const res = await fetch(`${BACKEND_URL}/health`, { method: "GET" });
    if (res.ok) {
      apiStatus.textContent = "Online";
      apiStatus.className = "status-value online";
      backendUrlElement.textContent = BACKEND_URL;
    } else {
      throw new Error("Not OK");
    }
  } catch {
    apiStatus.textContent = "Offline";
    apiStatus.className = "status-value offline";
    backendUrlElement.textContent = "Error Connecting";
  }
}

async function startCamera() {
  setError("");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    videoElement.srcObject = stream;
    await videoElement.play();

    isCameraOn = true;
    startCameraBtn.disabled = true;
    captureBtn.disabled = false;

    // Mirrored display only
    videoElement.style.transform = "scaleX(-1)";
    videoElement.style.objectFit = "cover";
  } catch (e) {
    setError("Camera access denied: " + e.message);
  }
}

// Capture frame WITHOUT mirroring (display is mirrored only)
function captureFrame() {
  if (!isCameraOn) return null;
  const canvas = document.createElement("canvas");
  canvas.width = 480;
  canvas.height = 360;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.75);
}

function addHistory(pred, conf01) {
  const conf = Math.round((Number(conf01) || 0) * 100);
  const timestamp = new Date().toLocaleTimeString();
  predictionHistory.unshift({ pred, conf, timestamp });
  if (predictionHistory.length > 5) predictionHistory.pop();

  historyList.innerHTML = predictionHistory
    .map(
      (x) =>
        `<div class="history-item"><span>${x.timestamp}</span><span>${x.pred} (${x.conf}%)</span></div>`
    )
    .join("");
}

function setHand(hasHand) {
  handStatus.textContent = hasHand ? "Yes" : "No";
  handStatus.className = hasHand ? "status-value online" : "status-value offline";
}

function setConfidence(conf01) {
  const conf = Math.max(0, Math.min(1, Number(conf01) || 0));
  const pct = Math.round(conf * 100);
  confidenceBar.style.width = `${pct}%`;
  confidenceValue.textContent = `${pct}%`;
}

function getMode() {
  const v = (modeSelect?.value || "auto").toLowerCase();
  if (v === "static") return "static";
  if (v === "dynamic") return "dynamic";
  return "auto";
}

async function processFrame(manual = false) {
  if (!isCameraOn || isProcessing) return;

  const mode = getMode();
  const interval = mode === "dynamic" ? DYNAMIC_INTERVAL_MS : STATIC_INTERVAL_MS;

  const now = Date.now();
  if (!manual && now - lastRequestAt < interval) return;

  const frame = captureFrame();
  if (!frame) return;

  // Build payload
  let framesToSend;
  if (mode === "dynamic") {
    frameBuffer.push(frame);
    if (frameBuffer.length > 30) frameBuffer.shift();
    framesToSend = frameBuffer;

    // Wait until enough frames for dynamic
    if (!manual && framesToSend.length < 20) {
      gestureText.textContent = "Keep moving your hand...";
      setHand(false);
      setConfidence(0);
      return;
    }
  } else {
    // static or auto -> single frame
    framesToSend = [frame];
    frameBuffer = [];
  }

  isProcessing = true;
  lastRequestAt = now;

  try {
    const res = await fetch(`${BACKEND_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames: framesToSend, mode }) 
    });

    // Read response safely
    const text = await res.text();
    let data = {};
    try { data = JSON.parse(text); } catch { data = { prediction: text }; }

    if (!res.ok) {
      // Show FastAPI validation errors if present
      const detail = data?.detail;
      const msg =
        Array.isArray(detail) && detail.length
          ? `Request invalid: ${detail[0]?.msg || "422"} (${detail[0]?.loc?.join(".") || ""})`
          : `Backend error (${res.status})`;

      setError(msg);
      gestureText.textContent = "Error";
      setHand(false);
      setConfidence(0);
      return;
    }

    setError("");
    gestureText.textContent = data.prediction || "No response";

    if (data.success) {
      setHand(true);
      setConfidence(data.confidence);
      addHistory(data.prediction, data.confidence);
      speak(data.prediction);
    } else {
      setHand(false);
      setConfidence(0);
    }
  } catch (e) {
    setError("API Error: " + e.message);
    gestureText.textContent = "API error";
    setHand(false);
    setConfidence(0);
  } finally {
    isProcessing = false;
  }
}

function toggleAutoMode() {
  isAutoMode = !isAutoMode;
  autoModeBtn.textContent = isAutoMode ? "🔄 Auto: ON" : "🔄 Auto: OFF";
  autoModeBtn.className = isAutoMode ? "btn btn-success" : "btn btn-secondary";

  if (isAutoMode) {
    const loop = () => {
      if (!isAutoMode) return;
      processFrame(false);
      requestAnimationFrame(loop);
    };
    loop();
  }
}

// Events
startCameraBtn.onclick = startCamera;
captureBtn.onclick = () => processFrame(true);
autoModeBtn.onclick = toggleAutoMode;
ttsToggleBtn.onclick = () => {
  isTTS = !isTTS;
  ttsToggleBtn.textContent = isTTS ? "🔊 TTS: ON" : "🔇 TTS: OFF";
  if (!isTTS) {
    try { window.speechSynthesis.cancel(); } catch (_) {}
  }
};

// If mode changes, reset buffers
modeSelect?.addEventListener("change", () => {
  frameBuffer = [];
  lastRequestAt = 0;
  gestureText.textContent = "Waiting...";
  setHand(false);
  setConfidence(0);
  setError("");
});

checkBackendStatus();
setInterval(checkBackendStatus, 10000);
