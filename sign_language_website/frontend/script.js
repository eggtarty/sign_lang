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

function setError(msg) {
  if (!errorMessage) return;
  if (!msg) { errorMessage.style.display = "none"; errorMessage.textContent = ""; return; }
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
    const res = await fetch(`${BACKEND_URL}/health`);
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
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoElement.srcObject = stream;
    await videoElement.play();
    isCameraOn = true;
    startCameraBtn.disabled = true;
    captureBtn.disabled = false;

    // Mirrored display (user sees mirror)
    videoElement.style.transform = "scaleX(-1)";
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
    .map(x => `<div class="history-item"><span>${x.timestamp}</span><span>${x.pred} (${x.conf}%)</span></div>`)
    .join("");
}

async function processFrame() {
  if (!isCameraOn || isProcessing) return;
  isProcessing = true;

  const frame = captureFrame();
  if (!frame) { isProcessing = false; return; }

  const mode = (modeSelect?.value || "auto").toLowerCase();

  // Build payload
  let framesToSend;
  if (mode === "dynamic") {
    frameBuffer.push(frame);
    if (frameBuffer.length > 30) frameBuffer.shift();
    framesToSend = frameBuffer;
  } else {
    // auto + static => send single frame (most reliable)
    framesToSend = [frame];
    frameBuffer = []; // keep buffer clean
  }

  try {
    const res = await fetch(`${BACKEND_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames: framesToSend, mode })
    });

    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      setError(`Backend error (${res.status}).`);
      gestureText.textContent = "Backend error";
      handStatus.textContent = "No";
      handStatus.className = "status-value offline";
      isProcessing = false;
      return;
    }

    setError("");

    // Always show message (no silent fail)
    gestureText.textContent = data.prediction || "No response";

    if (data.success) {
      const conf = Number(data.confidence || 0);
      confidenceBar.style.width = `${Math.round(conf * 100)}%`;
      confidenceValue.textContent = `${Math.round(conf * 100)}%`;

      handStatus.textContent = "Yes";
      handStatus.className = "status-value online";

      addHistory(data.prediction, conf);
      speak(data.prediction);
    } else {
      confidenceBar.style.width = "0%";
      confidenceValue.textContent = "0%";
      handStatus.textContent = "No";
      handStatus.className = "status-value offline";
    }

  } catch (e) {
    setError("API Error: " + e.message);
  } finally {
    isProcessing = false;
  }
}

function toggleAutoMode() {
  isAutoMode = !isAutoMode;
  autoModeBtn.textContent = isAutoMode ? "🔄 Auto: ON" : "🔄 Auto: OFF";
  autoModeBtn.className = isAutoMode ? "btn btn-success" : "btn btn-secondary";

  if (isAutoMode) {
    const loop = async () => {
      if (!isAutoMode) return;
      await processFrame();
      requestAnimationFrame(loop);
    };
    loop();
  }
}

startCameraBtn.onclick = startCamera;
captureBtn.onclick = processFrame;
autoModeBtn.onclick = toggleAutoMode;
ttsToggleBtn.onclick = () => {
  isTTS = !isTTS;
  ttsToggleBtn.textContent = isTTS ? "🔊 TTS: ON" : "🔇 TTS: OFF";
};

checkBackendStatus();
setInterval(checkBackendStatus, 10000);
