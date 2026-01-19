const BACKEND_URL = "https://signlanguage-detector-pi6d.onrender.com";

let sessionId = null;
let stream = null;

let isAutoMode = false;
let isSending = false;
let ttsEnabled = true;

const video = document.getElementById("videoElement");
const startBtn = document.getElementById("startCamera");
const autoBtn = document.getElementById("autoMode");
const ttsBtn = document.getElementById("ttsToggle");

const gestureText = document.getElementById("gestureText");
const confidenceBar = document.getElementById("confidenceBar");
const confidenceValue = document.getElementById("confidenceValue");
const cameraStatus = document.getElementById("cameraStatus");
const handStatus = document.getElementById("handStatus");
const apiStatus = document.getElementById("apiStatus");
const backendUrlSpan = document.getElementById("backendUrl");
const errorBox = document.getElementById("errorMessage");
const historyList = document.getElementById("historyList");

const modeSelect = document.getElementById("modeSelect"); // auto/static/dynamic

backendUrlSpan.textContent = BACKEND_URL;

function showError(msg) {
  errorBox.style.display = "block";
  errorBox.textContent = msg;
}

function clearError() {
  errorBox.style.display = "none";
  errorBox.textContent = "";
}

function setStatus(el, on, onText, offText) {
  el.textContent = on ? onText : offText;
  el.className = "status-value " + (on ? "online" : "offline");
}

function getBase64Frame() {
  const canvas = document.createElement("canvas");
  canvas.width = 480;
  canvas.height = 360;
  const ctx = canvas.getContext("2d");

  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  return canvas.toDataURL("image/jpeg", 0.65);
}

function speak(text) {
  if (!ttsEnabled) return;
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 0.95;
  window.speechSynthesis.speak(u);
}

function addHistory(mode, label, conf) {
  const row = document.createElement("div");
  row.className = "history-item";
  row.innerHTML = `<span>${mode}: ${label}</span><span>${Math.round(conf * 100)}%</span>`;
  historyList.prepend(row);

  // keep last 12
  while (historyList.children.length > 12) {
    historyList.removeChild(historyList.lastChild);
  }
}

async function pingHealth() {
  try {
    const res = await fetch(`${BACKEND_URL}/health`, { method: "GET" });
    setStatus(apiStatus, res.ok, "Online", "Offline");
  } catch {
    setStatus(apiStatus, false, "Online", "Offline");
  }
}

//mainLoop for the State-Machine Backend
async function mainLoop() {
    if (!isAutoMode) return;

    const currentFrame = getBase64Frame(); // Get just ONE frame

    const payload = {
        session_id: "user_123", 
        frame_b64: currentFrame,
        motion_threshold: 0.15
    };

    try {
        const res = await fetch(`${BACKEND_URL}/predict_frame`, { 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const data = await res.json();
        
        if (data.success) {
            // Update the UI with the prediction
            gestureText.textContent = data.label;
            
            // Update confidence bar if provided
            if (data.confidence) {
                updateUI(data); 
            }
        }
    } catch (e) {
        console.error("API Error:", e);
    }

    // Run again immediately for smooth tracking
    requestAnimationFrame(mainLoop);
}

async function sendFrameOnce() {
  if (!isAutoMode) return;
  if (isSending) return;
  if (!video.srcObject) return;

  isSending = true;
  clearError();

  const frame_b64 = getBase64Frame();

  const payload = {
    session_id: sessionId,
    frame_b64,
    motion_threshold: 0.15,
    static_confidence: 0.70,
    dynamic_confidence: 0.60,
    force_mode: modeSelect.value // auto/static/dynamic
  };

  try {
    const res = await fetch(`${BACKEND_URL}/predict_frame`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error("API request failed");

    const data = await res.json();
    if (!data.success) throw new Error(data.error || "Prediction failed");

    sessionId = data.session_id;

    // UI update
    setStatus(handStatus, data.hand, "Yes", "No");

    gestureText.textContent = data.label || "—";
    const conf = data.confidence || 0;
    confidenceBar.style.width = `${Math.max(0, Math.min(100, conf * 100))}%`;
    confidenceValue.textContent = `${Math.round(conf * 100)}%`;

    // Speak only on real predictions
    if (data.predicted && data.label && data.label !== "Uncertain" && data.label !== "Low Confidence") {
      addHistory(data.mode, data.label, conf);
      speak(data.label);
    }
  } catch (e) {
    showError(String(e.message || e));
    setStatus(apiStatus, false, "Online", "Offline");
  } finally {
    isSending = false;
  }
}

let loopTimer = null;
function startAutoLoop() {
  if (loopTimer) clearInterval(loopTimer);
  loopTimer = setInterval(sendFrameOnce, 140);
}
function stopAutoLoop() {
  if (loopTimer) clearInterval(loopTimer);
  loopTimer = null;
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    video.srcObject = stream;
    setStatus(cameraStatus, true, "On", "Off");
    clearError();
  } catch (e) {
    showError("Could not access camera. Please allow permissions.");
    setStatus(cameraStatus, false, "On", "Off");
  }
}

function setAutoUI(on) {
  isAutoMode = on;
  autoBtn.textContent = on ? "🔄 Auto Mode: ON" : "🔄 Auto Mode: OFF";
  autoBtn.className = on ? "btn btn-success" : "btn btn-secondary";

  if (on) {
    gestureText.textContent = "Detecting...";
    startAutoLoop();
  } else {
    stopAutoLoop();
    gestureText.textContent = "Standing By...";
    confidenceBar.style.width = "0%";
    confidenceValue.textContent = "0%";
  }
}

function setTTSUI(on) {
  ttsEnabled = on;
  ttsBtn.textContent = on ? "🔊 TTS: ON" : "🔇 TTS: OFF";
  ttsBtn.className = on ? "btn btn-success" : "btn btn-secondary";
}

// Bind
startBtn.addEventListener("click", async () => {
  await startCamera();
  await pingHealth();
  // Optional: auto start when camera starts
  setAutoUI(true);
});

autoBtn.addEventListener("click", () => setAutoUI(!isAutoMode));
ttsBtn.addEventListener("click", () => setTTSUI(!ttsEnabled));

// initial UI
setAutoUI(false);
setTTSUI(true);
pingHealth();


