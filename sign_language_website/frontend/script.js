// Configuration
const BACKEND_URL = "https://signlanguage-detector-pi6d.onrender.com";

// DOM Elements
const videoElement = document.getElementById('videoElement');
const startCameraBtn = document.getElementById('startCamera');
const captureBtn = document.getElementById('captureBtn');
const autoModeBtn = document.getElementById('autoMode');
const handStatus = document.getElementById('handStatus');
const apiStatus = document.getElementById('apiStatus');
const gestureText = document.getElementById('gestureText');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceValue = document.getElementById('confidenceValue');
const historyList = document.getElementById('historyList');
const backendUrlElement = document.getElementById('backendUrl');

// State
let isCameraOn = false;
let isAutoMode = false;
let isTTS = true;
let autoInterval = null;
let lastSpoken = "";
let predictionHistory = [];

// Text-to-Speech
function speak(text) {
    if (!isTTS || !text || text === lastSpoken || text === "No hand detected") return;
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
    lastSpoken = text;
}

// Check backend status
async function checkBackendStatus() {
    try {
        const response = await fetch(`${BACKEND_URL}/health`);
        if (response.ok) {
            apiStatus.textContent = 'Online';
            apiStatus.className = 'status-value online';
            backendUrlElement.textContent = BACKEND_URL;
        }
    } catch (error) {
        apiStatus.textContent = 'Offline';
        apiStatus.className = 'status-value offline';
        backendUrlElement.textContent = 'Error Connecting';
    }
}

// Start camera
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;
        isCameraOn = true;
        startCameraBtn.disabled = true;
        captureBtn.disabled = false;
    } catch (error) {
        alert("Camera access denied: " + error.message);
    }
}

// Capture frame (Mirrored for natural interaction)
function captureFrame() {
    if (!isCameraOn) return null;
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const context = canvas.getContext('2d');
    context.save();
    context.scale(-1, 1);
    context.drawImage(videoElement, -canvas.width, 0, canvas.width, canvas.height);
    context.restore();
    return canvas.toDataURL('image/jpeg', 0.8);
}

// Process Frame & Update History
async function processFrame() {
    if (!isCameraOn) return;
    const imageData = captureFrame();
    try {
        const response = await fetch(`${BACKEND_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        const result = await response.json();
        
        if (result.success) {
            gestureText.textContent = result.prediction;
            speak(result.prediction);
            const conf = Math.round(result.confidence * 100);
            confidenceBar.style.width = `${conf}%`;
            confidenceValue.textContent = `${conf}%`;
            handStatus.textContent = 'Yes';
            handStatus.className = 'status-value online';
            
            // Add to history list
            const timestamp = new Date().toLocaleTimeString();
            predictionHistory.unshift({ gesture: result.prediction, conf, timestamp });
            if (predictionHistory.length > 5) predictionHistory.pop();
            updateHistoryUI();
        } else {
            gestureText.textContent = 'Show Your Hand';
            handStatus.textContent = 'No';
            handStatus.className = 'status-value offline';
        }
    } catch (e) { console.error("API Error:", e); }
}

function updateHistoryUI() {
    historyList.innerHTML = predictionHistory.map(item => 
        `<div style="margin-bottom:5px;"><b>${item.timestamp}</b>: ${item.gesture} (${item.conf}%)</div>`
    ).join('');
}

// Toggle functions
function toggleAutoMode() {
    isAutoMode = !isAutoMode;
    autoModeBtn.textContent = isAutoMode ? '🔄 Auto: ON' : '🔄 Auto: OFF';
    autoModeBtn.className = isAutoMode ? 'btn btn-success' : 'btn btn-secondary';
    if (isAutoMode) autoInterval = setInterval(processFrame, 2000);
    else clearInterval(autoInterval);
}

// Event Listeners
startCameraBtn.onclick = startCamera;
captureBtn.onclick = processFrame;
autoModeBtn.onclick = toggleAutoMode;
document.getElementById('ttsToggle').onclick = () => {
    isTTS = !isTTS;
    document.getElementById('ttsToggle').textContent = isTTS ? "🔊 TTS: ON" : "🔇 TTS: OFF";
};

// Initial check
checkBackendStatus();
