// Configuration
const BACKEND_URL = "https://signlanguage-detector-pi6d.onrender.com";

// DOM Elements
const videoElement = document.getElementById('videoElement');
const gestureText = document.getElementById('gestureText');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceValue = document.getElementById('confidenceValue');
const apiStatus = document.getElementById('apiStatus');
const handStatus = document.getElementById('handStatus');

let isAuto = false;
let isTTS = true; // Text-to-Speech Toggle
let autoTimer = null;
let lastSpoken = "";

// 1. Text-to-Speech Function
function speak(text) {
    if (!isTTS || !text || text === lastSpoken || text === "No hand detected") return;
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
    lastSpoken = text;
}

// 2. Health Check
async function checkStatus() {
    try {
        const res = await fetch(`${BACKEND_URL}/health`);
        if (res.ok) {
            apiStatus.textContent = "Online";
            apiStatus.className = "status-value online";
        }
    } catch (e) {
        apiStatus.textContent = "Offline";
        apiStatus.className = "status-value offline";
    }
}

// 3. Camera Start
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;
        document.getElementById('startCamera').disabled = true;
        document.getElementById('captureBtn').disabled = false;
    } catch (e) { alert("Camera Error: " + e.message); }
}

// 4. Capture & Mirror Frame
function captureFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1); // Mirror for natural feel
    ctx.drawImage(videoElement, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.8);
}

// 5. Prediction Logic
async function predict() {
    const img = captureFrame();
    if (!img) return;

    try {
        const res = await fetch(`${BACKEND_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: img })
        });
        const data = await res.json();
        
        if (data.success) {
            gestureText.textContent = data.prediction;
            speak(data.prediction); // Trigger TTS
            const conf = Math.round(data.confidence * 100);
            confidenceBar.style.width = conf + "%";
            confidenceValue.textContent = conf + "%";
            handStatus.textContent = "Yes";
        } else {
            gestureText.textContent = "Show Your Hand";
            handStatus.textContent = "No";
        }
    } catch (e) { console.error("Prediction Error", e); }
}

// Listeners
document.getElementById('startCamera').onclick = startCamera;
document.getElementById('captureBtn').onclick = predict;
document.getElementById('autoMode').onclick = () => {
    isAuto = !isAuto;
    document.getElementById('autoMode').textContent = isAuto ? "🔄 Auto: ON" : "🔄 Auto: OFF";
    if (isAuto) autoTimer = setInterval(predict, 2000);
    else clearInterval(autoTimer);
};
document.getElementById('ttsToggle').onclick = () => {
    isTTS = !isTTS;
    document.getElementById('ttsToggle').textContent = isTTS ? "🔊 TTS: ON" : "🔇 TTS: OFF";
};

checkStatus();
