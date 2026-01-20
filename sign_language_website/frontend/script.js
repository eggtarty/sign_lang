[cite_start]// Configuration
const BACKEND_URL = "https://signlanguage-detector-pi6d.onrender.com";
const video = document.getElementById('videoElement');
const gestureText = document.getElementById('gestureText');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceValue = document.getElementById('confidenceValue');

let isAuto = false;
let isTTS = true;
let autoTimer = null;
let lastSpoken = "";

// Text-to-Speech Function
function speak(text) {
    if (!isTTS || !text || text === lastSpoken || text === "No hand detected") return;
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
    lastSpoken = text;
}

[cite_start]// Check backend status 
async function checkStatus() {
    try {
        const res = await fetch(`${BACKEND_URL}/health`);
        if (res.ok) {
            document.getElementById('apiStatus').textContent = "Online";
            document.getElementById('apiStatus').className = "status-value online";
        }
    } catch (e) { console.log("Waiting for backend..."); }
}

[cite_start]// Start Camera 
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        document.getElementById('startCamera').disabled = true;
        document.getElementById('captureBtn').disabled = false;
    } catch (e) { alert("Camera error: " + e.message); }
}

[cite_start]// Capture and Mirror 
function captureFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1); // Mirror capture 
    ctx.drawImage(video, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.8);
}

[cite_start]// Send Prediction 
async function predict() {
    const img = captureFrame();
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
            confidenceValue.textContent = conf + "% Confidence";
            document.getElementById('handStatus').textContent = "Yes";
            document.getElementById('handStatus').className = "status-value online";
        } else {
            gestureText.textContent = "Show Your Hand";
            document.getElementById('handStatus').textContent = "No";
            document.getElementById('handStatus').className = "status-value offline";
        }
    } catch (e) { console.error("Prediction failed", e); }
}

[cite_start]
document.getElementById('startCamera').onclick = startCamera;
document.getElementById('captureBtn').onclick = predict;
document.getElementById('ttsToggle').onclick = () => {
    isTTS = !isTTS;
    document.getElementById('ttsToggle').textContent = isTTS ? "🔊 TTS: ON" : "🔇 TTS: OFF";
};
document.getElementById('autoMode').onclick = () => {
    isAuto = !isAuto;
    document.getElementById('autoMode').textContent = isAuto ? "🔄 Auto: ON" : "🔄 Auto: OFF";
    if (isAuto) autoTimer = setInterval(predict, 2000); [cite_start]
    else clearInterval(autoTimer);
};

checkStatus(); [cite_start]// Initial check [cite: 44]
