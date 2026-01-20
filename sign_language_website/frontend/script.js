const BACKEND_URL = "https://signlanguage-detector-pi6d.onrender.com";
const video = document.getElementById('videoElement');
const gestureText = document.getElementById('gestureText');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceValue = document.getElementById('confidenceValue');

let isAuto = false;
let autoTimer = null;

async function checkStatus() {
    try {
        const res = await fetch(`${BACKEND_URL}/health`);
        if (res.ok) {
            document.getElementById('apiStatus').textContent = "Online";
            document.getElementById('apiStatus').className = "status-value online";
        }
    } catch (e) { console.log("Backend waking up..."); }
}

async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    document.getElementById('startCamera').disabled = true;
    document.getElementById('captureBtn').disabled = false;
}

function captureFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    // Mirroring for the AI
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.8);
}

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
            const conf = Math.round(data.confidence * 100);
            confidenceBar.style.width = conf + "%";
            confidenceValue.textContent = conf + "% Confidence";
            document.getElementById('handStatus').textContent = "Yes";
            document.getElementById('handStatus').className = "status-value online";
        } else {
            document.getElementById('handStatus').textContent = "No";
            document.getElementById('handStatus').className = "status-value offline";
        }
    } catch (e) { console.error(e); }
}

document.getElementById('startCamera').onclick = startCamera;
document.getElementById('captureBtn').onclick = predict;
document.getElementById('autoMode').onclick = () => {
    isAuto = !isAuto;
    document.getElementById('autoMode').textContent = isAuto ? "🔄 Auto: ON" : "🔄 Auto: OFF";
    if (isAuto) autoTimer = setInterval(predict, 2000);
    else clearInterval(autoTimer);
};

checkStatus();
