const BACKEND_URL = "https://signlanguage-detector-pi6d.onrender.com";
let isAutoMode = false;
let frameBuffer = []; 
let isProcessing = false;

const video = document.getElementById('videoElement');
const autoBtn = document.getElementById('autoMode');
const gestureText = document.getElementById('gestureText');

// 1. Camera Initialization
document.getElementById('startCamera').onclick = async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        document.getElementById('cameraStatus').textContent = "On";
        document.getElementById('cameraStatus').className = "status-value online";
    } catch (err) {
        alert("Please allow camera access.");
    }
};

// 2. The Loop Logic
async function detectionLoop() {
    if (!isAutoMode) return;

    // Capture current frame
    const canvas = document.createElement('canvas');
    canvas.width = 480; canvas.height = 360;
    const ctx = canvas.getContext('2d');
    
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const frame = canvas.toDataURL('image/jpeg', 0.5);

    // Update the buffer
    frameBuffer.push(frame);
    if (frameBuffer.length > 30) frameBuffer.shift();

    // Send to API if not busy
    if (!isProcessing && frameBuffer.length >= 20) {
        isProcessing = true;
        const mode = document.getElementById('modeSelect').value;
        const payload = { frames: mode === 'static' ? [frame] : frameBuffer };

        try {
            const res = await fetch(`${BACKEND_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (data.success) updateUI(data);
        } catch (e) { console.error("API connection error"); }
        isProcessing = false;
    }

    requestAnimationFrame(detectionLoop);
}

// 3. Toggle Button Fix
autoBtn.onclick = () => {
    isAutoMode = !isAutoMode;
    if (isAutoMode) {
        autoBtn.textContent = "🔄 Auto Mode: ON";
        autoBtn.className = "btn btn-success";
        detectionLoop(); // Start the loop
    } else {
        autoBtn.textContent = "🔄 Auto Mode: OFF";
        autoBtn.className = "btn btn-secondary";
        frameBuffer = [];
        gestureText.textContent = "Standing By...";
    }
};

function updateUI(data) {
    gestureText.textContent = data.prediction;
    document.getElementById('handStatus').textContent = "Yes";
    document.getElementById('handStatus').className = "status-value online";
    document.getElementById('confidenceBar').style.width = (data.confidence * 100) + "%";
    document.getElementById('confidenceValue').textContent = Math.round(data.confidence * 100) + "%";
}
