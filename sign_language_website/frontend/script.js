const BACKEND_URL = "https://signlanguage-detector-pi6d.onrender.com";
let isAutoMode = false;
let frameBuffer = [];
let isProcessing = false;

const video = document.getElementById('videoElement');
const autoBtn = document.getElementById('autoMode');
const gestureText = document.getElementById('gestureText');

// Mirroring the canvas is crucial for accuracy
function getBase64Frame() {
    const canvas = document.createElement('canvas');
    canvas.width = 480;
    canvas.height = 360;
    const ctx = canvas.getContext('2d');
    // Mirroring fix to match app.py behavior
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.6);
}

// THE MAIN LOOP (Similar to while True in app.py)
async function mainLoop() {
    if (!isAutoMode) return; // Exit if user turned it off

    const currentFrame = getBase64Frame();
    frameBuffer.push(currentFrame);
    if (frameBuffer.length > 30) frameBuffer.shift();

    // Trigger prediction every ~1.5 seconds if not already processing
    if (!isProcessing && frameBuffer.length >= 25) {
        isProcessing = true;
        
        const mode = document.getElementById('modeSelect').value;
        const payload = { frames: mode === 'static' ? [currentFrame] : frameBuffer };

        try {
            const res = await fetch(`${BACKEND_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (data.success) {
                gestureText.textContent = data.prediction;
                updateUI(data); // update confidence bars
            }
        } catch (e) {
            console.error("API Error");
        }
        isProcessing = false;
    }

    // High performance loop
    requestAnimationFrame(mainLoop);
}

// FIXED BUTTON FUNCTION
autoBtn.onclick = () => {
    isAutoMode = !isAutoMode;
    if (isAutoMode) {
        autoBtn.textContent = "🔄 Auto Mode: ON";
        autoBtn.className = "btn btn-success";
        mainLoop(); // Start the recursion
    } else {
        autoBtn.textContent = "🔄 Auto Mode: OFF";
        autoBtn.className = "btn btn-secondary";
        frameBuffer = [];
        gestureText.textContent = "Standing By...";
    }
};

// Start Camera on load
navigator.mediaDevices.getUserMedia({ video: true }).then(s => video.srcObject = s);
