// Configuration
const BACKEND_URL = "https://signlanguage-detector-pi6d.onrender.com";

// DOM Elements
const videoElement = document.getElementById('videoElement');
const startCameraBtn = document.getElementById('startCamera');
const captureBtn = document.getElementById('captureBtn');
const autoModeBtn = document.getElementById('autoMode');
const cameraStatus = document.getElementById('cameraStatus');
const handStatus = document.getElementById('handStatus');
const apiStatus = document.getElementById('apiStatus');
const gestureText = document.getElementById('gestureText');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceValue = document.getElementById('confidenceValue');
const historyList = document.getElementById('historyList');
const errorMessage = document.getElementById('errorMessage');
const backendUrlElement = document.getElementById('backendUrl');

// State
let stream = null;
let isCameraOn = false;
let isAutoMode = false;
let autoInterval = null;
let predictionHistory = [];

// Hide error message
function hideError() {
    errorMessage.style.display = 'none';
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    setTimeout(hideError, 5000);
}

// Check backend status
async function checkBackendStatus() {
    try {
        const response = await fetch(`${BACKEND_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            apiStatus.textContent = 'Online';
            apiStatus.className = 'status-value online';
            backendUrlElement.textContent = BACKEND_URL;
            return true;
        }
    } catch (error) {
        console.error('Backend offline:', error);
        apiStatus.textContent = 'Offline';
        apiStatus.className = 'status-value offline';
        backendUrlElement.textContent = 'Not connected';
        showError('Backend server is offline. Please try again later.');
        return false;
    }
}

// Start camera
async function startCamera() {
    try {
        const constraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user'
            },
            audio: false
        };

        stream = await navigator.mediaDevices.getUserMedia(constraints);
        videoElement.srcObject = stream;
        
        isCameraOn = true;
        cameraStatus.textContent = 'On';
        cameraStatus.className = 'status-value online';
        startCameraBtn.disabled = true;
        captureBtn.disabled = false;
        
        console.log('Camera started successfully');
    } catch (error) {
        console.error('Error accessing camera:', error);
        showError(`Cannot access camera: ${error.message}. Please check permissions.`);
    }
}

// Capture frame from video
function captureFrame() {
    if (!isCameraOn) return null;
    
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const context = canvas.getContext('2d');
    
    // Draw video frame to canvas (mirrored)
    context.save();
    context.scale(-1, 1);
    context.drawImage(videoElement, -canvas.width, 0, canvas.width, canvas.height);
    context.restore();
    
    // Convert to base64
    return canvas.toDataURL('image/jpeg', 0.8);
}

// Send prediction request to backend
async function sendPredictionRequest(imageData) {
    try {
        const startTime = Date.now();
        
        const response = await fetch(`${BACKEND_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageData
            })
        });
        
        const result = await response.json();
        const endTime = Date.now();
        
        return {
            ...result,
            responseTime: endTime - startTime
        };
        
    } catch (error) {
        console.error('Prediction error:', error);
        return {
            success: false,
            prediction: 'Error',
            confidence: 0,
            error: error.message
        };
    }
}

// Process captured frame
async function processFrame() {
    if (!isCameraOn) return;
    
    const imageData = captureFrame();
    if (!imageData) return;
    
    const result = await sendPredictionRequest(imageData);
    
    if (result.success) {
        // Update UI
        gestureText.textContent = result.prediction;
        const confidencePercent = Math.round(result.confidence * 100);
        confidenceBar.style.width = `${confidencePercent}%`;
        confidenceValue.textContent = `${confidencePercent}%`;
        
        // Update hand status
        handStatus.textContent = 'Yes';
        handStatus.className = 'status-value online';
        
        // Add to history
        addToHistory(result.prediction, confidencePercent, result.responseTime);
        
    } else {
        handStatus.textContent = 'No';
        handStatus.className = 'status-value offline';
        
        if (result.prediction === 'No hand detected') {
            gestureText.textContent = 'Show Your Hand';
        }
    }
}

// Add to prediction history
function addToHistory(gesture, confidence, responseTime) {
    const timestamp = new Date().toLocaleTimeString();
    const historyItem = {
        gesture,
        confidence,
        timestamp,
        responseTime
    };
    
    predictionHistory.unshift(historyItem);
    if (predictionHistory.length > 5) {
        predictionHistory = predictionHistory.slice(0, 5);
    }
    
    updateHistoryDisplay();
}

// Update history display
function updateHistoryDisplay() {
    historyList.innerHTML = '';
    
    if (predictionHistory.length === 0) {
        historyList.innerHTML = `
            <div class="history-item">
                <span>No predictions yet</span>
                <span>-</span>
            </div>
        `;
        return;
    }
    
    predictionHistory.forEach(item => {
        const historyElement = document.createElement('div');
        historyElement.className = 'history-item';
        historyElement.innerHTML = `
            <span>${item.gesture}</span>
            <span>${item.confidence}% (${item.responseTime}ms)</span>
        `;
        historyList.appendChild(historyElement);
    });
}

// Toggle auto mode
function toggleAutoMode() {
    isAutoMode = !isAutoMode;
    
    if (isAutoMode) {
        autoModeBtn.textContent = '🔄 Auto Mode: ON';
        autoModeBtn.className = 'btn btn-success';
        autoInterval = setInterval(processFrame, 2000); // Every 2 seconds
    } else {
        autoModeBtn.textContent = '🔄 Auto Mode: OFF';
        autoModeBtn.className = 'btn btn-secondary';
        clearInterval(autoInterval);
        autoInterval = null;
    }
}

// Event Listeners
startCameraBtn.addEventListener('click', startCamera);
captureBtn.addEventListener('click', processFrame);
autoModeBtn.addEventListener('click', toggleAutoMode);

// Initialize
async function initialize() {
    console.log('Initializing Sign Language Translator...');
    
    // Check backend status
    const isBackendOnline = await checkBackendStatus();
    
    if (!isBackendOnline) {
        showError('Cannot connect to AI server. Please try again later.');
        startCameraBtn.disabled = true;
    }
    
    // Get available gestures (optional)
    try {
        const response = await fetch(`${BACKEND_URL}/gestures`);
        const data = await response.json();
        console.log('Available gestures:', data);
    } catch (error) {
        console.error('Could not fetch gestures:', error);
    }
}

// Start the app
initialize();


