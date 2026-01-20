const BACKEND_URL = "https://signlanguage-detector-pi6d.onrender.com";

let autoMode = false;
let stream = null;

const video = document.getElementById("video");
const startCameraBtn = document.getElementById("startCamera");
const autoModeBtn = document.getElementById("autoMode");
const ttsBtn = document.getElementById("ttsToggle");
const modeSelect = document.getElementById("modeSelect");

const gestureText = document.getElementById("gestureText");
const confidenceBar = document.getElementById("confidenceBar");
const confidenceValue = document.getElementById("confidenceValue");

const cameraStatus = document.getElementById("cameraStatus");
const handStatus = document.getElementById("handStatus");
const apiStatus = document.getElementById("apiStatus");
const errorMessage = document.getElementById("errorMessage");

startCameraBtn.onclick = async () => {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    cameraStatus.innerText = "Camera On";
};

autoModeBtn.onclick = () => {
    autoMode = !autoMode;
    autoModeBtn.innerText = autoMode ? "🔄 Auto Mode: ON" : "🔄 Auto Mode: OFF";
};

ttsBtn.onclick = () => {
    ttsBtn.innerText = ttsBtn.innerText.includes("ON") ? "🔇 TTS: OFF" : "🔊 TTS: ON";
};
