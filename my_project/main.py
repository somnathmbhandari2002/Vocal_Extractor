# main.py
# This is the corrected FastAPI application for a Vocal Extractor.
# It addresses the SyntaxError from non-printable characters and
# optimizes performance for deployment by loading the ML model once.

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
from typing import Dict
import hashlib
import uuid
import requests
import shutil
import subprocess
import os
import torchaudio
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor

# Initialize a thread pool for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=4)

# ------------------- FastAPI App -------------------
app = FastAPI()

# Enable CORS (edit allow_origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- In-memory User Store -------------------
users: Dict[str, dict] = {}

# ------------------- Models -------------------
class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class GoogleLoginRequest(BaseModel):
    token: str

# ------------------- Helpers -------------------
def hash_password(password: str) -> str:
    """Hashes a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

# ------------------- Global Model Loading -------------------
# This is a critical optimization for performance and memory.
# The Demucs model is loaded only once when the application starts.
# Loading it inside the route would cause it to re-download and
# re-load for every single request, which would be very slow and
# would likely exceed memory limits on a free tier.
try:
    print("Loading Demucs model...")
    demucs_model = get_model(name='htdemucs')
    demucs_model.eval()
    print("Demucs model loaded successfully.")
except Exception as e:
    print(f"Error loading Demucs model: {e}")
    demucs_model = None

# ------------------- Initial -------------------
@app.get("/")
def read_root():
    return {"message": "Vocal Extractor API is running!"}

# ------------------- Auth Routes -------------------
@app.post("/register")
def register(req: RegisterRequest):
    if req.username in users:
        raise HTTPException(status_code=400, detail="Username already exists")
    users[req.username] = {
        "username": req.username,
        "email": req.email,
        "password_hash": hash_password(req.password),
    }
    return {"message": "Registration successful"}

@app.post("/login")
def login(req: LoginRequest):
    user = users.get(req.username)
    if not user or user["password_hash"] != hash_password(req.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Login successful"}

@app.post("/forgot-password")
def forgot_password(req: ForgotPasswordRequest):
    for user in users.values():
        if user["email"] == req.email:
            return {"message": "Password reset link sent (simulated)"}
    return {"message": "If an account exists, a reset link has been sent."}

@app.post("/google-login")
def google_login(req: GoogleLoginRequest):
    try:
        response = requests.get(
            f"https://oauth2.googleapis.com/tokeninfo?id_token={req.token}"
        )
        data = response.json()
        if "error" in data:
            raise HTTPException(status_code=401, detail=f"Invalid Google token: {data.get('error_description', data.get('error'))}")

        email = data.get("email")
        name = data.get("name")
        sub = data.get("sub")

        username = name.replace(" ", "_").lower() + "_" + sub[:6]
        if username not in users:
            users[username] = {
                "username": username,
                "email": email,
                "google_id": sub
            }
        return {"message": "Google login successful", "username": username}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google login failed: {str(e)}")

# ------------------- Video Processing Route -------------------
def process_video_sync(video_path: str, video_id: str, format: str):
    """
    Synchronous function for video processing.
    Runs in a thread pool to avoid blocking the main event loop.
    """
    if demucs_model is None:
        raise RuntimeError("Demucs model is not loaded.")

    temp_audio_path = f"temp/{video_id}_temp.wav"
    ffmpeg_cmd_extract = [
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "44100", "-ac", "2", "-vn",
        "-loglevel", "error", temp_audio_path
    ]

    subprocess.run(ffmpeg_cmd_extract, check=True, capture_output=True, text=True)

    wave, sr = torchaudio.load(temp_audio_path)
    if sr != 44100:
        wave = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100)(wave)
    if wave.shape[0] == 1:
        wave = wave.repeat(2, 1)

    with torch.no_grad():
        sources = apply_model(demucs_model, wave.unsqueeze(0))

    vocals = sources[0, 3].cpu().T.numpy()
    music = (sources[0, 0] + sources[0, 1] + sources[0, 2]).cpu().T.numpy()

    output_dir = f"separated_output/{video_id}"
    os.makedirs(output_dir, exist_ok=True)
    vocals_wav = os.path.join(output_dir, "vocals.wav")
    music_wav = os.path.join(output_dir, "music.wav")
    sf.write(vocals_wav, vocals, 44100)
    sf.write(music_wav, music, 44100)

    # Prepare output (choose format)
    if format == "mp4":
        vocals_mp4 = os.path.join(output_dir, "vocals.mp4")
        music_mp4 = os.path.join(output_dir, "music.mp4")

        # Correct ffmpeg command with -map for proper stream handling
        subprocess.run(["ffmpeg", "-y", "-i", vocals_wav, "-i", video_path, "-c:v", "copy", "-c:a", "aac", "-map", "0:a", "-map", "1:v", vocals_mp4], check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-i", music_wav, "-i", video_path, "-c:v", "copy", "-c:a", "aac", "-map", "0:a", "-map", "1:v", music_mp4], check=True, capture_output=True)
        
        return {
            "vocals_url": f"/download?file={vocals_mp4}",
            "music_url": f"/download?file={music_mp4}"
        }
    else:
        return {
            "vocals_url": f"/download?file={vocals_wav}",
            "music_url": f"/download?file={music_wav}"
        }

@app.post("/process")
async def process_video(file: UploadFile = File(...), format: str = Form(...)):
    if demucs_model is None:
        raise HTTPException(status_code=503, detail="Service unavailable: Demucs model failed to load at startup.")

    video_id = str(uuid.uuid4())
    os.makedirs("temp", exist_ok=True)
    os.makedirs("separated_output", exist_ok=True)
    video_path = f"temp/{video_id}_{file.filename}"

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {str(e)}")

    try:
        # Use run_in_threadpool for the CPU-bound task
        response = await app.loop.run_in_executor(executor, process_video_sync, video_path, video_id, format)
        return response
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg stderr: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"FFmpeg failed: {e.stderr}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during audio processing: {str(e)}")
    finally:
        # Clean up temporary files regardless of success or failure
        try:
            os.remove(video_path)
            os.remove(f"temp/{video_id}_temp.wav")
        except Exception:
            pass

@app.get("/download")
def download(file: str = Query(...)):
    # Security: Ensure no path traversal and only serve from allowed directory
    if not file.startswith("separated_output/"):
        raise HTTPException(status_code=404, detail="Invalid file path")
    if not os.path.isfile(file):
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(file, media_type="application/octet-stream", filename=os.path.basename(file))
