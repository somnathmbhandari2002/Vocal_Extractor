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
    return hashlib.sha256(password.encode()).hexdigest()

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
@app.post("/process")
async def process_video(file: UploadFile = File(...), format: str = Form(...)):
    video_id = str(uuid.uuid4())
    os.makedirs("temp", exist_ok=True)
    os.makedirs("separated_output", exist_ok=True)
    video_path = f"temp/{video_id}_{file.filename}"

    # Save uploaded video
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {str(e)}")

    # Extract audio
    temp_audio_path = f"temp/{video_id}_temp.wav"
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "44100", "-ac", "2", "-vn",
        "-loglevel", "error", temp_audio_path
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg stderr: {e.stderr}")
        os.remove(video_path)
        raise HTTPException(status_code=500, detail="ffmpeg failed to extract audio.")

    try:
        # Demucs model
        model = get_model(name='htdemucs')
        model.eval()
    
        wave, sr = torchaudio.load(temp_audio_path)
        if sr != 44100:
            wave = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100)(wave)
        if wave.shape[0] == 1:
            wave = wave.repeat(2, 1)

        with torch.no_grad():
            sources = apply_model(model, wave.unsqueeze(0))

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
    
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg muxing stderr: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"FFmpeg failed to create final video: {e.stderr}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during audio processing: {str(e)}")
    finally:
        # Clean up temporary files regardless of success or failure
        try:
            os.remove(video_path)
            os.remove(temp_audio_path)
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

# ------------------- Entry Point for Deployment -------------------
# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.environ.get("PORT", 8000))  # use $PORT from environment if available
#     uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)


