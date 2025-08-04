from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Dict
import hashlib
import uuid
import requests

# Dummy in-memory user store
users: Dict[str, dict] = {}

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Models ----------------

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

# ---------------- Helpers ----------------

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# ---------------- Routes ----------------

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
        # Verify Google token using Google's tokeninfo endpoint
        response = requests.get(
            f"https://oauth2.googleapis.com/tokeninfo?id_token={req.token}"
        )
        data = response.json()
        if "error" in data:
            raise HTTPException(status_code=401, detail="Invalid Google token")

        email = data.get("email")
        name = data.get("name")
        sub = data.get("sub")  # Unique Google ID

        # Register or update user
        username = name.replace(" ", "_").lower() + "_" + sub[:6]
        if username not in users:
            users[username] = {
                "username": username,
                "email": email,
                "google_id": sub
            }
        return {"message": "Google login successful", "username": username}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#  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