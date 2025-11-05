# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import router as api_router
from app.config import settings
import os

# ============================================================
# 🌐 App Initialization
# ============================================================

app = FastAPI(
    title=settings.APP_NAME,
    description="A scalable RAG-based Q&A assistant for call center agents.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================
# 🛡️ CORS Configuration (allow for frontend / testing)
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 📦 Router Registration
# ============================================================

app.include_router(api_router, prefix="/api")

# ============================================================
# ⚙️ Startup Event - Directory Checks
# ============================================================

@app.on_event("startup")
def startup_event():
    """Ensure all necessary directories exist on startup."""
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)
    print(f"✅ {settings.APP_NAME} started successfully.")
    print(f"📂 Upload Directory: {settings.UPLOAD_DIR}")
    print(f"💾 Vector Store Path: {settings.VECTOR_STORE_PATH}")

# ============================================================
# 🏠 Root Route
# ============================================================

@app.get("/")
def root():
    return {
        "message": f"Welcome to {settings.APP_NAME}!",
        "status": "running",
        "docs": "/docs",
        "api_base": "/api"
    }
