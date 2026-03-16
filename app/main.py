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
# ⚙️ Startup Event - Directory Checks & Metadata Aggregation
# ============================================================

@app.on_event("startup")
def startup_event():
    """Ensure all necessary directories exist and cache metadata stats on startup."""
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)
    print(f"✅ {settings.APP_NAME} started successfully.")
    print(f"📂 Upload Directory: {settings.UPLOAD_DIR}")
    print(f"💾 Vector Store Path: {settings.VECTOR_STORE_PATH}")
    
    # Pre-aggregate metadata for faster summary queries
    try:
        from app.utils import aggregate_metadata_from_faiss
        agg = aggregate_metadata_from_faiss(settings.VECTOR_STORE_PATH)
        if agg['total_files'] > 0:
            print(f"📊 Indexed Files: {agg['total_files']} | Contracts: {agg['total_contracts']} | Regions: {len(agg['regions'])}")
        else:
            print(f"⚠️  No documents indexed yet.")
    except Exception as e:
        print(f"⚠️  Could not pre-aggregate metadata: {e}")

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