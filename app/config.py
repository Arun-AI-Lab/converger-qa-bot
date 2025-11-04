# app/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file explicitly
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

class Settings:
    # App
    APP_NAME: str = "Contract RAG Q&A Bot"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Paths
    VECTOR_STORE_PATH: str = os.getenv("VECTORSTORE_PATH", "vector_store")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")

    # RAG Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    SIMILARITY_TOP_K: int = int(os.getenv("SIMILARITY_TOP_K", "4"))

    # Future-proof
    LIVE_DB_ENABLED: bool = os.getenv("LIVE_DB_ENABLED", "False").lower() == "true"

# Create folders
os.makedirs(Settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(Settings.VECTOR_STORE_PATH, exist_ok=True)

# Global instance
settings = Settings()

# Validate OpenAI key
if not settings.OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing in .env file!")