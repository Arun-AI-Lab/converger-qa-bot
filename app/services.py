# app/services.py
import os
import shutil
from datetime import datetime
from fastapi import UploadFile
from app.models import UploadResponse, AnswerResponse
from app.metadata_extractor import generate_file_metadata
from app.utils import extract_text_from_pdf, smart_chunk_text, create_faiss_index, load_faiss_index
from app.rag_pipeline import RAGPipeline
from app.config import settings


# ============================================================
# 📥 Handle File Uploads
# ============================================================

async def process_uploaded_pdf(file: UploadFile) -> UploadResponse:
    """
    Process an uploaded file:
      - Save it locally
      - Extract text
      - Chunk
      - Embed + update FAISS
      - Store metadata
    """
    # Save uploaded file
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text
    text = extract_text_from_pdf(file_path)
    if not text.strip():
        raise ValueError("Uploaded file appears to be empty or unreadable.")

    # Create simple file metadata
    metadata = generate_file_metadata(file_path)

    # Chunk and embed
    chunks = smart_chunk_text(text)
    create_faiss_index(chunks, metadata)

    return UploadResponse(
        message="✅ File processed and embedded successfully.",
        metadata=metadata,
    )


# ============================================================
# 💬 Handle Questions (Query)
# ============================================================

async def answer_question(question: str) -> AnswerResponse:
    """
    Passes the question to RAG pipeline for answering.
    Handles both general and contract-specific queries.
    """
    # Ensure vectorstore exists
    vectorstore = load_faiss_index(settings.VECTOR_STORE_PATH)
    if not vectorstore:
        return AnswerResponse(answer="❌ No documents indexed. Please upload files first.")

    pipeline = RAGPipeline()
    response = pipeline.answer_question(question)
    return response
