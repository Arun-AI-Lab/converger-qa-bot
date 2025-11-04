# app/utils.py
import os
import pickle
from typing import List
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import numpy as np

from app.config import settings


# ============================================================
# 📄 PDF Text Extraction
# ============================================================

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file using PyPDF2.
    Returns concatenated string of all pages.
    """
    text = ""
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text.strip()


# ============================================================
# ✂️ Smart Chunking (Recursive Splitter)
# ============================================================

def smart_chunk_text(text: str) -> List[str]:
    """
    Splits text into semantically-aware overlapping chunks.
    Uses RecursiveCharacterTextSplitter from LangChain.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return splitter.split_text(text)


# ============================================================
# 🧠 Embedding & Vectorstore (FAISS)
# ============================================================

def create_faiss_index(chunks: List[str], metadata: dict = None) -> FAISS:
    """
    Create a FAISS vector store from text chunks and metadata.
    Saves it to local path (settings.VECTOR_STORE_PATH).
    """
    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
    )
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings, metadatas=[metadata or {}] * len(chunks))
    save_faiss_index(vectorstore, settings.VECTOR_STORE_PATH)
    return vectorstore


def save_faiss_index(vectorstore: FAISS, path: str):
    """Saves FAISS index to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vectorstore.save_local(path)


def load_faiss_index(path: str) -> FAISS:
    """Loads FAISS index from disk. Returns None if not found."""
    if not os.path.exists(path):
        return None
    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
    )
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


# ============================================================
# 📊 Utility Helpers
# ============================================================

def count_unique_clients(metadata_list: List[dict]) -> int:
    """Counts unique client names from a list of metadata dicts."""
    names = {m.get("client_name", "").strip().lower() for m in metadata_list if m.get("client_name")}
    return len(names)


def count_unique_regions(metadata_list: List[dict]) -> int:
    """Counts unique regions across all contracts."""
    regions = {m.get("region", "").strip().lower() for m in metadata_list if m.get("region")}
    return len(regions)
