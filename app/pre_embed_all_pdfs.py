# app/pre_embed_all_pdfs.py
import os
import asyncio
from app.utils import extract_text_from_pdf, smart_chunk_text
from app.metadata_extractor import generate_file_metadata
from app.config import settings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


async def embed_all_pdfs():
    """
    Preprocess all PDFs in data/uploads, create embeddings with metadata,
    and save to a FAISS index in data/vector_store.
    """

    upload_dir = settings.UPLOAD_DIR
    vectorstore_path = settings.VECTOR_STORE_PATH
    os.makedirs(vectorstore_path, exist_ok=True)

    pdf_files = [f for f in os.listdir(upload_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("❌ No PDFs found in data/uploads/. Please add some files first.")
        return

    all_chunks = []
    all_metas = []

    print(f"📂 Found {len(pdf_files)} PDF files. Beginning processing...\n")

    for fname in pdf_files:
        file_path = os.path.join(upload_dir, fname)
        print(f"📄 Processing {fname}...")

        try:
            text = extract_text_from_pdf(file_path)
            if not text.strip():
                print(f"⚠️  Skipped {fname}: no readable text found.")
                continue

            chunks = smart_chunk_text(text)
            metadata = generate_file_metadata(file_path)

            # Store each chunk with file-level metadata
            for c in chunks:
                all_chunks.append(c)
                all_metas.append(metadata)

        except Exception as e:
            print(f"⚠️  Error processing {fname}: {e}")

    if not all_chunks:
        print("❌ No valid text chunks generated. Aborting embedding.")
        return

    print(f"\n🧠 Creating FAISS index for {len(all_chunks)} chunks across {len(pdf_files)} files...")

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
    )

    # Create and save FAISS vector store with full metadata
    vectorstore = FAISS.from_texts(all_chunks, embedding=embeddings, metadatas=all_metas)
    vectorstore.save_local(vectorstore_path)

    print(f"✅ Embedding complete! FAISS index saved to: {vectorstore_path}")
    print(f"📈 Total Chunks Embedded: {len(all_chunks)}")
    print(f"📦 Total Files Processed: {len(pdf_files)}")


if __name__ == "__main__":
    asyncio.run(embed_all_pdfs())
