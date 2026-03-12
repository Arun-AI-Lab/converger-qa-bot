# app/pre_embed_all_pdfs.py
import os
from app.utils import smart_chunk_text, extract_text_from_pdf, extract_text_from_docx
from app.metadata_extractor import generate_file_metadata
from app.config import settings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def embed_all_documents():
    """
    Preprocess all PDFs and DOCX files in data/uploads, create embeddings,
    and save to FAISS index.
    """

    upload_dir = settings.UPLOAD_DIR
    vectorstore_path = settings.VECTOR_STORE_PATH
    os.makedirs(vectorstore_path, exist_ok=True)

    # Find all supported files
    files = [
        f for f in os.listdir(upload_dir) 
        if f.lower().endswith(('.pdf', '.docx'))
    ]

    if not files:
        print("❌ No PDFs or DOCX files found in data/uploads/")
        return

    all_chunks = []
    all_metas = []

    print(f"\n📂 Found {len(files)} files\n")

    for fname in files:
        file_path = os.path.join(upload_dir, fname)
        print(f"📄 Processing {fname}...", end=" ")

        try:
            # Extract text based on file type
            if fname.lower().endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            elif fname.lower().endswith('.docx'):
                text = extract_text_from_docx(file_path)
            else:
                text = ""
            
            if not text.strip():
                print("⚠️  No text extracted")
                continue

            # Create chunks
            chunks = smart_chunk_text(text)
            metadata = generate_file_metadata(file_path, text)

            # Store chunks with metadata
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metas.append(metadata)

            print(f"✓ ({len(chunks)} chunks)")

        except Exception as e:
            print(f"✗ Error: {e}")

    if not all_chunks:
        print("\n❌ No valid chunks generated")
        return

    print(f"\n🧠 Creating FAISS index for {len(all_chunks)} chunks...")

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
    )

    # Create and save FAISS vector store with metadata
    vectorstore = FAISS.from_texts(all_chunks, embedding=embeddings, metadatas=all_metas)
    vectorstore.save_local(vectorstore_path)

    print(f"\n✅ Embedding complete!")
    print(f"📊 Files: {len(files)} | Chunks: {len(all_chunks)}")
    print(f"📁 Saved to: {vectorstore_path}")


if __name__ == "__main__":
    embed_all_documents()