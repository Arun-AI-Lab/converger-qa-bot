# app/rag_pipeline.py
from typing import List, Dict, Optional
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from app.config import settings
from app.models import AnswerResponse, SourceReference
from app.utils import load_faiss_index


class RAGPipeline:
    """
    Simple RAG pipeline - retrieve documents and answer questions.
    No metadata guardrails, just pure document-based answering.
    """

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.vectorstore: Optional[FAISS] = load_faiss_index(settings.VECTOR_STORE_PATH)

    # -------------------------------------------------------------------------
    # 🔍 Retrieval
    # -------------------------------------------------------------------------
    def retrieve_docs(self, query: str, k: int = 12) -> List[Dict]:
        """Retrieve relevant chunks from FAISS."""
        if not self.vectorstore:
            return []
        docs = self.vectorstore.similarity_search(query, k=k)
        return [{"text": d.page_content, "metadata": d.metadata} for d in docs]

    def _unique_sources(self, docs: List[Dict]) -> List[SourceReference]:
        """Extract unique file references."""
        seen, sources = set(), []
        for d in docs:
            meta = d.get("metadata") or {}
            fn = meta.get("filename") or "unknown"
            if fn not in seen:
                seen.add(fn)
                sources.append(SourceReference(filename=fn))
        return sources

    # -------------------------------------------------------------------------
    # 🧠 Answer Generation - Simple & Direct
    # -------------------------------------------------------------------------
    def generate_answer(self, question: str, retrieved_docs: List[Dict]) -> str:
        """
        Generate answer from retrieved documents.
        Uses gpt-4o, no metadata guardrails.
        """
        context_chunks = "\n\n---\n\n".join(
            [d["text"] for d in retrieved_docs[:8]]
        ) if retrieved_docs else "No documents found."

        prompt = f"""You are a helpful assistant answering questions about contracts and documents.

Based on the documents provided below, answer the following question directly and accurately.
If the information is not in the documents, say so clearly.

DOCUMENTS:
{context_chunks}

QUESTION: {question}

ANSWER:"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer questions based on the provided documents. Be direct and accurate."
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1500,
        )

        return response.choices[0].message.content.strip()

    # -------------------------------------------------------------------------
    # 🚀 Main Entry
    # -------------------------------------------------------------------------
    def answer_question(self, question: str) -> AnswerResponse:
        """Main entrypoint - retrieve and answer."""
        if not self.vectorstore:
            return AnswerResponse(answer="No documents indexed. Please run embedding first.")

        retrieved_docs = self.retrieve_docs(question, k=12)
        answer_text = self.generate_answer(question, retrieved_docs)
        sources = self._unique_sources(retrieved_docs)

        return AnswerResponse(answer=answer_text, sources=sources)