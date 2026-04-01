# app/rag_pipeline.py
from typing import List, Dict, Optional
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from app.config import settings
from app.models import AnswerResponse, SourceReference, SummaryStats
from app.utils import load_faiss_index, aggregate_metadata_from_faiss


class RAGPipeline:
    """
    Smart RAG pipeline - detects question type and routes appropriately.
    - Summary questions → aggregated metadata
    - Regular questions → chunk-based retrieval
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
        """Retrieve relevant chunks from FAISS, filtered by relevance score."""
        if not self.vectorstore:
            return []
        docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
        # Keep only chunks with relevance score >= 0.5
        filtered = [(doc, score) for doc, score in docs_and_scores if score >= 0.5]
        # Fall back to top-3 if nothing passes the threshold
        if not filtered:
            filtered = docs_and_scores[:3]
        return [{"text": d.page_content, "metadata": d.metadata, "score": score} for d, score in filtered]

    def _unique_sources(self, docs: List[Dict]) -> List[SourceReference]:
        """Extract unique file references with metadata and excerpt."""
        seen, sources = set(), []
        for d in docs:
            meta = d.get("metadata") or {}
            fn = meta.get("filename") or "unknown"
            if fn not in seen:
                seen.add(fn)
                raw_text = d.get("text", "")
                snippet = raw_text[:150].rsplit(" ", 1)[0] + "..." if len(raw_text) > 150 else raw_text
                sources.append(SourceReference(
                    filename=fn,
                    excerpt=snippet or None,
                ))
        return sources

    # -------------------------------------------------------------------------
    # 🧠 Question Type Detection
    # -------------------------------------------------------------------------
    def _is_summary_question(self, question: str) -> bool:
        """Detect if question is asking for summary/aggregate stats."""
        summary_keywords = [
            'how many', 'total', 'count', 'overview',
            'all contracts', 'all files', 'regions', 'contract types',
            'which contracts', 'how many contracts', 'statistics', 'stats',
            'breakdown', 'distribution', 'across regions', 'by type'
        ]
        q_lower = question.lower()
        return any(kw in q_lower for kw in summary_keywords)

    # -------------------------------------------------------------------------
    # 📊 Summary Answer Generation
    # -------------------------------------------------------------------------
    def get_summary_stats(self) -> SummaryStats:
        """Get aggregated metadata stats from FAISS index."""
        agg_data = aggregate_metadata_from_faiss(settings.VECTOR_STORE_PATH)
        return SummaryStats(
            total_files=agg_data['total_files'],
            total_contracts=agg_data['total_contracts'],
            total_general_files=agg_data['total_general_files'],
            contract_types=agg_data['contract_types'],
            regions=agg_data['regions'],
            party_types=agg_data['party_types'],
            region_contract_summary=agg_data['region_contract_summary']
        )

    def generate_summary_answer(self, question: str, stats: SummaryStats) -> str:
        """Generate answer using aggregated metadata (no chunk retrieval)."""
        stats_text = f"""
System Statistics:
- Total Files: {stats.total_files}
- Total Contracts: {stats.total_contracts}
- Total General Files: {stats.total_general_files}
- Contract Types: {stats.contract_types}
- Regions: {stats.regions}
- Party Types: {stats.party_types}
- Region-Contract Summary: {stats.region_contract_summary}
"""
        
        prompt = f"""You are a helpful assistant answering questions about contracts and documents in the system.

Based on the system statistics provided below, answer the following question directly and accurately.
Provide clear, well-formatted responses with specific numbers and breakdowns where relevant.

SYSTEM STATISTICS:
{stats_text}

QUESTION: {question}

ANSWER:"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer questions about contract system statistics based on the provided data. Be direct, accurate, and provide clear breakdowns. Format numbers clearly."
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1500,
        )

        return response.choices[0].message.content.strip()

    # -------------------------------------------------------------------------
    # 💬 Regular Answer Generation
    # -------------------------------------------------------------------------
    def generate_answer(self, question: str, retrieved_docs: List[Dict]) -> str:
        """
        Generate answer from retrieved documents.
        Uses gpt-4o, no metadata guardrails.
        """
        context_parts = []
        for d in retrieved_docs:
            source = d.get("metadata", {}).get("filename", "unknown")
            context_parts.append(f"[Source: {source}]\n{d['text']}")
        context_chunks = "\n\n---\n\n".join(context_parts) if context_parts else "No documents found."

        prompt = f"""You are a helpful assistant answering questions about contracts and legal documents.

You have been given relevant document excerpts. Compile everything relevant and present a well-structured, descriptive answer.

Guidelines:
- Write in a professional, readable style — not just a raw bullet list. Add a brief intro sentence that frames what was found.
- When listing items (clauses, conditions, etc.), give each item a short description so the answer is informative.
- Group related points together where it makes sense.
- Never open with phrases like "the documents do not...", "the excerpts provided...", or any hedging language.
- Never mention chunks, excerpts, or the retrieval process.
- Only say information is unavailable if none of the documents contain anything relevant to the question.

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
            temperature=0.1,
            max_tokens=1500,
        )

        return response.choices[0].message.content.strip()

    # -------------------------------------------------------------------------
    # 🚀 Main Entry - Smart Routing
    # -------------------------------------------------------------------------
    def answer_question(self, question: str) -> AnswerResponse:
        """Main entrypoint - detect question type and route appropriately."""
        if not self.vectorstore:
            return AnswerResponse(answer="No documents indexed. Please run embedding first.")

        # Check if it's a summary question
        if self._is_summary_question(question):
            stats = self.get_summary_stats()
            answer_text = self.generate_summary_answer(question, stats)
            return AnswerResponse(answer=answer_text, summary_stats=stats)
        
        # Otherwise use normal RAG retrieval
        retrieved_docs = self.retrieve_docs(question, k=12)
        answer_text = self.generate_answer(question, retrieved_docs)
        sources = self._unique_sources(retrieved_docs)

        return AnswerResponse(answer=answer_text, sources=sources)