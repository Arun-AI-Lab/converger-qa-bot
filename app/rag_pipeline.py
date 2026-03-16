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
    # 🧠 Question Type Detection
    # -------------------------------------------------------------------------
    def _is_summary_question(self, question: str) -> bool:
        """Detect if question is asking for summary/aggregate stats."""
        summary_keywords = [
            'how many', 'total', 'count', 'list', 'overview', 'summary',
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