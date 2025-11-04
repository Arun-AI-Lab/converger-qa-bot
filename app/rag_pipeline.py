# app/rag_pipeline.py
import json
from typing import List, Dict, Optional, Set
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from app.config import settings
from app.models import AnswerResponse, SummaryStats, SourceReference
from app.utils import load_faiss_index


class RAGPipeline:
    """
    Refined RAG pipeline — smarter metadata summarization and contextual reasoning.
    Accurately builds region-level breakdowns for better LLM factual grounding.
    """

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.vectorstore: Optional[FAISS] = load_faiss_index(settings.VECTOR_STORE_PATH)

    # -------------------------------------------------------------------------
    # 🔍 Retrieval + Utilities
    # -------------------------------------------------------------------------
    def retrieve_docs(self, query: str, k: int = 12) -> List[Dict]:
        """Retrieve relevant chunks from FAISS."""
        if not self.vectorstore:
            return []
        docs = self.vectorstore.similarity_search(query, k=k)
        return [{"text": d.page_content, "metadata": d.metadata} for d in docs]

    def _unique_sources(self, docs: List[Dict]) -> List[SourceReference]:
        """Extract unique file references for transparency."""
        seen, sources = set(), []
        for d in docs:
            meta = d.get("metadata") or {}
            fn = meta.get("filename") or "unknown"
            if fn not in seen:
                seen.add(fn)
                sources.append(SourceReference(filename=fn))
        return sources

    # -------------------------------------------------------------------------
    # 📊 Smarter Context Summarization
    # -------------------------------------------------------------------------
    def summarize_global_context(self) -> SummaryStats:
        """
        Aggregate global stats across FAISS metadata,
        now including nested region-contract breakdowns.
        """
        if not self.vectorstore or not hasattr(self.vectorstore, "docstore"):
            return SummaryStats()

        all_metas = [
            d.metadata for d in self.vectorstore.docstore._dict.values()
            if getattr(d, "metadata", None)
        ]
        filename_to_meta = {m.get("filename"): m for m in all_metas if m.get("filename")}

        total_files = len(filename_to_meta)
        total_contracts = len([
            m for m in filename_to_meta.values() if m.get("file_type") == "contract"
        ])
        total_general = len([
            m for m in filename_to_meta.values() if m.get("file_type") == "general"
        ])

        regions: Dict[str, int] = {}
        contract_types: Dict[str, int] = {}
        party_types: Dict[str, int] = {"Clients": 0, "Vendors": 0}

        # New nested breakdown: region → contract_type → count
        region_contract_summary: Dict[str, Dict[str, int]] = {}

        for meta in filename_to_meta.values():
            region = meta.get("region", "Australia")
            ctype = meta.get("contract_type", "Unknown")
            party = meta.get("party_type", "General")

            # Overall region and contract type counts
            regions[region] = regions.get(region, 0) + 1
            contract_types[ctype] = contract_types.get(ctype, 0) + 1
            if party.lower() == "client":
                party_types["Clients"] += 1
            elif party.lower() == "vendor":
                party_types["Vendors"] += 1

            # Region-wise contract breakdown
            if region not in region_contract_summary:
                region_contract_summary[region] = {}
            region_contract_summary[region][ctype] = (
                region_contract_summary[region].get(ctype, 0) + 1
            )

        return SummaryStats(
            total_files=total_files,
            total_contracts=total_contracts,
            total_general_files=total_general,
            contract_types=contract_types,
            regions=regions,
            party_types=party_types,
            region_contract_summary=region_contract_summary,  # ✅ Added
        )

    # -------------------------------------------------------------------------
    # 🧠 Core LLM Generation
    # -------------------------------------------------------------------------
    def generate_answer(
        self,
        question: str,
        retrieved_docs: List[Dict],
        summary_stats: Optional[SummaryStats] = None
    ) -> str:
        """Let GPT synthesize a factual, state-level aware answer."""

        context_chunks = "\n\n---\n\n".join(
            [d["text"] for d in retrieved_docs[:8]]
        ) if retrieved_docs else "No specific text found."

        stats_context = ""
        if summary_stats:
            stats_context = json.dumps({
                "files_total": summary_stats.total_files,
                "contracts_total": summary_stats.total_contracts,
                "general_total": summary_stats.total_general_files,
                "regions": summary_stats.regions,
                "contract_types": summary_stats.contract_types,
                "party_types": summary_stats.party_types,
                "region_contract_summary": summary_stats.region_contract_summary,
            }, indent=2)

        prompt = f"""
You are a factual and context-aware assistant for call center agents.
You analyze questions about contracts, regions, and document statistics.

Use the provided structured data to reason correctly — do not guess.
If a region has no contracts, explicitly mention it.
You can express summaries, insights, and comparisons freely, but always ground them in the data.

System Context (for reasoning, not verbatim output):
{stats_context}

Document Context (reference snippets):
{context_chunks}

Question:
{question}

Now generate your best, well-structured and insightful answer with bullet points or sections where useful.
"""

        response = self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a factual, analytical assistant that writes organized, "
                        "state-level contract summaries using provided metadata. "
                        "You may include short insights but no invented numbers. Use bullet points"
                    )
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=900,
        )

        return response.choices[0].message.content.strip()

    # -------------------------------------------------------------------------
    # 🚀 Main Entry
    # -------------------------------------------------------------------------
    def answer_question(self, question: str) -> AnswerResponse:
        """Main entrypoint — routes all questions through vector + metadata context."""
        if not self.vectorstore:
            return AnswerResponse(answer="⚠️ No indexed documents found. Please embed or reload first.")

        summary_stats = self.summarize_global_context()
        retrieved_docs = self.retrieve_docs(question, k=settings.SIMILARITY_TOP_K)
        answer_text = self.generate_answer(question, retrieved_docs, summary_stats)
        sources = self._unique_sources(retrieved_docs)

        return AnswerResponse(answer=answer_text, sources=sources)
