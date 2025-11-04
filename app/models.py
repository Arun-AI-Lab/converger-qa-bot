# app/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

# ============================================================
# 📊 System Summary & Metadata Models
# ============================================================

class SummaryStats(BaseModel):
    """System-level summary across all embedded PDFs."""
    total_files: Optional[int] = 0
    total_contracts: Optional[int] = 0
    total_general_files: Optional[int] = 0
    contract_types: Optional[Dict[str, int]] = None
    regions: Optional[Dict[str, int]] = None
    party_types: Optional[Dict[str, int]] = None
    region_contract_summary: Optional[Dict[str, Dict[str, int]]] = None  # ✅ Added

# ============================================================
# 📄 Source & Response Models
# ============================================================

class SourceReference(BaseModel):
    filename: str
    page_numbers: Optional[List[int]] = None


class AnswerResponse(BaseModel):
    """Response returned to frontend after answering a question."""
    answer: str = Field(..., description="LLM-formatted answer for the frontend UI.")
    sources: Optional[List[SourceReference]] = None
    summary_stats: Optional[SummaryStats] = None


# ============================================================
# 🧾 Query Model
# ============================================================

class QueryRequest(BaseModel):
    """Schema for incoming questions."""
    question: str = Field(..., description="The user or agent’s question about uploaded contracts.")
