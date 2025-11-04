from fastapi import APIRouter
from app.models import QueryRequest
from app.rag_pipeline import RAGPipeline

router = APIRouter()

@router.post("/query")
async def query_contract(request: QueryRequest):
    """
    Simplified query endpoint for frontend integration.
    Returns only the formatted answer string for chat display.
    """
    pipeline = RAGPipeline()
    response = pipeline.answer_question(request.question)

    # ✅ Return only what frontend needs
    return {
        "answer": response.answer.strip()
    }
