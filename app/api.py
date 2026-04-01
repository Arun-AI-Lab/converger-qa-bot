from fastapi import APIRouter
from app.models import QueryRequest
from app.rag_pipeline import RAGPipeline

router = APIRouter()

@router.post("/query")
async def query_contract(request: QueryRequest):
    """
    Query endpoint for frontend integration.
    Returns answer with sources appended as formatted text.
    """
    pipeline = RAGPipeline()
    response = pipeline.answer_question(request.question)

    answer = response.answer.strip()

    if response.sources:
        source_lines = []
        for s in response.sources:
            line = f"- {s.filename}"
            if s.excerpt:
                line += f'\n  "{s.excerpt}"'
            source_lines.append(line)
        answer += "\n\n---\n**Sources:**\n" + "\n".join(source_lines)

    return {"answer": answer}
