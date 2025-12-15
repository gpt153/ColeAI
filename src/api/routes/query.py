from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.database import get_session
from src.models import Persona
from src.agents.persona_agent import query_persona
from pydantic import BaseModel

router = APIRouter()


class QueryRequest(BaseModel):
    persona_slug: str
    question: str


class QueryResponse(BaseModel):
    answer: str


@router.post("/", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    session: AsyncSession = Depends(get_session)
):
    """Query a persona with a question."""
    # Find persona
    stmt = select(Persona).where(Persona.slug == request.persona_slug)
    result = await session.execute(stmt)
    persona = result.scalar_one_or_none()

    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")

    # Query persona agent
    answer = await query_persona(str(persona.id), request.question, session)

    return QueryResponse(answer=answer)
