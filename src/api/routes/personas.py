from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.database import get_session
from src.models import Persona
from typing import List
from pydantic import BaseModel, field_serializer
from uuid import UUID

router = APIRouter()


class PersonaResponse(BaseModel):
    id: UUID
    name: str
    slug: str
    description: str | None
    sources: dict
    expertise: list

    @field_serializer('id')
    def serialize_id(self, value: UUID) -> str:
        return str(value)

    class Config:
        from_attributes = True


@router.get("/", response_model=List[PersonaResponse])
async def list_personas(session: AsyncSession = Depends(get_session)):
    """List all personas."""
    stmt = select(Persona)
    result = await session.execute(stmt)
    personas = result.scalars().all()
    return personas


@router.get("/{slug}", response_model=PersonaResponse)
async def get_persona(slug: str, session: AsyncSession = Depends(get_session)):
    """Get persona by slug."""
    stmt = select(Persona).where(Persona.slug == slug)
    result = await session.execute(stmt)
    persona = result.scalar_one_or_none()

    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")

    return persona
