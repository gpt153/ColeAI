from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.database import get_session
from sqlalchemy import text

router = APIRouter()


@router.get("/")
async def health():
    """Basic health check."""
    return {"status": "ok"}


@router.get("/db")
async def health_db(session: AsyncSession = Depends(get_session)):
    """Database health check."""
    try:
        await session.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        return {"status": "error", "database": "disconnected", "error": str(e)}
