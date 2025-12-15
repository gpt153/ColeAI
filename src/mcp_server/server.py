from fastmcp import FastMCP
from typing import List, Dict, Any
from src.database import AsyncSessionLocal
from src.agents.persona_agent import query_persona
from src.models import Persona
from sqlalchemy import select
import logging

logger = logging.getLogger(__name__)

# Create MCP server
mcp = FastMCP("persona-knowledge")


@mcp.tool()
async def query_persona_knowledge(persona_slug: str, question: str) -> str:
    """
    Query a persona's knowledge base with a question.

    Args:
        persona_slug: Persona identifier (e.g., "cole-medin")
        question: Question to ask the persona

    Returns:
        Answer with source citations
    """
    async with AsyncSessionLocal() as session:
        # Find persona by slug
        stmt = select(Persona).where(Persona.slug == persona_slug)
        result = await session.execute(stmt)
        persona = result.scalar_one_or_none()

        if not persona:
            return f"Persona '{persona_slug}' not found. Available personas: {await list_available_personas()}"

        # Query persona agent
        answer = await query_persona(str(persona.id), question, session)
        return answer


@mcp.tool()
async def list_available_personas() -> List[str]:
    """
    List all available persona agents.

    Returns:
        List of persona slugs
    """
    async with AsyncSessionLocal() as session:
        stmt = select(Persona.slug, Persona.name)
        result = await session.execute(stmt)
        personas = result.all()

        return [f"{slug} ({name})" for slug, name in personas]


if __name__ == "__main__":
    # Run MCP server
    mcp.run()
