import asyncio
import logging
from sqlalchemy import select
from src.database import AsyncSessionLocal, init_db
from src.models import Persona
from src.agents.persona_agent import query_persona

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_persona_agent():
    """Test the PydanticAI persona agent with RAG."""

    # Initialize database
    await init_db()

    async with AsyncSessionLocal() as session:
        # Get test persona
        logger.info("Loading test persona...")
        stmt = select(Persona).where(Persona.slug == "test-dev")
        result = await session.execute(stmt)
        persona = result.scalar_one_or_none()

        if not persona:
            logger.error("Test persona not found. Run test_github_only.py first.")
            return

        logger.info(f"Using persona: {persona.name} ({persona.id})")

        # Test query
        question = "How do I use the Anthropic Python SDK to make async requests?"

        logger.info(f"\n{'='*60}")
        logger.info(f"QUESTION: {question}")
        logger.info(f"{'='*60}\n")

        # Query the agent
        response = await query_persona(
            persona_id=str(persona.id),
            question=question,
            session=session
        )

        logger.info(f"\n{'='*60}")
        logger.info(f"AGENT RESPONSE:")
        logger.info(f"{'='*60}")
        logger.info(response)
        logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(test_persona_agent())
