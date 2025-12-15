"""
Test script to compare different RAG strategies.
Based on Cole Medin's best practices implementation.
"""
import asyncio
import logging
import time
from sqlalchemy import select
from src.database import AsyncSessionLocal, init_db
from src.models import Persona
from src.agents.persona_agent import query_persona
from src.config import settings

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


async def test_rag_strategies():
    """Test different RAG strategy configurations."""

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
        question = "Show me how to make async API calls with the Anthropic SDK"

        logger.info(f"\n{'='*80}")
        logger.info(f"TEST QUESTION: {question}")
        logger.info(f"{'='*80}\n")

        # Display current configuration
        logger.info(f"📊 Current RAG Configuration:")
        logger.info(f"  - Contextual Embeddings: {settings.use_contextual_embeddings}")
        logger.info(f"  - Reranking: {settings.use_reranking}")
        logger.info(f"  - Code Extraction: {settings.use_code_extraction}")
        logger.info(f"  - Header Chunking: {settings.use_header_chunking}")
        logger.info(f"  - Context Model: {settings.context_model}")
        logger.info(f"  - Reranking Model: {settings.reranking_model}\n")

        # Test query with current configuration
        logger.info("🚀 Querying with current RAG configuration...")
        start_time = time.time()

        try:
            response = await query_persona(
                persona_id=str(persona.id),
                question=question,
                session=session
            )

            elapsed_time = time.time() - start_time

            logger.info(f"\n{'='*80}")
            logger.info(f"✅ RESPONSE (took {elapsed_time:.2f}s):")
            logger.info(f"{'='*80}")
            logger.info(response)
            logger.info(f"{'='*80}\n")

        except Exception as e:
            logger.error(f"❌ Error during query: {e}", exc_info=True)

        # Suggestions for testing different configurations
        logger.info("\n" + "="*80)
        logger.info("💡 To test different RAG strategies, update your .env file:")
        logger.info("="*80)
        logger.info("""
Strategy Recommendations:

1️⃣  BASIC RAG (fastest, good for development):
   USE_CONTEXTUAL_EMBEDDINGS=false
   USE_RERANKING=false
   USE_CODE_EXTRACTION=false
   USE_HEADER_CHUNKING=false

2️⃣  ENHANCED RAG (recommended for production):
   USE_CONTEXTUAL_EMBEDDINGS=false
   USE_RERANKING=true          # Biggest impact on quality
   USE_CODE_EXTRACTION=true
   USE_HEADER_CHUNKING=true

3️⃣  MAXIMUM QUALITY RAG (slowest, best results):
   USE_CONTEXTUAL_EMBEDDINGS=true  # Requires reindexing!
   USE_RERANKING=true
   USE_CODE_EXTRACTION=true
   USE_HEADER_CHUNKING=true

⚠️  NOTE: Contextual embeddings and header chunking require rebuilding
   the knowledge base (re-running crawl and indexing).

💰 Cost Considerations:
   - Contextual Embeddings: +1 LLM call per chunk during indexing
   - Reranking: No additional API costs (local model)
   - Code Extraction: +1 LLM call per code block during indexing
        """)


if __name__ == "__main__":
    asyncio.run(test_rag_strategies())
