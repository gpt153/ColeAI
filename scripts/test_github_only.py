import asyncio
import logging
from sqlalchemy import select
from src.database import AsyncSessionLocal, init_db
from src.models import Persona, ContentChunk
from src.crawlers.registry import get_crawler
from src.vector_store.chunking import chunk_text
from src.vector_store.store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_github_crawl():
    """Test persona system with GitHub only (skip YouTube)."""

    # Initialize database
    await init_db()

    async with AsyncSessionLocal() as session:
        # 1. Check if test persona exists, create if not
        logger.info("Checking for existing test persona...")
        stmt = select(Persona).where(Persona.slug == "test-dev")
        result = await session.execute(stmt)
        persona = result.scalar_one_or_none()

        if persona:
            logger.info(f"Using existing persona: {persona.id}")
        else:
            logger.info("Creating test persona...")
            persona = Persona(
                name="Test AI Developer",
                slug="test-dev",
                description="Test persona for validating the system",
                sources={
                    "github": "anthropics"
                },
                expertise=["AI", "Python", "Testing"]
            )
            session.add(persona)
            await session.commit()
            await session.refresh(persona)

            logger.info(f"Created persona: {persona.id}")

        # 2. Check if we already have content chunks
        stmt = select(ContentChunk).where(ContentChunk.persona_id == persona.id).limit(1)
        result = await session.execute(stmt)
        existing_chunk = result.scalar_one_or_none()

        if existing_chunk:
            logger.info("Content already exists, skipping crawl and storage.")
        else:
            # 2. Crawl GitHub (using a public repo we know exists)
            logger.info("Crawling GitHub repositories...")
            github_crawler = get_crawler("github", str(persona.id))
            github_contents = await github_crawler.crawl(
                "anthropics",
                repo_filter=["anthropic-sdk-python"]  # Public repo we know exists
            )

            logger.info(f"Found {len(github_contents)} GitHub items")

            # 3. Chunk and store content
            for content in github_contents[:1]:  # Limit to first item for testing
                logger.info(f"Processing: {content['title']}")

                # Chunk text
                chunks = chunk_text(content['content_text'])

                # Store each chunk with embedding
                for i, chunk in enumerate(chunks):
                    logger.info(f"  Storing chunk {i+1}/{len(chunks)}...")
                    await VectorStore.add_content(
                        session=session,
                        persona_id=str(persona.id),
                        source_type="github",
                        title=f"{content['title']} (part {i+1})",
                        content_text=chunk,
                        source_url=content['source_url'],
                        metadata=content.get('metadata', {})
                    )

            logger.info("Storage complete!")

        # 4. Test querying
        logger.info("Testing vector search...")
        results = await VectorStore.similarity_search(
            session=session,
            persona_id=str(persona.id),
            query="API client",
            k=3
        )

        logger.info(f"Found {len(results)} similar chunks")
        for i, result in enumerate(results):
            logger.info(f"  {i+1}. {result.title}: {result.content_text[:100]}...")


if __name__ == "__main__":
    asyncio.run(test_github_crawl())
