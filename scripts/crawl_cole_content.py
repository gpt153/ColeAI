"""Crawl Cole Medin's content for existing persona."""
import asyncio
import logging
from sqlalchemy import select
from src.database import AsyncSessionLocal, init_db
from src.models import Persona
from src.crawlers.registry import get_crawler
from src.vector_store.store import VectorStore

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


async def crawl_cole_content():
    """Crawl Cole Medin's content."""
    await init_db()

    async with AsyncSessionLocal() as session:
        # Get existing persona
        stmt = select(Persona).where(Persona.slug == "cole-medin")
        result = await session.execute(stmt)
        persona = result.scalar_one_or_none()

        if not persona:
            logger.error("Cole Medin persona not found!")
            return

        logger.info(f"Found persona: {persona.name} ({persona.id})")

        # Crawl GitHub repositories
        logger.info("\n🔍 Crawling GitHub repositories...")
        github_crawler = get_crawler("github", str(persona.id))

        repos = [
            "ai-agents-masterclass",
            "context-engineering-intro",
            "mcp-crawl4ai-rag"
        ]

        for repo in repos:
            logger.info(f"\n📦 Crawling {repo}...")
            try:
                contents = await github_crawler.crawl(
                    "coleam00",
                    repo_filter=[repo]
                )
                logger.info(f"  ✓ Found {len(contents)} files")

                # Index content
                for idx, content in enumerate(contents[:3], 1):  # First 3 files only for speed
                    logger.info(f"  📄 Indexing {idx}/{min(3, len(contents))}: {content['title']}")
                    await VectorStore.add_content(
                        session=session,
                        persona_id=str(persona.id),
                        source_type="github",
                        title=content['title'],
                        content_text=content['content_text'],
                        source_url=content['source_url']
                    )
                logger.info(f"  ✓ Indexed {min(3, len(contents))} files from {repo}")

            except Exception as e:
                logger.error(f"  ❌ Error crawling {repo}: {e}")

        logger.info("\n✅ Crawling complete!")

        # Show stats
        from sqlalchemy import func
        from src.models import ContentChunk
        stmt = select(func.count(ContentChunk.id)).where(ContentChunk.persona_id == persona.id)
        result = await session.execute(stmt)
        count = result.scalar()
        logger.info(f"📊 Total chunks for Cole Medin: {count}")


if __name__ == "__main__":
    asyncio.run(crawl_cole_content())
