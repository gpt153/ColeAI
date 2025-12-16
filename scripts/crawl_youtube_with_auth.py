"""Crawl YouTube with cookie authentication to bypass bot detection."""
import asyncio
import logging
from sqlalchemy import select
from src.database import AsyncSessionLocal, init_db
from src.models import Persona
from src.crawlers.registry import get_crawler
from src.vector_store.store import VectorStore

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


async def crawl_youtube_authenticated():
    """Crawl YouTube with browser cookies for authentication."""
    await init_db()

    async with AsyncSessionLocal() as session:
        # Get persona
        stmt = select(Persona).where(Persona.slug == "cole-medin")
        result = await session.execute(stmt)
        persona = result.scalar_one_or_none()

        if not persona:
            logger.error("❌ Cole Medin persona not found!")
            return

        logger.info("="*80)
        logger.info("🎥 YOUTUBE CRAWL WITH AUTHENTICATION")
        logger.info("="*80)
        logger.info(f"Persona: {persona.name} ({persona.id})\n")

        # Check existing YouTube content
        logger.info("📊 Checking existing YouTube content...")
        from sqlalchemy import func
        stmt = select(func.count()).where(
            Persona.id == persona.id,
            # Assuming we have a way to filter by source_type
        )

        try:
            youtube_crawler = get_crawler("youtube", str(persona.id))
            logger.info("Crawling @ColeMedin channel with authentication...")
            logger.info("⚠️  Using cookies from Chrome browser")
            logger.info("   Make sure you're logged into YouTube in Chrome!\n")

            youtube_contents = await youtube_crawler.crawl(
                "https://www.youtube.com/@ColeMedin",
                rate_limit_delay=3.0,  # Longer delay with auth
                cookies_from_browser="chrome"  # Use Chrome cookies
            )

            logger.info(f"\n✓ Extracted {len(youtube_contents)} video transcripts")
            logger.info(f"⏳ Indexing into vector store...\n")

            new_count = 0
            failed_count = 0
            for idx, content in enumerate(youtube_contents, 1):
                logger.info(f"  🎬 [{idx}/{len(youtube_contents)}] {content['title'][:60]}")
                try:
                    await VectorStore.add_content(
                        session=session,
                        persona_id=str(persona.id),
                        source_type="youtube",
                        title=content['title'],
                        content_text=content['content_text'],
                        source_url=content['source_url']
                    )
                    new_count += 1
                except Exception as e:
                    logger.error(f"    ❌ Failed to index: {e}")
                    failed_count += 1
                    continue

            logger.info(f"\n📊 YouTube crawl complete:")
            logger.info(f"  - Successfully indexed: {new_count}")
            logger.info(f"  - Failed: {failed_count}")

        except Exception as e:
            logger.error(f"❌ YouTube error: {e}")
            import traceback
            traceback.print_exc()

        # Final stats
        logger.info("\n" + "=" * 80)
        logger.info("✅ YOUTUBE CRAWL COMPLETE")
        logger.info("=" * 80)

        from sqlalchemy import func
        stmt = select(func.count()).select_from(Persona).where(Persona.id == persona.id)
        result = await session.execute(stmt)

        logger.info(f"\nYou can now query Cole Medin's YouTube knowledge via MCP!")


if __name__ == "__main__":
    asyncio.run(crawl_youtube_authenticated())
