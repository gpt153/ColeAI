"""Full crawl of Cole Medin's content - GitHub + YouTube."""
import asyncio
import logging
from sqlalchemy import select, delete
from src.database import AsyncSessionLocal, init_db
from src.models import Persona, ContentChunk
from src.crawlers.registry import get_crawler
from src.vector_store.store import VectorStore

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


async def full_crawl():
    """Complete crawl of Cole Medin's content."""
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
        logger.info("🚀 FULL COLE MEDIN CRAWL")
        logger.info("="*80)
        logger.info(f"Persona: {persona.name} ({persona.id})\n")

        # Clear existing content
        logger.info("🗑️  Clearing existing content...")
        delete_stmt = delete(ContentChunk).where(ContentChunk.persona_id == persona.id)
        result = await session.execute(delete_stmt)
        await session.commit()
        logger.info(f"✓ Deleted {result.rowcount} existing chunks\n")

        # 1. Crawl GitHub repos (ALL files, not just READMEs)
        logger.info("=" * 80)
        logger.info("📦 GITHUB REPOSITORIES")
        logger.info("=" * 80)

        github_crawler = get_crawler("github", str(persona.id))

        repos = [
            "ai-agents-masterclass",
            "context-engineering-intro",
            "mcp-crawl4ai-rag"
        ]

        total_github = 0
        for repo in repos:
            logger.info(f"\n📁 Crawling {repo}...")
            try:
                contents = await github_crawler.crawl(
                    "coleam00",
                    repo_filter=[repo]
                )
                logger.info(f"  ✓ Found {len(contents)} files")

                # Index ALL files (not just first 3)
                for idx, content in enumerate(contents, 1):
                    logger.info(f"  📄 [{idx}/{len(contents)}] {content['title'][:60]}")
                    await VectorStore.add_content(
                        session=session,
                        persona_id=str(persona.id),
                        source_type="github",
                        title=content['title'],
                        content_text=content['content_text'],
                        source_url=content['source_url']
                    )

                total_github += len(contents)
                logger.info(f"  ✅ Indexed {len(contents)} files from {repo}")

            except Exception as e:
                logger.error(f"  ❌ Error: {e}")

        logger.info(f"\n📊 GitHub total: {total_github} files")

        # 2. Crawl YouTube (if available)
        logger.info("\n" + "=" * 80)
        logger.info("🎥 YOUTUBE CHANNEL")
        logger.info("=" * 80)

        try:
            youtube_crawler = get_crawler("youtube", str(persona.id))
            logger.info("Crawling @coleam00 channel...")

            youtube_contents = await youtube_crawler.crawl(
                "https://www.youtube.com/@coleam00",
                max_videos=5  # Limit to 5 for speed
            )

            logger.info(f"✓ Found {len(youtube_contents)} videos")

            for idx, content in enumerate(youtube_contents, 1):
                logger.info(f"  🎬 [{idx}/{len(youtube_contents)}] {content['title'][:60]}")
                await VectorStore.add_content(
                    session=session,
                    persona_id=str(persona.id),
                    source_type="youtube",
                    title=content['title'],
                    content_text=content['content_text'],
                    source_url=content['source_url']
                )

            logger.info(f"📊 YouTube total: {len(youtube_contents)} videos")

        except Exception as e:
            logger.error(f"❌ YouTube error: {e}")
            logger.info("Skipping YouTube (crawler may not be fully implemented)")

        # Final stats
        logger.info("\n" + "=" * 80)
        logger.info("✅ CRAWL COMPLETE")
        logger.info("=" * 80)

        from sqlalchemy import func
        stmt = select(func.count(ContentChunk.id)).where(ContentChunk.persona_id == persona.id)
        result = await session.execute(stmt)
        total = result.scalar()

        stmt = select(ContentChunk.source_type, func.count(ContentChunk.id)).where(
            ContentChunk.persona_id == persona.id
        ).group_by(ContentChunk.source_type)
        result = await session.execute(stmt)
        breakdown = dict(result.all())

        logger.info(f"\n📊 Final Statistics:")
        logger.info(f"  - Total chunks: {total}")
        for source, count in breakdown.items():
            logger.info(f"  - {source}: {count}")
        logger.info("")


if __name__ == "__main__":
    asyncio.run(full_crawl())
