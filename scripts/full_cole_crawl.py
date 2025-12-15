"""Full crawl of Cole Medin's content - GitHub + YouTube with rate limiting."""
import asyncio
import logging
from sqlalchemy import select, delete, func
from src.database import AsyncSessionLocal, init_db
from src.models import Persona, ContentChunk
from src.crawlers.registry import get_crawler
from src.vector_store.store import VectorStore

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


async def full_crawl():
    """Complete crawl of Cole Medin's content with rate limiting."""
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
        logger.info("🚀 FULL COLE MEDIN CRAWL (WITH RATE LIMITING)")
        logger.info("="*80)
        logger.info(f"Persona: {persona.name} ({persona.id})\n")

        # Check existing content (don't delete!)
        logger.info("📊 Checking existing content...")
        stmt = select(func.count(ContentChunk.id)).where(ContentChunk.persona_id == persona.id)
        result = await session.execute(stmt)
        existing_count = result.scalar()

        stmt = select(ContentChunk.source_url).where(ContentChunk.persona_id == persona.id)
        result = await session.execute(stmt)
        existing_urls = {row[0] for row in result.all()}

        logger.info(f"✓ Found {existing_count} existing chunks")
        logger.info(f"✓ Will skip {len(existing_urls)} existing URLs\n")

        total_github = 0
        total_youtube = 0

        # 1. Crawl Dynamous Community Organization (10 repos)
        logger.info("=" * 80)
        logger.info("🏢 DYNAMOUS COMMUNITY ORGANIZATION")
        logger.info("=" * 80)
        logger.info("⏱️  Rate limit: 3 seconds between repos\n")

        github_crawler = get_crawler("github", str(persona.id))

        try:
            # Crawl all repos from dynamous-community org
            logger.info(f"📁 Crawling dynamous-community organization...")
            contents = await github_crawler.crawl(
                "dynamous-community",
                is_org=True,
                rate_limit_delay=3.0  # 3 seconds between repos
            )

            logger.info(f"  ✓ Found {len(contents)} README files")

            # Index all files (skip existing)
            new_count = 0
            skipped_count = 0
            for idx, content in enumerate(contents, 1):
                if content['source_url'] in existing_urls:
                    logger.info(f"  ⏭️  [{idx}/{len(contents)}] Skipping existing: {content['title'][:60]}")
                    skipped_count += 1
                    continue

                logger.info(f"  📄 [{idx}/{len(contents)}] {content['title'][:60]}")
                try:
                    await VectorStore.add_content(
                        session=session,
                        persona_id=str(persona.id),
                        source_type="github",
                        title=content['title'],
                        content_text=content['content_text'],
                        source_url=content['source_url']
                    )
                    new_count += 1
                    existing_urls.add(content['source_url'])  # Add to set to avoid duplicates within this run
                except Exception as e:
                    logger.error(f"    ❌ Failed to index: {e}")
                    continue

            total_github += new_count
            logger.info(f"  ✅ Indexed {new_count} new files from dynamous-community ({skipped_count} skipped)")

        except Exception as e:
            logger.error(f"  ❌ Organization crawl error: {e}")

        # 2. Crawl Cole's Personal Repos (3 repos)
        logger.info("\n" + "=" * 80)
        logger.info("👤 COLE'S PERSONAL REPOSITORIES")
        logger.info("=" * 80)
        logger.info("⏱️  Rate limit: 3 seconds between repos\n")

        cole_repos = [
            "ai-agents-masterclass",
            "context-engineering-intro",
            "mcp-crawl4ai-rag"
        ]

        try:
            logger.info(f"📁 Crawling coleam00 repositories...")
            contents = await github_crawler.crawl(
                "coleam00",
                repo_filter=cole_repos,
                rate_limit_delay=3.0  # 3 seconds between repos
            )

            logger.info(f"  ✓ Found {len(contents)} README files")

            # Index all files (skip existing)
            new_count = 0
            skipped_count = 0
            for idx, content in enumerate(contents, 1):
                if content['source_url'] in existing_urls:
                    logger.info(f"  ⏭️  [{idx}/{len(contents)}] Skipping existing: {content['title'][:60]}")
                    skipped_count += 1
                    continue

                logger.info(f"  📄 [{idx}/{len(contents)}] {content['title'][:60]}")
                try:
                    await VectorStore.add_content(
                        session=session,
                        persona_id=str(persona.id),
                        source_type="github",
                        title=content['title'],
                        content_text=content['content_text'],
                        source_url=content['source_url']
                    )
                    new_count += 1
                    existing_urls.add(content['source_url'])  # Add to set to avoid duplicates within this run
                except Exception as e:
                    logger.error(f"    ❌ Failed to index: {e}")
                    continue

            total_github += new_count
            logger.info(f"  ✅ Indexed {new_count} new files from coleam00 ({skipped_count} skipped)")

        except Exception as e:
            logger.error(f"  ❌ Personal repos error: {e}")

        logger.info(f"\n📊 GitHub total: {total_github} files")

        # 3. Crawl YouTube (ALL 205+ videos)
        logger.info("\n" + "=" * 80)
        logger.info("🎥 YOUTUBE CHANNEL")
        logger.info("=" * 80)
        logger.info("⏱️  Rate limit: 2 seconds between videos\n")

        try:
            youtube_crawler = get_crawler("youtube", str(persona.id))
            logger.info("Crawling @ColeMedin channel (ALL videos)...")

            youtube_contents = await youtube_crawler.crawl(
                "https://www.youtube.com/@ColeMedin",
                rate_limit_delay=2.0  # 2 seconds between videos
            )

            logger.info(f"\n✓ Extracted {len(youtube_contents)} video transcripts")
            logger.info(f"⏳ Indexing into vector store...\n")

            new_count = 0
            skipped_count = 0
            for idx, content in enumerate(youtube_contents, 1):
                if content['source_url'] in existing_urls:
                    logger.info(f"  ⏭️  [{idx}/{len(youtube_contents)}] Skipping existing: {content['title'][:60]}")
                    skipped_count += 1
                    continue

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
                    existing_urls.add(content['source_url'])  # Add to set to avoid duplicates within this run
                except Exception as e:
                    # Log but continue - some videos may have transcripts too large for embeddings
                    logger.error(f"    ❌ Failed to index: {e}")
                    continue

            total_youtube = new_count
            logger.info(f"\n📊 YouTube total: {new_count} videos indexed ({skipped_count} skipped)")

        except Exception as e:
            logger.error(f"❌ YouTube error: {e}")
            import traceback
            traceback.print_exc()

        # Final stats
        logger.info("\n" + "=" * 80)
        logger.info("✅ CRAWL COMPLETE")
        logger.info("=" * 80)

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
