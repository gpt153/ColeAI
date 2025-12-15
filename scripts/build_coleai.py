import asyncio
import logging
from sqlalchemy import select
from src.database import AsyncSessionLocal, init_db
from src.models import Persona
from src.crawlers.registry import get_crawler
from src.vector_store.chunking import chunk_text
from src.vector_store.store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def build_coleai():
    """Build ColeAI persona from Cole Medin's public content."""

    # Initialize database
    await init_db()

    async with AsyncSessionLocal() as session:
        # 1. Create persona
        logger.info("Creating ColeAI persona...")
        persona = Persona(
            name="Cole Medin",
            slug="cole-medin",
            description="AI educator, founder of Dynamous AI, expert in agentic coding and context engineering",
            sources={
                "youtube": "@coleam00",
                "github": "coleam00"
            },
            expertise=["AI Agents", "Context Engineering", "Agentic Coding", "PydanticAI"]
        )
        session.add(persona)
        await session.commit()
        await session.refresh(persona)

        logger.info(f"Created persona: {persona.id}")

        # 2. Crawl YouTube
        logger.info("Crawling YouTube channel...")
        youtube_crawler = get_crawler("youtube", str(persona.id))
        youtube_contents = await youtube_crawler.crawl(
            "https://www.youtube.com/@coleam00",
            max_videos=10  # Limit for MVP testing
        )

        logger.info(f"Found {len(youtube_contents)} YouTube videos")

        # 3. Crawl GitHub
        logger.info("Crawling GitHub repositories...")
        github_crawler = get_crawler("github", str(persona.id))
        github_contents = await github_crawler.crawl(
            "coleam00",
            repo_filter=["ai-agents-masterclass", "context-engineering-intro"]
        )

        logger.info(f"Found {len(github_contents)} GitHub items")

        # 4. Chunk and store all content
        all_contents = youtube_contents + github_contents

        for content in all_contents:
            logger.info(f"Processing: {content['title']}")

            # Chunk text
            chunks = chunk_text(content['content_text'])

            # Store each chunk with embedding
            for i, chunk in enumerate(chunks):
                await VectorStore.add_content(
                    session=session,
                    persona_id=str(persona.id),
                    source_type=content.get('source_url', '').split('/')[2].split('.')[0],  # Extract domain
                    title=f"{content['title']} (part {i+1})",
                    content_text=chunk,
                    source_url=content['source_url'],
                    metadata=content.get('metadata', {})
                )

        logger.info("ColeAI persona build complete!")
        logger.info(f"Total content chunks stored: {len(all_contents) * 5}")  # Approx estimate


if __name__ == "__main__":
    asyncio.run(build_coleai())
