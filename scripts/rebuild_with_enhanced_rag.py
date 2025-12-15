"""
Rebuild knowledge base with enhanced RAG strategies.
- Header-based chunking
- Code block extraction with LLM summaries
- Reranking enabled
"""
import asyncio
import logging
from sqlalchemy import select, delete
from src.database import AsyncSessionLocal, init_db
from src.models import Persona, ContentChunk
from src.crawlers.registry import get_crawler
from src.vector_store.chunking import chunk_text, extract_code_blocks, generate_code_summary
from src.vector_store.embeddings import generate_embeddings
from src.vector_store.store import VectorStore
from src.config import settings

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


async def rebuild_knowledge_base():
    """Rebuild the knowledge base with enhanced RAG strategies."""

    logger.info("="*80)
    logger.info("🚀 REBUILDING KNOWLEDGE BASE WITH ENHANCED RAG STRATEGIES")
    logger.info("="*80)
    logger.info("")

    # Display configuration
    logger.info("📊 Current RAG Configuration:")
    logger.info(f"  - Header Chunking: {settings.use_header_chunking}")
    logger.info(f"  - Code Extraction: {settings.use_code_extraction}")
    logger.info(f"  - Reranking: {settings.use_reranking}")
    logger.info(f"  - Contextual Embeddings: {settings.use_contextual_embeddings}")
    logger.info("")

    # Initialize database
    await init_db()

    async with AsyncSessionLocal() as session:
        # 1. Get test persona
        logger.info("📋 Step 1: Loading test persona...")
        stmt = select(Persona).where(Persona.slug == "test-dev")
        result = await session.execute(stmt)
        persona = result.scalar_one_or_none()

        if not persona:
            logger.error("❌ Test persona not found. Run test_github_only.py first.")
            return

        logger.info(f"✓ Found persona: {persona.name} ({persona.id})")
        logger.info("")

        # 2. Clear existing content chunks
        logger.info("🗑️  Step 2: Clearing existing content chunks...")
        delete_stmt = delete(ContentChunk).where(ContentChunk.persona_id == persona.id)
        result = await session.execute(delete_stmt)
        await session.commit()
        logger.info(f"✓ Deleted {result.rowcount} existing chunks")
        logger.info("")

        # 3. Crawl GitHub repository
        logger.info("🔍 Step 3: Crawling GitHub repository...")
        github_crawler = get_crawler("github", str(persona.id))
        github_contents = await github_crawler.crawl(
            "anthropics",
            repo_filter=["anthropic-sdk-python"]
        )
        logger.info(f"✓ Found {len(github_contents)} items from GitHub")
        logger.info("")

        # 4. Process content with enhanced strategies
        logger.info("⚙️  Step 4: Processing content with enhanced strategies...")

        total_chunks = 0
        total_code_blocks = 0

        for idx, content in enumerate(github_contents[:1], 1):  # Process first item for testing
            logger.info(f"\n📄 Processing item {idx}/{len(github_contents[:1])}: {content['title']}")

            full_text = content['content_text']

            # 4a. Chunk with header-based strategy
            if settings.use_header_chunking:
                logger.info("  📑 Chunking by headers...")
                chunks = chunk_text(full_text)
                logger.info(f"  ✓ Created {len(chunks)} header-based chunks")
            else:
                logger.info("  📑 Chunking by size...")
                chunks = chunk_text(full_text)
                logger.info(f"  ✓ Created {len(chunks)} size-based chunks")

            # 4b. Store regular chunks with embeddings
            logger.info("  💾 Storing chunks with embeddings...")
            for i, chunk in enumerate(chunks):
                await VectorStore.add_content(
                    session=session,
                    persona_id=str(persona.id),
                    source_type="github",
                    title=f"{content['title']} (chunk {i+1})",
                    content_text=chunk,
                    source_url=content['source_url'],
                    metadata={
                        "chunk_type": "header_based" if settings.use_header_chunking else "size_based",
                        "chunk_index": i
                    }
                )

            total_chunks += len(chunks)
            logger.info(f"  ✓ Stored {len(chunks)} chunks")

            # 4c. Extract and index code blocks if enabled
            if settings.use_code_extraction:
                logger.info("  💻 Extracting code blocks...")
                code_blocks = extract_code_blocks(full_text)
                logger.info(f"  ✓ Found {len(code_blocks)} code blocks (≥{settings.min_code_block_length} chars)")

                if code_blocks:
                    logger.info("  🤖 Generating code summaries with LLM...")
                    for j, code_block in enumerate(code_blocks):
                        # Generate summary
                        summary = await generate_code_summary(
                            code_block['code'],
                            code_block['context_before'],
                            code_block['context_after']
                        )

                        # Store code block with summary
                        code_content = f"Code Summary: {summary}\n\n{code_block['full_context']}"

                        await VectorStore.add_content(
                            session=session,
                            persona_id=str(persona.id),
                            source_type="github",
                            title=f"{content['title']} - Code Example {j+1}",
                            content_text=code_content,
                            source_url=content['source_url'],
                            metadata={
                                "chunk_type": "code_block",
                                "language": code_block['language'],
                                "summary": summary
                            }
                        )

                        logger.info(f"    ✓ Code block {j+1}: {summary[:60]}...")

                    total_code_blocks += len(code_blocks)
                    logger.info(f"  ✓ Stored {len(code_blocks)} code blocks with summaries")

        logger.info("")
        logger.info("="*80)
        logger.info("✅ REBUILD COMPLETE!")
        logger.info("="*80)
        logger.info(f"📊 Statistics:")
        logger.info(f"  - Total text chunks: {total_chunks}")
        logger.info(f"  - Total code blocks: {total_code_blocks}")
        logger.info(f"  - Total indexed items: {total_chunks + total_code_blocks}")
        logger.info(f"  - Chunking strategy: {'Header-based' if settings.use_header_chunking else 'Size-based'}")
        logger.info(f"  - Code extraction: {'Enabled' if settings.use_code_extraction else 'Disabled'}")
        logger.info("")
        logger.info("🎯 Ready to test enhanced RAG with:")
        logger.info("   poetry run python scripts/test_rag_strategies.py")
        logger.info("")


if __name__ == "__main__":
    asyncio.run(rebuild_knowledge_base())
