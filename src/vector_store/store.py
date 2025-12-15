from typing import List, Dict, Any
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pgvector.sqlalchemy import Vector
from src.models import ContentChunk
from src.vector_store.embeddings import generate_embedding
from src.config import settings
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store operations using pgvector."""

    @staticmethod
    async def add_content(
        session: AsyncSession,
        persona_id: str,
        source_type: str,
        title: str,
        content_text: str,
        source_url: str,
        metadata: Dict[str, Any] = None
    ) -> ContentChunk:
        """Add content chunk with embedding to vector store."""
        try:
            # Generate embedding
            embedding = await generate_embedding(content_text)

            # Create content chunk
            chunk = ContentChunk(
                persona_id=persona_id,
                source_type=source_type,
                source_url=source_url,
                title=title,
                content_text=content_text,
                embedding=embedding,
                metadata=metadata or {}
            )

            session.add(chunk)
            await session.commit()
            await session.refresh(chunk)

            logger.info(f"Added content chunk: {chunk.id}")
            return chunk

        except Exception as e:
            logger.error(f"Failed to add content chunk: {e}", exc_info=True)
            await session.rollback()
            raise

    @staticmethod
    async def similarity_search(
        session: AsyncSession,
        persona_id: str,
        query: str,
        k: int = 5,
        source_filter: str = None
    ) -> List[ContentChunk]:
        """
        Search for similar content using vector similarity with optional reranking.
        Enhanced with Cole Medin's best practices: reranking and source filtering.

        Args:
            session: Database session
            persona_id: Filter by persona ID
            query: Search query text
            k: Number of results to return
            source_filter: Optional source URL/domain filter

        Returns:
            List of most similar content chunks (reranked if enabled)
        """
        try:
            # Generate query embedding
            query_embedding = await generate_embedding(query)

            # Build query with optional source filter
            stmt = (
                select(ContentChunk)
                .where(ContentChunk.persona_id == persona_id)
            )

            # Apply source filter if provided
            if source_filter:
                stmt = stmt.where(ContentChunk.source_url.ilike(f"%{source_filter}%"))

            # Order by cosine distance and limit
            # Fetch more results if reranking is enabled
            fetch_limit = k * 3 if settings.use_reranking else k
            stmt = (
                stmt
                .order_by(ContentChunk.embedding.cosine_distance(query_embedding))
                .limit(fetch_limit)
            )

            result = await session.execute(stmt)
            chunks = list(result.scalars().all())

            logger.info(f"Found {len(chunks)} similar chunks for query")

            # Apply reranking if enabled
            if settings.use_reranking and chunks:
                from src.vector_store.reranking import rerank_results

                # Convert chunks to dict format for reranking
                results_dict = [
                    {
                        'content_text': chunk.content_text,
                        'chunk': chunk
                    }
                    for chunk in chunks
                ]

                # Rerank and extract top k
                reranked = rerank_results(query, results_dict, top_k=k)
                chunks = [r['chunk'] for r in reranked]
                logger.info(f"Reranked to top {len(chunks)} results")

            return chunks[:k]  # Ensure we return exactly k results

        except Exception as e:
            logger.error(f"Similarity search failed: {e}", exc_info=True)
            raise
