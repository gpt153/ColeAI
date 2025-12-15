from typing import List, Dict, Any
from src.config import settings
import logging

logger = logging.getLogger(__name__)

# Lazy load CrossEncoder only if reranking is enabled
_reranker = None


def get_reranker():
    """Lazy load CrossEncoder model."""
    global _reranker
    if _reranker is None and settings.use_reranking:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(settings.reranking_model)
            logger.info(f"Loaded reranking model: {settings.reranking_model}")
        except Exception as e:
            logger.error(f"Failed to load reranking model: {e}")
            _reranker = None
    return _reranker


def rerank_results(query: str, results: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
    """
    Rerank search results using a CrossEncoder model.
    Based on Cole Medin's reranking approach for improved retrieval quality.

    Args:
        query: The search query
        results: List of search results with 'content_text' field
        top_k: Number of top results to return (default: return all reranked)

    Returns:
        Reranked list of results
    """
    if not settings.use_reranking or not results:
        return results

    reranker = get_reranker()
    if reranker is None:
        logger.warning("Reranker not available, returning original results")
        return results

    try:
        # Prepare query-document pairs for the cross-encoder
        pairs = []
        for result in results:
            content = result.get('content_text', result.get('content', ''))
            pairs.append([query, content])

        # Get cross-encoder scores
        scores = reranker.predict(pairs)

        # Add scores to results and sort
        for i, result in enumerate(results):
            result['rerank_score'] = float(scores[i])

        # Sort by rerank score (descending)
        reranked_results = sorted(results, key=lambda x: x.get('rerank_score', 0), reverse=True)

        # Return top_k if specified
        if top_k:
            reranked_results = reranked_results[:top_k]

        logger.info(f"Reranked {len(results)} results, returning top {len(reranked_results)}")
        return reranked_results

    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        return results
