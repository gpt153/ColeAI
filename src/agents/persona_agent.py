from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from src.vector_store.store import VectorStore
from src.config import settings
import logging

logger = logging.getLogger(__name__)


class PersonaDeps:
    """Dependencies for PersonaAgent."""

    def __init__(self, persona_id: str, session: AsyncSession):
        self.persona_id = persona_id
        self.session = session


# Create PersonaAgent with explicit API key via provider
provider = AnthropicProvider(api_key=settings.anthropic_api_key)
model = AnthropicModel(
    model_name=settings.pydantic_ai_model.replace("anthropic:", ""),
    provider=provider
)

persona_agent = Agent(
    model=model,
    deps_type=PersonaDeps,
    system_prompt="""You are a helpful assistant that answers questions based on a persona's knowledge base.

CRITICAL RULES:
1. ONLY use knowledge from the provided context (retrieved chunks)
2. ALWAYS cite your sources with [Source: title, URL]
3. If the answer is not in the context, say "I don't have information about that in this persona's knowledge base."
4. Match the persona's communication style based on the content patterns
5. Be concise but thorough

When answering:
- Start with the direct answer
- Provide supporting details from the context
- End with source citations
"""
)


@persona_agent.tool
async def search_knowledge_base(ctx: RunContext[PersonaDeps], query: str) -> str:
    """Search the persona's knowledge base for relevant information."""
    try:
        chunks = await VectorStore.similarity_search(
            session=ctx.deps.session,
            persona_id=ctx.deps.persona_id,
            query=query,
            k=5
        )

        if not chunks:
            return "No relevant information found in knowledge base."

        # Format context for LLM
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[{i}] Title: {chunk.title}\n"
                f"Content: {chunk.content_text[:500]}...\n"
                f"Source: {chunk.source_url}\n"
            )

        return "\n\n".join(context_parts)

    except Exception as e:
        logger.error(f"Knowledge base search failed: {e}", exc_info=True)
        return f"Error searching knowledge base: {str(e)}"


async def query_persona(
    persona_id: str,
    question: str,
    session: AsyncSession
) -> str:
    """
    Query a persona agent with a question.

    Args:
        persona_id: UUID of the persona
        question: User's question
        session: Database session

    Returns:
        Answer with citations
    """
    deps = PersonaDeps(persona_id=persona_id, session=session)
    result = await persona_agent.run(question, deps=deps)
    return result.data
