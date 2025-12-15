from openai import AsyncOpenAI
from typing import List, Tuple
from src.config import settings
import logging
import asyncio

logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=settings.openai_api_key)


async def generate_embeddings(texts: List[str], max_retries: int = 3) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using OpenAI with retry logic.

    Args:
        texts: List of text strings to embed
        max_retries: Maximum number of retry attempts

    Returns:
        List of embedding vectors (1536 dimensions)
    """
    if not texts:
        return []

    retry_delay = 1.0  # Start with 1 second delay

    for retry in range(max_retries):
        try:
            # Batch embed (OpenAI supports up to 2048 texts per request)
            response = await client.embeddings.create(
                model=settings.embedding_model,
                input=texts
            )

            embeddings = [item.embedding for item in response.data]

            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            if retry < max_retries - 1:
                logger.warning(f"Error generating batch embeddings (attempt {retry + 1}/{max_retries}): {e}")
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"Failed to generate batch embeddings after {max_retries} attempts: {e}")
                # Try creating embeddings one by one as fallback
                logger.info("Attempting to create embeddings individually...")
                embeddings = []
                successful_count = 0

                for i, text in enumerate(texts):
                    try:
                        individual_response = await client.embeddings.create(
                            model=settings.embedding_model,
                            input=[text]
                        )
                        embeddings.append(individual_response.data[0].embedding)
                        successful_count += 1
                    except Exception as individual_error:
                        logger.error(f"Failed to create embedding for text {i}: {individual_error}")
                        # Add zero embedding as fallback
                        embeddings.append([0.0] * settings.embedding_dimension)

                logger.info(f"Successfully created {successful_count}/{len(texts)} embeddings individually")
                return embeddings


async def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a single text."""
    embeddings = await generate_embeddings([text])
    return embeddings[0]


async def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.
    Based on Cole Medin's contextual embeddings approach.

    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for

    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    if not settings.use_contextual_embeddings:
        return chunk, False

    try:
        # Create the prompt for generating contextual information
        prompt = f"""<document>
{full_document[:25000]}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        # Call the OpenAI API to generate contextual information
        response = await client.chat.completions.create(
            model=settings.context_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise contextual information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )

        # Extract the generated context
        context = response.choices[0].message.content.strip()

        # Combine the context with the original chunk
        contextual_text = f"{context}\n---\n{chunk}"

        return contextual_text, True

    except Exception as e:
        logger.error(f"Error generating contextual embedding: {e}. Using original chunk instead.")
        return chunk, False
