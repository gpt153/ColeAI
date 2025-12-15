from typing import List, Dict, Any
from src.config import settings
from openai import AsyncOpenAI
import re
import logging

logger = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=settings.openai_api_key)


def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """
    Split text into chunks using header-based or fixed-size chunking.
    Based on Cole Medin's intelligent chunking approach.

    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk (default from settings)
        chunk_overlap: Overlap between chunks (default from settings)

    Returns:
        List of text chunks
    """
    if settings.use_header_chunking:
        return chunk_by_headers(text, chunk_size or settings.chunk_size)
    else:
        return chunk_by_size(text, chunk_size or settings.chunk_size, chunk_overlap or settings.chunk_overlap)


def chunk_by_size(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Fixed-size chunking with overlap (original implementation).

    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return chunks


def chunk_by_headers(text: str, max_chunk_size: int) -> List[str]:
    """
    Intelligent chunking by markdown/HTML headers.
    Inspired by Cole Medin's header-based chunking approach.

    Args:
        text: Input text to chunk
        max_chunk_size: Maximum characters per chunk

    Returns:
        List of text chunks
    """
    # Split by markdown headers (# ## ### etc) or HTML headers (<h1> <h2> etc)
    header_pattern = r'(^#{1,6}\s+.+$|<h[1-6]>.*?</h[1-6]>)'
    parts = re.split(header_pattern, text, flags=re.MULTILINE)

    chunks = []
    current_chunk = ""
    current_header = ""

    for part in parts:
        if not part or not part.strip():
            continue

        # Check if this part is a header
        is_header = bool(re.match(header_pattern, part.strip()))

        if is_header:
            # Start a new chunk with this header
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_header = part.strip()
            current_chunk = current_header + "\n\n"
        else:
            # Add content to current chunk
            potential_chunk = current_chunk + part

            # If adding this would exceed max size, split it
            if len(potential_chunk) > max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append(current_chunk.strip())
                # Start new chunk with header
                current_chunk = current_header + "\n\n" + part
            else:
                current_chunk = potential_chunk

    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # If no headers found, fall back to size-based chunking
    if len(chunks) <= 1:
        return chunk_by_size(text, max_chunk_size, settings.chunk_overlap)

    return chunks


def extract_code_blocks(markdown_content: str, min_length: int = None) -> List[Dict[str, Any]]:
    """
    Extract code blocks from markdown content along with context.
    Based on Cole Medin's code extraction approach for Agentic RAG.

    Args:
        markdown_content: The markdown content to extract code blocks from
        min_length: Minimum length of code blocks to extract (default from settings)

    Returns:
        List of dictionaries containing code blocks and their context
    """
    min_length = min_length or settings.min_code_block_length
    code_blocks = []

    # Skip if content starts with triple backticks (edge case for files wrapped in backticks)
    content = markdown_content.strip()
    start_offset = 0
    if content.startswith('```'):
        start_offset = 3
        logger.debug("Skipping initial triple backticks")

    # Find all occurrences of triple backticks
    backtick_positions = []
    pos = start_offset
    while True:
        pos = markdown_content.find('```', pos)
        if pos == -1:
            break
        backtick_positions.append(pos)
        pos += 3

    # Process pairs of backticks
    i = 0
    while i < len(backtick_positions) - 1:
        start_pos = backtick_positions[i]
        end_pos = backtick_positions[i + 1]

        # Extract the content between backticks
        code_section = markdown_content[start_pos+3:end_pos]

        # Check if there's a language specifier on the first line
        lines = code_section.split('\n', 1)
        if len(lines) > 1:
            first_line = lines[0].strip()
            if first_line and ' ' not in first_line and len(first_line) < 20:
                language = first_line
                code_content = lines[1].strip() if len(lines) > 1 else ""
            else:
                language = ""
                code_content = code_section.strip()
        else:
            language = ""
            code_content = code_section.strip()

        # Skip if code block is too short
        if len(code_content) < min_length:
            i += 2
            continue

        # Extract context before (1000 chars)
        context_start = max(0, start_pos - 1000)
        context_before = markdown_content[context_start:start_pos].strip()

        # Extract context after (1000 chars)
        context_end = min(len(markdown_content), end_pos + 3 + 1000)
        context_after = markdown_content[end_pos + 3:context_end].strip()

        code_blocks.append({
            'code': code_content,
            'language': language,
            'context_before': context_before,
            'context_after': context_after,
            'full_context': f"{context_before}\n\n```{language}\n{code_content}\n```\n\n{context_after}"
        })

        # Move to next pair
        i += 2

    return code_blocks


async def generate_code_summary(code: str, context_before: str, context_after: str) -> str:
    """
    Generate a summary for a code example using its surrounding context.
    Based on Cole Medin's LLM-based code summarization.

    Args:
        code: The code example
        context_before: Context before the code
        context_after: Context after the code

    Returns:
        A summary of what the code example demonstrates
    """
    if not settings.use_code_extraction:
        return "Code example for demonstration purposes."

    prompt = f"""<context_before>
{context_before[-500:] if len(context_before) > 500 else context_before}
</context_before>

<code_example>
{code[:1500] if len(code) > 1500 else code}
</code_example>

<context_after>
{context_after[:500] if len(context_after) > 500 else context_after}
</context_after>

Based on the code example and its surrounding context, provide a concise summary (2-3 sentences) that describes what this code example demonstrates and its purpose. Focus on the practical application and key concepts illustrated."""

    try:
        response = await client.chat.completions.create(
            model=settings.context_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise code example summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Error generating code example summary: {e}")
        return "Code example for demonstration purposes."
