from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = Field(..., env="DATABASE_URL")

    # OpenAI (for embeddings)
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")

    # Anthropic (for PydanticAI agent)
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")

    # PydanticAI
    pydantic_ai_model: str = Field(default="anthropic:claude-3-5-sonnet-20241022", env="PYDANTIC_AI_MODEL")

    # GitHub
    github_token: str = Field(..., env="GITHUB_TOKEN")

    # Server ports
    mcp_server_port: int = Field(default=8100, env="MCP_SERVER_PORT")
    api_port: int = Field(default=8000, env="API_PORT")
    api_host: str = Field(default="0.0.0.0", env="API_HOST")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Embedding settings
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=1536, env="EMBEDDING_DIMENSION")

    # Chunking settings
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")

    # RAG Strategy Toggles (inspired by Cole Medin's best practices)
    use_contextual_embeddings: bool = Field(default=False, env="USE_CONTEXTUAL_EMBEDDINGS")
    use_reranking: bool = Field(default=False, env="USE_RERANKING")
    use_code_extraction: bool = Field(default=False, env="USE_CODE_EXTRACTION")
    use_header_chunking: bool = Field(default=False, env="USE_HEADER_CHUNKING")

    # LLM for contextual embeddings and code summaries
    context_model: str = Field(default="gpt-4o-mini", env="CONTEXT_MODEL")

    # Reranking model
    reranking_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", env="RERANKING_MODEL")

    # Code extraction settings
    min_code_block_length: int = Field(default=300, env="MIN_CODE_BLOCK_LENGTH")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
