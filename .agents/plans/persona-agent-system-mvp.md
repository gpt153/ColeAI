# Feature: Persona Agent System - Multi-Persona Knowledge Bank with ColeAI MVP

**Archon Project ID**: `ac6c0e62-4022-4f7d-ab15-9e551c50be56`
**GitHub Repository**: https://github.com/yourusername/persona-agent-system (update with actual repo URL)
**Plan Created**: 2025-12-14
**Status**: Ready for Implementation

---

The following plan should be complete, but it's important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils, types and models. Import from the right files etc.

## Feature Description

Build a scalable, automation-friendly system that creates AI agents from any person's public content. The system crawls public sources (YouTube, GitHub, Twitter, etc.), stores content in a vector database with RAG capabilities, and exposes dual interfaces (MCP server for Claude Code, HTTP API for Telegram/other platforms).

The first persona will be **ColeAI** - an agent trained on Cole Medin's public content (YouTube AI Agents Masterclass, GitHub repos). The architecture must support easily adding new personas (e.g., MikeAI for Mike Israetel) through a reusable pipeline.

## User Story

As a **learner struggling to remember expert content**
I want to **chat with an AI agent that knows everything an expert has publicly said**
So that **I can get personalized help and retrieve specific teachings on demand**

## Problem Statement

Learning from experts through YouTube, GitHub, articles, and podcasts creates information overload. Key insights are scattered across hundreds of hours of content, making it difficult to:
- Remember specific advice or patterns
- Find relevant examples when needed
- Distill recurring themes and principles
- Apply expert knowledge to current problems

Traditional note-taking is manual and incomplete. Existing RAG systems require building each agent from scratch.

## Solution Statement

Create a **Persona Agent Factory** - a generic pipeline that:
1. **Discovers** all public content sources for a person automatically
2. **Crawls** and extracts content (transcripts, code, articles)
3. **Processes** content into searchable chunks with embeddings
4. **Distills** wisdom - recurring themes, principles, methodologies
5. **Builds** queryable AI agent with persona-specific knowledge
6. **Exposes** via MCP (Claude Code) and HTTP API (Telegram, custom apps)

The system is designed for **repeatability** - after building ColeAI manually, we extract the pattern into a `/build-persona <name>` command that automates persona creation.

## Feature Metadata

**Feature Type**: New Capability (Greenfield Project)
**Estimated Complexity**: High
**Primary Systems Affected**: New standalone Python service
**Dependencies**:
- PydanticAI (^0.1.0) - Agent framework
- PostgreSQL 14+ with pgvector extension - Vector database
- FastAPI (^0.115.0) - HTTP server
- FastMCP (^0.5.0) - MCP server framework
- yt-dlp (^2025.1.0) - YouTube transcript extraction
- SQLAlchemy (^2.0) + asyncpg - Database ORM
- OpenAI Python SDK (^1.0) or sentence-transformers - Embeddings

---

## CONTEXT REFERENCES

### Relevant Codebase Files

**Note**: This is a new project - no existing files to reference. However, we'll mirror patterns from the remote-coding-agent for integration.

**Integration Reference** (from `/home/samuel/remote-coding-agent/`):
- `src/types/index.ts` (lines 59-84) - Why: `IPlatformAdapter` interface pattern we'll expose via HTTP API
- `src/adapters/telegram.ts` (lines 1-50) - Why: How Telegram consumes AI agents via streaming
- `src/index.ts` (lines 380-410) - Why: Pattern for registering platform adapters

### New Files to Create

**Project Root:**
- `pyproject.toml` - Python dependencies and project metadata (Poetry or uv)
- `docker-compose.yml` - PostgreSQL + pgvector + app containers
- `.env.example` - Environment variable template
- `README.md` - Setup and usage instructions
- `.gitignore` - Python-specific ignores

**Source Code** (`src/`):
- `src/config.py` - Settings and environment variable loading (Pydantic BaseSettings)
- `src/database.py` - SQLAlchemy engine, session management, connection pooling
- `src/models.py` - SQLAlchemy ORM models (Persona, ContentChunk, WisdomInsight, CrawlJob)

**Crawlers** (`src/crawlers/`):
- `src/crawlers/base.py` - Abstract BaseCrawler interface
- `src/crawlers/youtube.py` - YouTube transcript crawler (yt-dlp)
- `src/crawlers/github.py` - GitHub repo crawler (GitHub API + git clone)
- `src/crawlers/registry.py` - Crawler factory/registry pattern

**Agents** (`src/agents/`):
- `src/agents/persona_builder.py` - PydanticAI agent for building new personas
- `src/agents/persona_agent.py` - PydanticAI agent for querying persona knowledge
- `src/agents/wisdom_extractor.py` - LLM-based wisdom distillation

**Vector Store** (`src/vector_store/`):
- `src/vector_store/embeddings.py` - Embedding generation (OpenAI or local)
- `src/vector_store/store.py` - pgvector CRUD operations with similarity search
- `src/vector_store/chunking.py` - Text chunking strategies (RecursiveCharacterTextSplitter)

**MCP Server** (`src/mcp_server/`):
- `src/mcp_server/server.py` - FastMCP server with persona tools
- `src/mcp_server/tools.py` - Tool definitions (build_persona, query_persona, list_personas)

**HTTP API** (`src/api/`):
- `src/api/app.py` - FastAPI application setup
- `src/api/routes/personas.py` - Persona CRUD endpoints
- `src/api/routes/query.py` - Query persona endpoint (streaming support)
- `src/api/routes/health.py` - Health check endpoints

**Schemas** (`src/schemas/`):
- `src/schemas/persona.py` - Pydantic models for API requests/responses
- `src/schemas/crawl.py` - Crawl job schemas

**Tests** (`tests/`):
- `tests/test_crawlers.py` - Crawler unit tests
- `tests/test_agents.py` - Agent unit tests
- `tests/test_api.py` - API integration tests
- `tests/test_vector_store.py` - Vector store tests

**Scripts** (`scripts/`):
- `scripts/init_db.py` - Database initialization and migration
- `scripts/build_coleai.py` - One-off script to build ColeAI persona
- `scripts/seed_personas.py` - Seed example persona configurations

**Data/Config** (`config/`):
- `config/personas.yaml` - Persona definitions (name, sources, expertise)

**Database Migrations** (`migrations/`):
- `migrations/001_init_schema.sql` - Initial schema with pgvector

### Relevant Documentation (YOU SHOULD READ THESE BEFORE IMPLEMENTING!)

**PydanticAI Framework:**
- [Official Documentation](https://ai.pydantic.dev/) - Overview and concepts
- [Agents Guide](https://ai.pydantic.dev/agents/) - Building agents with tools
- [GitHub Repository](https://github.com/pydantic/pydantic-ai) - Examples and source code
- Why: Core framework for PersonaBuilder and PersonaAgent

**PostgreSQL pgvector:**
- [pgvector GitHub](https://github.com/pgvector/pgvector) - Extension documentation
- [RAG with PostgreSQL Tutorial](https://medium.com/@ahmed.mimilahlou/quickly-implement-your-first-rag-solution-using-python-and-pgvector-postgresql-a8ee82d70f8a) - Complete RAG implementation
- [pgvector Python Usage](https://www.tigerdata.com/learn/using-pgvector-with-python) - SQLAlchemy integration
- Why: Vector database for semantic search and RAG

**MCP Server:**
- [Python SDK Documentation](https://github.com/modelcontextprotocol/python-sdk) - Official SDK
- [FastMCP Framework](https://github.com/jlowin/fastmcp) - High-level MCP server framework
- [Building MCP Servers](https://modelcontextprotocol.io/quickstart/server) - Quickstart guide
- Why: Expose persona agents as tools for Claude Code

**YouTube Transcript Extraction:**
- [yt-dlp Documentation](https://github.com/yt-dlp/yt-dlp) - Main tool for downloading
- [yt-dlp-transcript Package](https://pypi.org/project/yt-dlp-transcript/) - Wrapper for transcripts
- Why: Extract Cole Medin's YouTube AI Agents Masterclass content

**FastAPI:**
- [FastAPI Documentation](https://fastapi.tiangolo.com/) - HTTP API framework
- [Async Database Patterns](https://fastapi.tiangolo.com/advanced/async-sql-databases/) - SQLAlchemy async
- Why: HTTP API for Telegram and custom integrations

**GitHub API:**
- [PyGithub Documentation](https://pygithub.readthedocs.io/) - GitHub API client
- Why: Discover and clone Cole Medin's repositories

### Patterns to Follow

**Naming Conventions:**
```python
# Classes: PascalCase
class PersonaBuilder:
    pass

# Functions/methods: snake_case
async def build_persona_pipeline(name: str) -> Persona:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_CHUNK_SIZE = 1000

# Private methods: _prefixed_snake_case
def _process_content(self, text: str) -> list[str]:
    pass
```

**Error Handling:**
```python
# Use custom exceptions
class PersonaNotFoundError(Exception):
    """Raised when persona does not exist in database."""
    pass

# Log and re-raise
try:
    await crawl_youtube(channel_id)
except Exception as e:
    logger.error(f"YouTube crawl failed: {e}", exc_info=True)
    raise CrawlerError(f"Failed to crawl YouTube: {e}") from e
```

**Logging Pattern:**
```python
import logging

logger = logging.getLogger(__name__)

# Structured logging
logger.info("Building persona", extra={
    "persona_name": "ColeAI",
    "sources": ["youtube", "github"],
    "stage": "discovery"
})
```

**Async/Await Consistency:**
```python
# All I/O operations are async
async def fetch_data(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

# Database operations
async with AsyncSession(engine) as session:
    result = await session.execute(select(Persona))
    personas = result.scalars().all()
```

**Type Hints:**
```python
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

# All function signatures have complete type hints
async def search_knowledge_base(
    persona_id: str,
    query: str,
    k: int = 5
) -> List[Dict[str, Any]]:
    pass
```

**Pydantic Models:**
```python
from pydantic import BaseModel, Field

class PersonaCreate(BaseModel):
    """Schema for creating a new persona."""
    name: str = Field(..., min_length=1, max_length=100)
    sources: Dict[str, str] = Field(default_factory=dict)
    expertise: List[str] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Cole Medin",
                "sources": {"youtube": "UCxxxxx", "github": "coleam00"},
                "expertise": ["AI Agents", "Context Engineering"]
            }
        }
```

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation & Infrastructure

Set up the project structure, database, and core dependencies before building any agents or crawlers.

**Tasks:**
- Initialize Python project with Poetry/uv
- Configure PostgreSQL with pgvector extension
- Create database schema (personas, content_chunks, wisdom_insights, crawl_jobs)
- Set up SQLAlchemy models and async session management
- Implement configuration management (Pydantic BaseSettings)
- Create Docker Compose setup for local development

### Phase 2: Crawlers & Content Ingestion

Build pluggable crawlers for YouTube and GitHub that can extract and store content.

**Tasks:**
- Implement BaseCrawler abstract interface
- Build YouTube crawler using yt-dlp (transcript extraction)
- Build GitHub crawler using PyGithub + git clone (READMEs, code)
- Create crawler registry for easy extension
- Implement content chunking strategy (RecursiveCharacterTextSplitter)
- Build embedding pipeline (OpenAI text-embedding-3-small or local)

### Phase 3: Vector Store & RAG

Integrate pgvector for semantic search and implement RAG retrieval patterns.

**Tasks:**
- Implement pgvector store with similarity search
- Create embedding generation utilities
- Build RAG retrieval functions (top-k with reranking)
- Add metadata filtering by persona_id
- Implement wisdom extraction using LLM (distill themes/principles)

### Phase 4: PydanticAI Agents

Build the core agents: PersonaBuilder (creates personas) and PersonaAgent (queries knowledge).

**Tasks:**
- Implement PersonaBuilder agent with discovery tools
- Implement PersonaAgent with RAG retrieval tools
- Add tool for wisdom querying
- Test agents locally with sample queries
- Implement agent context management (persona-specific prompts)

### Phase 5: MCP Server

Expose persona functionality as MCP tools for Claude Code integration.

**Tasks:**
- Set up FastMCP server
- Implement build_persona tool
- Implement query_persona tool
- Implement list_personas tool
- Test MCP server with Claude Code locally

### Phase 6: HTTP API

Build FastAPI endpoints for Telegram and other external integrations.

**Tasks:**
- Set up FastAPI application structure
- Create persona CRUD endpoints
- Implement query endpoint with streaming support
- Add health check endpoints
- Implement CORS and security middleware

### Phase 7: ColeAI Proof of Concept

Manually build the first persona (ColeAI) to validate the entire pipeline.

**Tasks:**
- Create ColeAI persona configuration
- Crawl Cole Medin's YouTube channel (AI Agents Masterclass)
- Crawl Cole Medin's GitHub repos (coleam00, dynamous-community)
- Generate embeddings and store in pgvector
- Extract wisdom on "Context Engineering" and "Agentic Coding"
- Test query quality with sample questions

### Phase 8: Automation & Documentation

Extract reusable patterns and document the system for future persona creation.

**Tasks:**
- Create `build_persona.py` script for automated persona creation
- Write comprehensive README with setup instructions
- Document API endpoints (OpenAPI/Swagger)
- Create example persona configurations (personas.yaml)
- Write integration guide for remote-coding-agent TypeScript platform

---

## STEP-BY-STEP TASKS

IMPORTANT: Execute every task in order, top to bottom. Each task is atomic and independently testable.

### CREATE pyproject.toml

- **IMPLEMENT**: Poetry project configuration with dependencies
- **PATTERN**: Standard Python project structure
- **IMPORTS**: N/A (root config file)
- **GOTCHA**: Pin major versions, use `^` for semver compatibility
- **VALIDATE**: `poetry install && poetry check`

```toml
[tool.poetry]
name = "persona-agent-system"
version = "0.1.0"
description = "Multi-persona knowledge bank with PydanticAI and RAG"
authors = ["Your Name <you@example.com>"]

[tool.poetry.dependencies]
python = "^3.11"
pydantic-ai = "^0.1.0"
fastapi = "^0.115.0"
uvicorn = {extras = ["standard"], version = "^0.32.0"}
sqlalchemy = "^2.0.0"
asyncpg = "^0.30.0"
pgvector = "^0.3.0"
fastmcp = "^0.5.0"
yt-dlp = "^2025.1.0"
pygithub = "^2.5.0"
openai = "^1.0.0"
httpx = "^0.28.0"
pydantic-settings = "^2.7.0"
python-dotenv = "^1.0.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.3.0"
pytest-asyncio = "^0.24.0"
black = "^24.10.0"
ruff = "^0.8.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

### CREATE .env.example

- **IMPLEMENT**: Environment variable template for configuration
- **PATTERN**: Document all required and optional variables
- **IMPORTS**: N/A
- **GOTCHA**: Never commit actual `.env` file with secrets
- **VALIDATE**: Manual review

```env
# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/persona_agents

# OpenAI (for embeddings and LLM calls)
OPENAI_API_KEY=sk-...

# PydanticAI Model (for agents)
PYDANTIC_AI_MODEL=openai:gpt-4

# GitHub (for crawling repos)
GITHUB_TOKEN=ghp_...

# MCP Server
MCP_SERVER_PORT=8100

# HTTP API
API_PORT=8000
API_HOST=0.0.0.0

# Logging
LOG_LEVEL=INFO
```

### CREATE docker-compose.yml

- **IMPLEMENT**: PostgreSQL 16 with pgvector extension
- **PATTERN**: Docker Compose for local development
- **IMPORTS**: N/A
- **GOTCHA**: Ensure pgvector extension is enabled in init script
- **VALIDATE**: `docker-compose up -d && docker-compose ps`

```yaml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: persona_agents
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./migrations:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
```

### CREATE migrations/001_init_schema.sql

- **IMPLEMENT**: Database schema with pgvector extension and tables
- **PATTERN**: SQL migration for PostgreSQL with vector types
- **IMPORTS**: N/A
- **GOTCHA**: Create extension first, then tables
- **VALIDATE**: `psql $DATABASE_URL < migrations/001_init_schema.sql`

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Personas table
CREATE TABLE personas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    sources JSONB DEFAULT '{}'::jsonb,
    expertise JSONB DEFAULT '[]'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Content chunks with vector embeddings
CREATE TABLE content_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    persona_id UUID REFERENCES personas(id) ON DELETE CASCADE,
    source_type VARCHAR(50) NOT NULL,  -- 'youtube', 'github', 'twitter', 'manual'
    source_url TEXT,
    title TEXT,
    content_text TEXT NOT NULL,
    embedding vector(1536),  -- OpenAI text-embedding-3-small dimension
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_content_persona_id ON content_chunks(persona_id);
CREATE INDEX idx_content_source_type ON content_chunks(source_type);
CREATE INDEX idx_content_embedding ON content_chunks
  USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Wisdom insights (distilled knowledge)
CREATE TABLE wisdom_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    persona_id UUID REFERENCES personas(id) ON DELETE CASCADE,
    category VARCHAR(100),
    insight_title VARCHAR(255),
    insight_text TEXT NOT NULL,
    evidence_chunk_ids UUID[],
    confidence_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Crawl jobs tracking
CREATE TABLE crawl_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    persona_id UUID REFERENCES personas(id) ON DELETE CASCADE,
    source_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'queued',  -- 'queued', 'running', 'completed', 'failed'
    progress JSONB DEFAULT '{}'::jsonb,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_crawl_jobs_persona ON crawl_jobs(persona_id);
CREATE INDEX idx_crawl_jobs_status ON crawl_jobs(status);
```

### CREATE src/config.py

- **IMPLEMENT**: Pydantic BaseSettings for environment configuration
- **PATTERN**: Type-safe configuration with validation
- **IMPORTS**: `from pydantic_settings import BaseSettings`
- **GOTCHA**: Use `.env` file for local dev, environment variables in production
- **VALIDATE**: `python -c "from src.config import settings; print(settings.database_url)"`

```python
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = Field(..., env="DATABASE_URL")

    # OpenAI
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")

    # PydanticAI
    pydantic_ai_model: str = Field(default="openai:gpt-4", env="PYDANTIC_AI_MODEL")

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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
```

### CREATE src/database.py

- **IMPLEMENT**: SQLAlchemy async engine and session management
- **PATTERN**: Async context manager for database sessions
- **IMPORTS**: `from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession`
- **GOTCHA**: Use `postgresql+asyncpg://` scheme for async
- **VALIDATE**: `python -c "from src.database import engine; print(engine)"`

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from src.config import settings

# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.log_level == "DEBUG",
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Base class for models
Base = declarative_base()


async def get_session() -> AsyncSession:
    """Dependency for FastAPI routes."""
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    """Initialize database (create tables if not exist)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
```

### CREATE src/models.py

- **IMPLEMENT**: SQLAlchemy ORM models for all database tables
- **PATTERN**: Declarative models with relationships
- **IMPORTS**: `from sqlalchemy import Column, String, Text, TIMESTAMP, ForeignKey, UUID, ARRAY, Float, JSON`
- **GOTCHA**: Use `pgvector` type for embeddings: `from pgvector.sqlalchemy import Vector`
- **VALIDATE**: `python -c "from src.models import Persona, ContentChunk; print('Models loaded')"`

```python
from sqlalchemy import Column, String, Text, TIMESTAMP, ForeignKey, UUID, ARRAY, Float, JSON, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from pgvector.sqlalchemy import Vector
from src.database import Base
import uuid


class Persona(Base):
    __tablename__ = "personas"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    sources = Column(JSON, default={})
    expertise = Column(JSON, default=[])
    metadata = Column(JSON, default={})
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())


class ContentChunk(Base):
    __tablename__ = "content_chunks"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    persona_id = Column(PG_UUID(as_uuid=True), ForeignKey("personas.id", ondelete="CASCADE"))
    source_type = Column(String(50), nullable=False)
    source_url = Column(Text)
    title = Column(Text)
    content_text = Column(Text, nullable=False)
    embedding = Column(Vector(1536))  # OpenAI text-embedding-3-small
    metadata = Column(JSON, default={})
    created_at = Column(TIMESTAMP, server_default=func.now())


class WisdomInsight(Base):
    __tablename__ = "wisdom_insights"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    persona_id = Column(PG_UUID(as_uuid=True), ForeignKey("personas.id", ondelete="CASCADE"))
    category = Column(String(100))
    insight_title = Column(String(255))
    insight_text = Column(Text, nullable=False)
    evidence_chunk_ids = Column(ARRAY(PG_UUID(as_uuid=True)))
    confidence_score = Column(Float)
    created_at = Column(TIMESTAMP, server_default=func.now())


class CrawlJob(Base):
    __tablename__ = "crawl_jobs"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    persona_id = Column(PG_UUID(as_uuid=True), ForeignKey("personas.id", ondelete="CASCADE"))
    source_type = Column(String(50), nullable=False)
    status = Column(String(50), default="queued")
    progress = Column(JSON, default={})
    error_message = Column(Text)
    started_at = Column(TIMESTAMP)
    completed_at = Column(TIMESTAMP)
    created_at = Column(TIMESTAMP, server_default=func.now())
```

### CREATE src/crawlers/base.py

- **IMPLEMENT**: Abstract BaseCrawler interface for pluggable crawlers
- **PATTERN**: Abstract base class with required methods
- **IMPORTS**: `from abc import ABC, abstractmethod`
- **GOTCHA**: All crawlers must implement `crawl()` method
- **VALIDATE**: `python -c "from src.crawlers.base import BaseCrawler; print('Base crawler loaded')"`

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseCrawler(ABC):
    """Abstract base class for all content crawlers."""

    def __init__(self, persona_id: str):
        self.persona_id = persona_id
        self.logger = logger

    @abstractmethod
    async def crawl(self, source_url: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Crawl a source and return extracted content.

        Args:
            source_url: URL or identifier for the source (channel ID, username, etc.)
            **kwargs: Crawler-specific options

        Returns:
            List of content dictionaries with keys:
            - title: str
            - content_text: str
            - source_url: str
            - metadata: dict (optional, source-specific data)
        """
        pass

    @abstractmethod
    def get_source_type(self) -> str:
        """Return the source type identifier (youtube, github, etc.)."""
        pass
```

### CREATE src/crawlers/youtube.py

- **IMPLEMENT**: YouTube transcript crawler using yt-dlp
- **PATTERN**: Inherit from BaseCrawler, use yt-dlp subprocess
- **IMPORTS**: `import subprocess, json, from yt_dlp import YoutubeDL`
- **GOTCHA**: Handle missing transcripts gracefully (auto-generated fallback)
- **VALIDATE**: `python -c "from src.crawlers.youtube import YouTubeCrawler; print('YouTube crawler loaded')"`

```python
from typing import List, Dict, Any
import yt_dlp
from src.crawlers.base import BaseCrawler
import logging

logger = logging.getLogger(__name__)


class YouTubeCrawler(BaseCrawler):
    """Crawl YouTube channel for video transcripts."""

    def get_source_type(self) -> str:
        return "youtube"

    async def crawl(self, channel_url: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Crawl YouTube channel and extract all video transcripts.

        Args:
            channel_url: YouTube channel URL or @username
            **kwargs:
                max_videos: int - Limit number of videos (default: None = all)

        Returns:
            List of video content dictionaries
        """
        max_videos = kwargs.get("max_videos", None)

        ydl_opts = {
            'quiet': True,
            'extract_flat': True,  # Get video list without downloading
            'force_generic_extractor': False,
        }

        videos = []

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                # Get channel info and video list
                channel_info = ydl.extract_info(channel_url, download=False)

                if 'entries' not in channel_info:
                    logger.warning(f"No videos found for {channel_url}")
                    return []

                video_urls = [
                    f"https://www.youtube.com/watch?v={entry['id']}"
                    for entry in channel_info['entries']
                    if entry.get('id')
                ]

                if max_videos:
                    video_urls = video_urls[:max_videos]

                logger.info(f"Found {len(video_urls)} videos to process")

                # Extract transcripts for each video
                for video_url in video_urls:
                    try:
                        transcript_data = await self._get_video_transcript(video_url)
                        if transcript_data:
                            videos.append(transcript_data)
                    except Exception as e:
                        logger.error(f"Failed to get transcript for {video_url}: {e}")
                        continue

                logger.info(f"Successfully extracted {len(videos)} transcripts")
                return videos

            except Exception as e:
                logger.error(f"Failed to crawl YouTube channel: {e}", exc_info=True)
                raise

    async def _get_video_transcript(self, video_url: str) -> Dict[str, Any] | None:
        """Extract transcript from a single video."""
        ydl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'skip_download': True,
            'quiet': True,
            'subtitleslangs': ['en'],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(video_url, download=False)

                # Get transcript from subtitles
                subtitles = info.get('subtitles', {}).get('en') or \
                           info.get('automatic_captions', {}).get('en')

                if not subtitles:
                    logger.warning(f"No transcript available for {video_url}")
                    return None

                # Download subtitle file
                subtitle_url = subtitles[0]['url']
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.get(subtitle_url)
                    subtitle_text = response.text

                # Parse VTT format and extract text
                transcript = self._parse_vtt(subtitle_text)

                return {
                    'title': info.get('title', 'Unknown'),
                    'content_text': transcript,
                    'source_url': video_url,
                    'metadata': {
                        'duration': info.get('duration'),
                        'upload_date': info.get('upload_date'),
                        'view_count': info.get('view_count'),
                    }
                }

            except Exception as e:
                logger.error(f"Error extracting transcript: {e}")
                return None

    def _parse_vtt(self, vtt_text: str) -> str:
        """Parse VTT subtitle format and extract clean text."""
        lines = vtt_text.split('\n')
        text_lines = []

        for line in lines:
            line = line.strip()
            # Skip timestamp lines and empty lines
            if '-->' in line or line.startswith('WEBVTT') or not line:
                continue
            # Skip cue identifiers (numbers)
            if line.isdigit():
                continue
            text_lines.append(line)

        return ' '.join(text_lines)
```

### CREATE src/crawlers/github.py

- **IMPLEMENT**: GitHub repository crawler using PyGithub
- **PATTERN**: Inherit from BaseCrawler, clone repos and extract content
- **IMPORTS**: `from github import Github, import subprocess`
- **GOTCHA**: Respect GitHub API rate limits (use authentication)
- **VALIDATE**: `python -c "from src.crawlers.github import GitHubCrawler; print('GitHub crawler loaded')"`

```python
from typing import List, Dict, Any
from github import Github
import tempfile
import subprocess
from pathlib import Path
from src.crawlers.base import BaseCrawler
from src.config import settings
import logging

logger = logging.getLogger(__name__)


class GitHubCrawler(BaseCrawler):
    """Crawl GitHub repositories for READMEs, code, and documentation."""

    def __init__(self, persona_id: str):
        super().__init__(persona_id)
        self.github = Github(settings.github_token)

    def get_source_type(self) -> str:
        return "github"

    async def crawl(self, username: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Crawl all public repositories for a GitHub user.

        Args:
            username: GitHub username
            **kwargs:
                repo_filter: List[str] - Specific repos to crawl (default: all)
                include_code: bool - Extract code files (default: False)

        Returns:
            List of content dictionaries
        """
        repo_filter = kwargs.get("repo_filter", None)
        include_code = kwargs.get("include_code", False)

        contents = []

        try:
            user = self.github.get_user(username)
            repos = user.get_repos()

            for repo in repos:
                # Skip if filtering and repo not in list
                if repo_filter and repo.name not in repo_filter:
                    continue

                logger.info(f"Crawling repository: {repo.full_name}")

                # Extract README
                readme_content = await self._get_readme(repo)
                if readme_content:
                    contents.append(readme_content)

                # Extract code files if requested
                if include_code:
                    code_contents = await self._extract_code_files(repo)
                    contents.extend(code_contents)

            logger.info(f"Extracted {len(contents)} items from GitHub")
            return contents

        except Exception as e:
            logger.error(f"Failed to crawl GitHub user {username}: {e}", exc_info=True)
            raise

    async def _get_readme(self, repo) -> Dict[str, Any] | None:
        """Extract README from repository."""
        try:
            readme = repo.get_readme()
            content = readme.decoded_content.decode('utf-8')

            return {
                'title': f"{repo.name} - README",
                'content_text': content,
                'source_url': repo.html_url,
                'metadata': {
                    'repo_name': repo.name,
                    'stars': repo.stargazers_count,
                    'language': repo.language,
                }
            }
        except Exception as e:
            logger.warning(f"No README found for {repo.name}: {e}")
            return None

    async def _extract_code_files(self, repo) -> List[Dict[str, Any]]:
        """Clone repo and extract Python/TypeScript/JavaScript files."""
        # TODO: Implement selective file extraction
        # This would clone the repo, extract .py, .ts, .js files
        # For MVP, we'll skip this to avoid complexity
        return []
```

### CREATE src/crawlers/registry.py

- **IMPLEMENT**: Crawler factory/registry for easy extension
- **PATTERN**: Factory pattern with registration
- **IMPORTS**: `from typing import Type, Dict`
- **GOTCHA**: Register all crawlers in `__init__`
- **VALIDATE**: `python -c "from src.crawlers.registry import get_crawler; print(get_crawler('youtube'))"`

```python
from typing import Type, Dict
from src.crawlers.base import BaseCrawler
from src.crawlers.youtube import YouTubeCrawler
from src.crawlers.github import GitHubCrawler


class CrawlerRegistry:
    """Registry for all available crawlers."""

    _crawlers: Dict[str, Type[BaseCrawler]] = {}

    @classmethod
    def register(cls, source_type: str, crawler_class: Type[BaseCrawler]):
        """Register a crawler for a source type."""
        cls._crawlers[source_type] = crawler_class

    @classmethod
    def get(cls, source_type: str) -> Type[BaseCrawler]:
        """Get crawler class for source type."""
        if source_type not in cls._crawlers:
            raise ValueError(f"No crawler registered for source type: {source_type}")
        return cls._crawlers[source_type]

    @classmethod
    def list_sources(cls) -> list[str]:
        """List all registered source types."""
        return list(cls._crawlers.keys())


# Register all crawlers
CrawlerRegistry.register("youtube", YouTubeCrawler)
CrawlerRegistry.register("github", GitHubCrawler)


def get_crawler(source_type: str, persona_id: str) -> BaseCrawler:
    """Factory function to create crawler instance."""
    crawler_class = CrawlerRegistry.get(source_type)
    return crawler_class(persona_id)
```

### CREATE src/vector_store/embeddings.py

- **IMPLEMENT**: Embedding generation using OpenAI API
- **PATTERN**: Batch embedding with retry logic
- **IMPORTS**: `from openai import AsyncOpenAI`
- **GOTCHA**: Batch embed for efficiency (max 2048 texts per request)
- **VALIDATE**: `python -c "from src.vector_store.embeddings import generate_embeddings; print('Embeddings module loaded')"`

```python
from openai import AsyncOpenAI
from typing import List
from src.config import settings
import logging

logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=settings.openai_api_key)


async def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using OpenAI.

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors (1536 dimensions)
    """
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
        logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
        raise


async def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a single text."""
    embeddings = await generate_embeddings([text])
    return embeddings[0]
```

### CREATE src/vector_store/chunking.py

- **IMPLEMENT**: Text chunking with RecursiveCharacterTextSplitter pattern
- **PATTERN**: Recursive chunking with overlap for context preservation
- **IMPORTS**: N/A (implement manually or use langchain)
- **GOTCHA**: Overlap prevents context loss at chunk boundaries
- **VALIDATE**: `python -c "from src.vector_store.chunking import chunk_text; print(chunk_text('test' * 1000))"`

```python
from typing import List
from src.config import settings


def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """
    Split text into chunks with overlap.

    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk (default from settings)
        chunk_overlap: Overlap between chunks (default from settings)

    Returns:
        List of text chunks
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

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
```

### CREATE src/vector_store/store.py

- **IMPLEMENT**: pgvector CRUD operations and similarity search
- **PATTERN**: SQLAlchemy with pgvector distance operators
- **IMPORTS**: `from sqlalchemy import select, from pgvector.sqlalchemy import Vector`
- **GOTCHA**: Use cosine distance for similarity (1 - cosine_similarity)
- **VALIDATE**: `python -c "from src.vector_store.store import VectorStore; print('Vector store loaded')"`

```python
from typing import List, Dict, Any
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pgvector.sqlalchemy import Vector
from src.models import ContentChunk
from src.vector_store.embeddings import generate_embedding
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
        k: int = 5
    ) -> List[ContentChunk]:
        """
        Search for similar content using vector similarity.

        Args:
            session: Database session
            persona_id: Filter by persona ID
            query: Search query text
            k: Number of results to return

        Returns:
            List of most similar content chunks
        """
        try:
            # Generate query embedding
            query_embedding = await generate_embedding(query)

            # Query with cosine distance
            stmt = (
                select(ContentChunk)
                .where(ContentChunk.persona_id == persona_id)
                .order_by(ContentChunk.embedding.cosine_distance(query_embedding))
                .limit(k)
            )

            result = await session.execute(stmt)
            chunks = result.scalars().all()

            logger.info(f"Found {len(chunks)} similar chunks for query")
            return chunks

        except Exception as e:
            logger.error(f"Similarity search failed: {e}", exc_info=True)
            raise
```

### CREATE src/agents/persona_agent.py

- **IMPLEMENT**: PydanticAI agent for querying persona knowledge
- **PATTERN**: Agent with RAG retrieval tool
- **IMPORTS**: `from pydantic_ai import Agent, RunContext`
- **GOTCHA**: Include source citations in responses
- **VALIDATE**: `python -c "from src.agents.persona_agent import PersonaAgent; print('Persona agent loaded')"`

```python
from pydantic_ai import Agent, RunContext
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


# Create PersonaAgent
persona_agent = Agent(
    model=settings.pydantic_ai_model,
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
```

### CREATE src/mcp_server/server.py

- **IMPLEMENT**: FastMCP server exposing persona tools
- **PATTERN**: FastMCP with async tools
- **IMPORTS**: `from fastmcp import FastMCP`
- **GOTCHA**: Tools must return JSON-serializable data
- **VALIDATE**: `python src/mcp_server/server.py` (should start server)

```python
from fastmcp import FastMCP
from typing import List, Dict, Any
from src.database import AsyncSessionLocal
from src.agents.persona_agent import query_persona
from src.models import Persona
from sqlalchemy import select
import logging

logger = logging.getLogger(__name__)

# Create MCP server
mcp = FastMCP("persona-knowledge")


@mcp.tool()
async def query_persona_knowledge(persona_slug: str, question: str) -> str:
    """
    Query a persona's knowledge base with a question.

    Args:
        persona_slug: Persona identifier (e.g., "cole-medin")
        question: Question to ask the persona

    Returns:
        Answer with source citations
    """
    async with AsyncSessionLocal() as session:
        # Find persona by slug
        stmt = select(Persona).where(Persona.slug == persona_slug)
        result = await session.execute(stmt)
        persona = result.scalar_one_or_none()

        if not persona:
            return f"Persona '{persona_slug}' not found. Available personas: {await list_available_personas()}"

        # Query persona agent
        answer = await query_persona(str(persona.id), question, session)
        return answer


@mcp.tool()
async def list_available_personas() -> List[str]:
    """
    List all available persona agents.

    Returns:
        List of persona slugs
    """
    async with AsyncSessionLocal() as session:
        stmt = select(Persona.slug, Persona.name)
        result = await session.execute(stmt)
        personas = result.all()

        return [f"{slug} ({name})" for slug, name in personas]


if __name__ == "__main__":
    # Run MCP server
    mcp.run()
```

### CREATE src/api/app.py

- **IMPLEMENT**: FastAPI application setup with routes
- **PATTERN**: FastAPI with async endpoints
- **IMPORTS**: `from fastapi import FastAPI, Depends`
- **GOTCHA**: Enable CORS for browser access
- **VALIDATE**: `uvicorn src.api.app:app --reload` (should start server)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import personas, query, health
from src.config import settings
import logging

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Persona Agent System API",
    description="Multi-persona knowledge bank with RAG",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(personas.router, prefix="/personas", tags=["personas"])
app.include_router(query.router, prefix="/query", tags=["query"])


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Persona Agent System API")
    logger.info(f"API running on http://{settings.api_host}:{settings.api_port}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
```

### CREATE src/api/routes/health.py

- **IMPLEMENT**: Health check endpoints
- **PATTERN**: Simple health check + database check
- **IMPORTS**: `from fastapi import APIRouter`
- **GOTCHA**: Database health check should be fast
- **VALIDATE**: `curl http://localhost:8000/health`

```python
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.database import get_session
from sqlalchemy import text

router = APIRouter()


@router.get("/")
async def health():
    """Basic health check."""
    return {"status": "ok"}


@router.get("/db")
async def health_db(session: AsyncSession = Depends(get_session)):
    """Database health check."""
    try:
        await session.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        return {"status": "error", "database": "disconnected", "error": str(e)}
```

### CREATE src/api/routes/personas.py

- **IMPLEMENT**: Persona CRUD endpoints
- **PATTERN**: RESTful API with Pydantic schemas
- **IMPORTS**: `from fastapi import APIRouter, Depends, HTTPException`
- **GOTCHA**: Return 404 for not found, 400 for validation errors
- **VALIDATE**: `curl http://localhost:8000/personas`

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.database import get_session
from src.models import Persona
from typing import List
from pydantic import BaseModel

router = APIRouter()


class PersonaResponse(BaseModel):
    id: str
    name: str
    slug: str
    description: str | None
    sources: dict
    expertise: list

    class Config:
        from_attributes = True


@router.get("/", response_model=List[PersonaResponse])
async def list_personas(session: AsyncSession = Depends(get_session)):
    """List all personas."""
    stmt = select(Persona)
    result = await session.execute(stmt)
    personas = result.scalars().all()
    return personas


@router.get("/{slug}", response_model=PersonaResponse)
async def get_persona(slug: str, session: AsyncSession = Depends(get_session)):
    """Get persona by slug."""
    stmt = select(Persona).where(Persona.slug == slug)
    result = await session.execute(stmt)
    persona = result.scalar_one_or_none()

    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")

    return persona
```

### CREATE src/api/routes/query.py

- **IMPLEMENT**: Query endpoint with streaming support
- **PATTERN**: FastAPI streaming response
- **IMPORTS**: `from fastapi.responses import StreamingResponse`
- **GOTCHA**: Stream must yield bytes, not strings
- **VALIDATE**: `curl -X POST http://localhost:8000/query -d '{"persona_slug":"cole-medin","question":"What is context engineering?"}'`

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from src.database import get_session
from src.models import Persona
from src.agents.persona_agent import query_persona
from pydantic import BaseModel

router = APIRouter()


class QueryRequest(BaseModel):
    persona_slug: str
    question: str


class QueryResponse(BaseModel):
    answer: str


@router.post("/", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    session: AsyncSession = Depends(get_session)
):
    """Query a persona with a question."""
    # Find persona
    stmt = select(Persona).where(Persona.slug == request.persona_slug)
    result = await session.execute(stmt)
    persona = result.scalar_one_or_none()

    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")

    # Query persona agent
    answer = await query_persona(str(persona.id), request.question, session)

    return QueryResponse(answer=answer)
```

### CREATE scripts/build_coleai.py

- **IMPLEMENT**: One-off script to build ColeAI persona
- **PATTERN**: Async script with full pipeline
- **IMPORTS**: `import asyncio, from src.crawlers.registry import get_crawler`
- **GOTCHA**: This is the proof-of-concept - run manually to validate
- **VALIDATE**: `python scripts/build_coleai.py`

```python
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
```

### CREATE README.md

- **IMPLEMENT**: Comprehensive setup and usage documentation
- **PATTERN**: Standard README structure
- **IMPORTS**: N/A
- **GOTCHA**: Include both local dev and production setup
- **VALIDATE**: Manual review

```markdown
# Persona Agent System

Multi-persona knowledge bank powered by PydanticAI, PostgreSQL pgvector, and RAG. Create AI agents from any person's public content.

## Features

- 🤖 **PydanticAI Agents**: Query persona knowledge with natural language
- 🔍 **RAG with pgvector**: Semantic search over all content
- 🕷️ **Pluggable Crawlers**: YouTube, GitHub, Twitter (extensible)
- 🎯 **Dual Interfaces**: MCP server (Claude Code) + HTTP API (Telegram, custom apps)
- 🧠 **Wisdom Extraction**: Distill recurring themes and principles
- ♻️ **Repeatable Pipeline**: Build new personas easily

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 14+ with pgvector
- OpenAI API key (for embeddings)
- GitHub token (for crawling)

### Installation

1. Clone repository:
```bash
git clone https://github.com/yourusername/persona-agent-system
cd persona-agent-system
```

2. Install dependencies:
```bash
poetry install
```

3. Set up environment:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Start PostgreSQL with pgvector:
```bash
docker-compose up -d
```

5. Initialize database:
```bash
poetry run python scripts/init_db.py
```

### Build Your First Persona (ColeAI)

```bash
poetry run python scripts/build_coleai.py
```

This will:
- Create Cole Medin persona
- Crawl YouTube AI Agents Masterclass
- Crawl GitHub repos (ai-agents-masterclass, context-engineering-intro)
- Generate embeddings and store in pgvector
- Extract wisdom on context engineering

### Query ColeAI

**Via HTTP API:**
```bash
# Start API server
poetry run uvicorn src.api.app:app --reload

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"persona_slug":"cole-medin","question":"What is context engineering?"}'
```

**Via MCP (Claude Code):**
```bash
# Start MCP server
poetry run python src/mcp_server/server.py

# In Claude Code, add MCP server config:
# Then use: query_persona_knowledge("cole-medin", "What is context engineering?")
```

## Architecture

See `.agents/plans/persona-agent-system-mvp.md` for detailed implementation plan.

## Development

```bash
# Run tests
poetry run pytest

# Format code
poetry run black src/

# Lint
poetry run ruff check src/
```

## Adding New Personas

1. Create persona config in `config/personas.yaml`
2. Run build script:
```bash
poetry run python scripts/build_persona.py --config config/personas.yaml --persona mike-israetel
```

## License

MIT
```

---

## TESTING STRATEGY

### Unit Tests

**Framework**: pytest with pytest-asyncio

**Coverage Requirements**: 80%+ for core modules (crawlers, agents, vector_store)

**Test Structure**:
```
tests/
├── test_crawlers.py - Test YouTube and GitHub crawlers
├── test_agents.py - Test PersonaAgent RAG retrieval
├── test_vector_store.py - Test embedding and similarity search
└── test_api.py - Test FastAPI endpoints
```

### Integration Tests

**Database Tests**:
- Test full pipeline: crawl → chunk → embed → store → retrieve
- Use test database with pgvector
- Clean up after each test

**API Tests**:
- Test all endpoints with real database
- Mock external APIs (YouTube, GitHub, OpenAI)
- Validate response schemas

### Edge Cases

1. **Missing YouTube transcripts** - Fall back to auto-generated captions
2. **GitHub rate limits** - Implement exponential backoff
3. **Empty search results** - Return helpful message
4. **Persona not found** - 404 with available personas list
5. **Large content** - Chunk properly (test 10k+ character texts)

---

## VALIDATION COMMANDS

Execute every command to ensure zero regressions and 100% feature correctness.

### Level 1: Environment Setup

```bash
# Install dependencies
poetry install

# Verify PostgreSQL with pgvector
docker-compose up -d
docker exec -it persona-agent-system-postgres-1 psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS vector; SELECT * FROM pg_extension WHERE extname='vector';"

# Expected: vector extension listed
```

### Level 2: Database Initialization

```bash
# Run migrations
psql postgresql://postgres:postgres@localhost:5432/persona_agents < migrations/001_init_schema.sql

# Verify tables
psql postgresql://postgres:postgres@localhost:5432/persona_agents -c "\dt"

# Expected: personas, content_chunks, wisdom_insights, crawl_jobs tables
```

### Level 3: Module Imports

```bash
# Test all modules load without errors
poetry run python -c "
from src.config import settings;
from src.database import engine;
from src.models import Persona, ContentChunk;
from src.crawlers.registry import get_crawler;
from src.vector_store.embeddings import generate_embedding;
from src.agents.persona_agent import persona_agent;
print('✓ All modules loaded successfully')
"

# Expected: ✓ All modules loaded successfully
```

### Level 4: Build ColeAI Persona

```bash
# Build ColeAI (proof of concept)
poetry run python scripts/build_coleai.py

# Expected: Logs showing:
# - Persona created
# - YouTube videos crawled
# - GitHub repos crawled
# - Content chunks stored
```

### Level 5: Query Testing

```bash
# Start API server (background)
poetry run uvicorn src.api.app:app --host 0.0.0.0 --port 8000 &

# Test health endpoint
curl http://localhost:8000/health
# Expected: {"status":"ok"}

# Test personas list
curl http://localhost:8000/personas
# Expected: JSON array with cole-medin persona

# Test query endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"persona_slug":"cole-medin","question":"What is context engineering?"}'

# Expected: JSON with answer and source citations
```

### Level 6: MCP Server Testing

```bash
# Start MCP server
poetry run python src/mcp_server/server.py &

# Test with Claude Code (manual)
# - Add MCP server in Claude Code settings
# - Call query_persona_knowledge("cole-medin", "Explain agentic coding")
# - Verify response with citations
```

### Level 7: Code Quality

```bash
# Format check
poetry run black --check src/

# Lint check
poetry run ruff check src/

# Type check (if using mypy)
poetry run mypy src/

# Expected: All checks pass
```

---

## ACCEPTANCE CRITERIA

- [x] PostgreSQL database with pgvector extension created
- [x] All database tables created (personas, content_chunks, wisdom_insights, crawl_jobs)
- [x] YouTube crawler extracts transcripts from Cole Medin's channel
- [x] GitHub crawler extracts READMEs from coleam00 repositories
- [x] Content chunking splits large texts with overlap
- [x] Embeddings generated using OpenAI text-embedding-3-small
- [x] pgvector similarity search returns relevant chunks
- [x] PersonaAgent answers questions with RAG retrieval
- [x] MCP server exposes query_persona_knowledge tool
- [x] HTTP API endpoints functional (personas, query, health)
- [x] ColeAI persona built successfully via build_coleai.py
- [x] Sample queries return accurate answers with source citations
- [x] All validation commands pass
- [x] Code follows Python best practices (type hints, async/await, logging)
- [x] README documentation complete with setup instructions

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order (top to bottom)
- [ ] Each task validation passed immediately after implementation
- [ ] All validation commands executed successfully:
  - [ ] Level 1: Environment setup (Poetry, PostgreSQL, pgvector)
  - [ ] Level 2: Database initialization (migrations, tables)
  - [ ] Level 3: Module imports (all modules load)
  - [ ] Level 4: ColeAI persona built (crawlers work, embeddings generated)
  - [ ] Level 5: Query testing (API endpoints functional)
  - [ ] Level 6: MCP server testing (Claude Code integration)
  - [ ] Level 7: Code quality (format, lint, types)
- [ ] ColeAI persona answers questions accurately with citations
- [ ] HTTP API documented with OpenAPI/Swagger
- [ ] MCP server tools tested with Claude Code
- [ ] README.md complete with setup and usage guide
- [ ] All acceptance criteria met
- [ ] Code reviewed for quality, security, and maintainability

---

## NOTES

### Design Decisions

**1. OpenAI Embeddings vs Local Models**
- **Choice**: OpenAI text-embedding-3-small
- **Rationale**: Better quality, consistent dimensions (1536), no local GPU required
- **Trade-off**: Cost (~$0.02 per million tokens) vs quality and simplicity
- **Future**: Can add local model support (sentence-transformers) later

**2. FastMCP vs Low-Level MCP SDK**
- **Choice**: FastMCP
- **Rationale**: Simpler API, automatic type inference, less boilerplate
- **Trade-off**: Less control vs faster development

**3. SQLAlchemy ORM vs Raw SQL**
- **Choice**: SQLAlchemy async ORM
- **Rationale**: Type safety, relationship management, easier migrations
- **Trade-off**: Slight performance overhead vs developer productivity

**4. Chunking Strategy**
- **Choice**: Fixed-size with overlap (1000 chars, 200 overlap)
- **Rationale**: Simple, predictable, preserves context at boundaries
- **Trade-off**: May split mid-sentence vs semantic chunking complexity

**5. YouTube Transcript Source**
- **Choice**: yt-dlp with VTT subtitle parsing
- **Rationale**: Reliable, handles auto-generated captions
- **Alternative Considered**: youtube-transcript-api (simpler but less robust)

### Future Enhancements

**Phase 2 (Post-ColeAI MVP)**:
1. **Automated Persona Builder**: Extract build_coleai.py pattern into generic pipeline
2. **Additional Crawlers**: Twitter/X, Instagram, podcasts (RSS)
3. **Wisdom Extraction**: LLM-based theme distillation (not just RAG retrieval)
4. **Multi-modal**: Support video screenshots, code syntax highlighting
5. **Persona Switching**: Chat interface that switches between personas
6. **Reranking**: Add cross-encoder for better retrieval quality

**Integration with remote-coding-agent**:
- Add PersonaAgent as new AI assistant type in TypeScript platform
- Consume via HTTP API endpoints
- Support `/switch-persona` command in Telegram

### Known Limitations

1. **No Dynamous.ai Content**: Respecting member-only content boundaries
2. **YouTube Limit**: Building ColeAI with only 10 videos for MVP (can increase)
3. **No Code Analysis**: GitHub crawler extracts READMEs only (not code files)
4. **Basic Chunking**: Simple character-based (future: semantic chunking)
5. **Single Embedding Model**: OpenAI only (future: pluggable embeddings)

### Security Considerations

- API keys stored in environment variables only (never committed)
- Database credentials isolated in Docker Compose
- HTTP API needs authentication for production (add JWT/API keys)
- MCP server should run locally only (not exposed to internet)
- pgvector indexes use cosine distance (normalized vectors required)

### Performance Notes

- **Embedding Generation**: Batch embed for efficiency (up to 2048 texts/request)
- **pgvector Index**: IVFFlat with 100 lists (tune for dataset size)
- **Database Pool**: 10 connections, 20 overflow (tune for concurrent load)
- **Chunking**: ~5 chunks per YouTube video (~30min) = 50 chunks for 10 videos
- **Search Latency**: <200ms for similarity search (5 results from 500 chunks)

---

**This plan is ready for execution. Good luck building ColeAI!** 🚀
