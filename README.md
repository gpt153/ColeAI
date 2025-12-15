# ColeAI 🤖

**Production-Grade RAG System with Cole Medin's Best Practices**

Multi-persona knowledge bank powered by **PydanticAI**, **PostgreSQL pgvector**, and **advanced RAG strategies**. Create AI agents from any person's public content with state-of-the-art retrieval quality.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL](https://img.shields.io/badge/postgresql-14+-blue.svg)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-0.5.0+-blue.svg)](https://github.com/pgvector/pgvector)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 What Makes This Special?

This project implements **production RAG best practices** from [Cole Medin's mcp-crawl4ai-rag](https://github.com/coleam00/mcp-crawl4ai-rag), featuring:

- ✅ **Header-Based Chunking** - Intelligent text segmentation by markdown/HTML headers (better semantic boundaries)
- ✅ **Code Block Extraction** - Separate indexing of code examples with LLM-generated summaries
- ✅ **CrossEncoder Reranking** - Post-retrieval scoring for improved relevance (3x fetch → rerank → top K)
- ✅ **Batch Embeddings** - Exponential backoff retry logic with fault tolerance
- ✅ **Toggleable Strategies** - Independent configuration of each RAG enhancement
- ✅ **Source Filtering** - Precision queries by URL/domain

### Performance Metrics

| Configuration | Response Time | Quality |
|--------------|--------------|---------|
| Basic RAG | 34.38s | Good |
| Enhanced RAG (reranking only) | 48.10s | Better |
| **Enhanced RAG (full)** | **46.31s** | **Best** |

**Result**: 85% more semantic chunks (61 vs 33), 17 code blocks with summaries, significantly improved retrieval quality with acceptable latency increase.

---

## 🚀 Features

### Core Capabilities
- 🤖 **PydanticAI Agents** - Query persona knowledge with natural language using Claude 3 Opus
- 🔍 **Advanced RAG** - Header-based chunking, code extraction, and CrossEncoder reranking
- 📊 **pgvector Search** - Semantic similarity search with cosine distance
- 🕷️ **Pluggable Crawlers** - YouTube, GitHub, Twitter (extensible architecture)
- 🎯 **Dual Interfaces** - MCP server (Claude Code) + HTTP API (Telegram, custom apps)
- 🧠 **Wisdom Extraction** - Distill recurring themes and principles from content
- ♻️ **Repeatable Pipeline** - Build new personas with a single script

### RAG Enhancements (Toggleable)

All strategies can be independently enabled/disabled via environment variables:

| Strategy | Feature | Impact | Cost |
|----------|---------|--------|------|
| **Header Chunking** | `USE_HEADER_CHUNKING=true` | Better semantic boundaries | No extra cost |
| **Code Extraction** | `USE_CODE_EXTRACTION=true` | Improved code search | +1 LLM call per code block (indexing) |
| **Reranking** | `USE_RERANKING=true` | Better relevance | Local model (no API cost) |
| **Contextual Embeddings** | `USE_CONTEXTUAL_EMBEDDINGS=true` | Enhanced retrieval | +1 LLM call per chunk (indexing) |

---

## 📦 Installation

### Prerequisites

- **Python 3.11+**
- **PostgreSQL 14+** with **pgvector** extension
- **Docker** (recommended for PostgreSQL)
- **Poetry** for dependency management

### API Keys Required

- **OpenAI API Key** - For embeddings (`text-embedding-3-small`)
- **Anthropic API Key** - For PydanticAI agent (Claude 3 Opus)
- **GitHub Token** - For crawling repositories

### Setup Steps

1. **Clone repository**:
```bash
git clone https://github.com/gpt153/ColeAI.git
cd ColeAI
```

2. **Install dependencies**:
```bash
poetry install
```

3. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your API keys and RAG configuration
```

4. **Start PostgreSQL with pgvector**:
```bash
docker-compose up -d
```

5. **Initialize database**:
```bash
psql $DATABASE_URL < migrations/001_init_schema.sql
```

### Docker Installation (Recommended)

The easiest way to run ColeAI is using Docker Compose, which handles both the application and PostgreSQL setup.

#### Option A: Pre-built Image (Fastest) ⭐

Use the pre-built image from GitHub Container Registry - no local build required!

**System Requirements**:
- Docker 20.10+ with Docker Compose V2
- At least 8GB RAM
- At least 2GB free disk space (just to pull the image)

**Steps:**

1. **Clone repository**:
```bash
git clone https://github.com/gpt153/ColeAI.git
cd ColeAI
```

2. **Configure API keys**:
```bash
cp docker-compose.override.yml.example docker-compose.override.yml
# Edit docker-compose.override.yml with your API keys
```

3. **Start services** (uses pre-built image):
```bash
docker compose -f docker-compose.prebuilt.yml up -d
```

4. **Check logs**:
```bash
docker compose -f docker-compose.prebuilt.yml logs -f app
```

5. **Access**:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

---

#### Option B: Build Locally

Build the Docker image yourself. Requires more disk space and time.

**System Requirements**:
- Docker 20.10+ with Docker Compose V2
- At least 8GB RAM
- At least 10GB free disk space (for build process)

1. **Clone repository**:
```bash
git clone https://github.com/gpt153/ColeAI.git
cd ColeAI
```

2. **Configure API keys**:
```bash
cp docker-compose.override.yml.example docker-compose.override.yml
# Edit docker-compose.override.yml with your API keys:
# - OPENAI_API_KEY
# - ANTHROPIC_API_KEY
# - GITHUB_TOKEN
```

3. **Start all services**:
```bash
docker-compose up -d --build
```

4. **Check logs**:
```bash
# View all logs
docker-compose logs -f

# View app logs only
docker-compose logs -f app

# View postgres logs only
docker-compose logs -f postgres
```

5. **Access the application**:
- **HTTP API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

6. **Initialize with test data** (optional):
```bash
# Run test script inside container
docker-compose exec app python scripts/test_github_only.py
```

7. **Stop services**:
```bash
docker-compose down
```

**What's included**:
- ✅ PostgreSQL 16 with pgvector extension
- ✅ Automatic database initialization (migrations run on first start)
- ✅ FastAPI server on port 8000
- ✅ MCP server on port 8100
- ✅ Health checks and automatic restarts
- ✅ Persistent data storage (Docker volumes)

---

## ⚙️ Configuration

### RAG Strategy Configuration

Edit `.env` to enable/disable RAG enhancements:

```env
# ===== RAG Strategy Toggles =====

# Header-based chunking (recommended for production)
USE_HEADER_CHUNKING=true

# Code extraction with LLM summaries (recommended for code-heavy content)
USE_CODE_EXTRACTION=true

# CrossEncoder reranking (recommended for production)
USE_RERANKING=true

# Contextual embeddings (optional, requires reindexing)
USE_CONTEXTUAL_EMBEDDINGS=false

# ===== RAG Settings =====

# LLM for contextual embeddings and code summaries
CONTEXT_MODEL=gpt-4o-mini

# CrossEncoder model for reranking
RERANKING_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Minimum code block length for extraction (characters)
MIN_CODE_BLOCK_LENGTH=300

# Embedding model
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# Chunking settings (for fixed-size fallback)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Recommended Configurations

**1. Development (Fast)**
```env
USE_HEADER_CHUNKING=false
USE_CODE_EXTRACTION=false
USE_RERANKING=false
USE_CONTEXTUAL_EMBEDDINGS=false
```

**2. Production (Balanced)** ⭐ Recommended
```env
USE_HEADER_CHUNKING=true
USE_CODE_EXTRACTION=true
USE_RERANKING=true
USE_CONTEXTUAL_EMBEDDINGS=false
```

**3. Maximum Quality (Slow)**
```env
USE_HEADER_CHUNKING=true
USE_CODE_EXTRACTION=true
USE_RERANKING=true
USE_CONTEXTUAL_EMBEDDINGS=true  # Requires reindexing!
```

---

## 🎓 Build Your First Persona (ColeAI)

### Option 1: Quick Test (Single GitHub Repo)

```bash
poetry run python scripts/test_github_only.py
```

This creates a test persona with:
- Persona: `test-dev`
- Source: anthropic-sdk-python README
- Purpose: Quick RAG testing

### Option 2: Full ColeAI Persona

```bash
poetry run python scripts/build_coleai.py
```

This will:
- Create Cole Medin persona (`cole-medin`)
- Crawl YouTube AI Agents Masterclass
- Crawl GitHub repos (ai-agents-masterclass, context-engineering-intro)
- Generate embeddings with enhanced RAG strategies
- Extract wisdom on context engineering

### Rebuild with Enhanced Strategies

After changing RAG configuration in `.env`, rebuild the knowledge base:

```bash
poetry run python scripts/rebuild_with_enhanced_rag.py
```

This will:
1. Clear existing content chunks
2. Re-crawl all sources
3. Apply new chunking strategy (header-based or size-based)
4. Extract and index code blocks with LLM summaries
5. Generate embeddings with optional contextual enhancement

**Output Example**:
```
================================================================================
✅ REBUILD COMPLETE!
================================================================================
📊 Statistics:
  - Total text chunks: 61
  - Total code blocks: 17
  - Total indexed items: 78
  - Chunking strategy: Header-based
  - Code extraction: Enabled
```

---

## 🔍 Query ColeAI

### Via HTTP API

```bash
# Start API server
poetry run uvicorn src.api.app:app --reload --port 8000

# Query the persona
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "persona_slug": "test-dev",
    "question": "Show me how to make async API calls with the Anthropic SDK"
  }'
```

**Response includes**:
- AI-generated answer
- Source citations with URLs
- Reranked context chunks (if reranking enabled)

### Via MCP Server (Claude Code)

```bash
# Start MCP server on port 8100
poetry run python src/mcp_server/server.py
```

Add to your Claude Code MCP config (`~/.config/claude-code/mcp_config.json`):
```json
{
  "mcpServers": {
    "coleai": {
      "command": "poetry",
      "args": ["run", "python", "src/mcp_server/server.py"],
      "cwd": "/path/to/ColeAI"
    }
  }
}
```

Then in Claude Code:
```
query_persona_knowledge("test-dev", "What is context engineering?")
```

---

## 🧪 Testing RAG Strategies

Test and compare different RAG configurations:

```bash
poetry run python scripts/test_rag_strategies.py
```

**Output**:
```
================================================================================
TEST QUESTION: Show me how to make async API calls with the Anthropic SDK
================================================================================

📊 Current RAG Configuration:
  - Contextual Embeddings: false
  - Reranking: true
  - Code Extraction: true
  - Header Chunking: true
  - Context Model: gpt-4o-mini
  - Reranking Model: cross-encoder/ms-marco-MiniLM-L-6-v2

🚀 Querying with current RAG configuration...

================================================================================
✅ RESPONSE (took 46.31s):
================================================================================
[AI response with code examples and citations]
```

The script provides recommendations for optimal RAG configurations based on your use case.

---

## 🏗️ Architecture

### Tech Stack

- **PydanticAI** - Type-safe agent framework with Claude 3 Opus
- **PostgreSQL + pgvector** - Vector database for similarity search
- **OpenAI Embeddings** - `text-embedding-3-small` (1536 dimensions)
- **sentence-transformers** - CrossEncoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **FastAPI** - HTTP API server
- **MCP Protocol** - Claude Code integration
- **Docker Compose** - Container orchestration

### Project Structure

```
ColeAI/
├── src/
│   ├── agents/              # PydanticAI persona agents
│   │   └── persona_agent.py # Query handler with RAG
│   ├── api/                 # FastAPI HTTP endpoints
│   │   ├── app.py
│   │   └── routes/
│   ├── crawlers/            # Content crawlers
│   │   ├── github.py        # GitHub repository crawler
│   │   └── youtube.py       # YouTube transcript crawler
│   ├── mcp_server/          # MCP server for Claude Code
│   │   └── server.py
│   ├── vector_store/        # RAG implementation ⭐
│   │   ├── chunking.py      # Header-based chunking + code extraction
│   │   ├── embeddings.py    # Batch embeddings with retry logic
│   │   ├── reranking.py     # CrossEncoder reranking module
│   │   └── store.py         # pgvector similarity search
│   ├── config.py            # Configuration with RAG toggles
│   ├── database.py          # SQLAlchemy async setup
│   └── models.py            # ORM models (Persona, ContentChunk)
├── scripts/
│   ├── build_coleai.py              # Build Cole Medin persona
│   ├── test_github_only.py          # Quick test persona
│   ├── rebuild_with_enhanced_rag.py # Rebuild with new RAG config ⭐
│   └── test_rag_strategies.py       # Compare RAG configurations ⭐
├── migrations/
│   └── 001_init_schema.sql          # Database schema
├── .env.example                     # Configuration template
├── docker-compose.yml               # PostgreSQL + pgvector
├── pyproject.toml                   # Poetry dependencies
└── README.md
```

### Data Flow

1. **Crawling** → Content sources (GitHub, YouTube) → Raw content
2. **Chunking** → Header-based segmentation → Semantic chunks
3. **Code Extraction** → LLM summaries → Indexed code blocks
4. **Embedding** → OpenAI API (batch) → 1536-dim vectors
5. **Storage** → PostgreSQL + pgvector → Vector search ready
6. **Query** → User question → Query embedding
7. **Retrieval** → Vector similarity (3x if reranking) → Candidate chunks
8. **Reranking** → CrossEncoder scoring → Top K results
9. **Generation** → PydanticAI + Claude → Final answer with citations

---

## 📈 Performance & Optimization

### Indexing Performance

With enhanced RAG (header chunking + code extraction + reranking):
- **anthropic-sdk-python README**: 61 chunks + 17 code blocks = 78 items
- **Indexing time**: ~2-3 minutes (includes LLM calls for code summaries)
- **Embeddings**: Batch processing with exponential backoff (1s, 2s, 4s)

### Query Performance

| Metric | Basic RAG | Enhanced RAG | Impact |
|--------|-----------|--------------|--------|
| Response time | 34.38s | 46.31s | +35% latency |
| Retrieval quality | Good | Excellent | Significantly better |
| Code examples | Limited | Rich | 17 summarized blocks |
| Semantic accuracy | Good | Excellent | Improved via reranking |

### Cost Analysis

**Indexing Costs** (one-time per rebuild):
- Contextual embeddings: +$0.01-0.05 per 1000 chunks (gpt-4o-mini)
- Code summaries: +$0.01-0.03 per 100 code blocks (gpt-4o-mini)
- Embeddings: ~$0.13 per 1M tokens (text-embedding-3-small)

**Query Costs** (per query):
- Reranking: $0 (local CrossEncoder model)
- Embeddings: ~$0.0001 per query
- Claude 3 Opus: ~$0.01-0.05 per query (varies by response length)

---

## 🧑‍💻 Development

### Run Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
poetry run black src/ scripts/

# Lint
poetry run ruff check src/ scripts/

# Type checking
poetry run mypy src/
```

### Database Management

```bash
# Access PostgreSQL shell
docker exec -it persona_agents_db psql -U postgres -d persona_agents

# Check vector extensions
SELECT * FROM pg_extension WHERE extname = 'vector';

# View personas
SELECT id, name, slug FROM personas;

# View content chunks
SELECT id, persona_id, source_type, title FROM content_chunks LIMIT 10;
```

---

## 🔧 Adding New Personas

### Manual Approach

1. Create persona in database:
```python
from src.models import Persona
from src.database import AsyncSessionLocal

async with AsyncSessionLocal() as session:
    persona = Persona(
        name="Mike Israetel",
        slug="mike-israetel",
        bio="Exercise scientist and bodybuilding coach"
    )
    session.add(persona)
    await session.commit()
```

2. Crawl content:
```python
from src.crawlers.youtube import YouTubeCrawler

crawler = YouTubeCrawler(persona_id=str(persona.id))
contents = await crawler.crawl(channel_url="...")
```

3. Index content:
```python
from src.vector_store.store import VectorStore

for content in contents:
    await VectorStore.add_content(
        session=session,
        persona_id=str(persona.id),
        source_type="youtube",
        title=content["title"],
        content_text=content["content_text"],
        source_url=content["source_url"]
    )
```

### Automated Script (Template)

Create `scripts/build_<persona>.py`:
```python
import asyncio
from src.database import init_db, AsyncSessionLocal
from src.models import Persona
from src.crawlers.registry import get_crawler
from src.vector_store.store import VectorStore

async def build_persona():
    await init_db()
    async with AsyncSessionLocal() as session:
        # Create persona
        persona = Persona(name="...", slug="...", bio="...")
        session.add(persona)
        await session.commit()

        # Crawl and index
        crawler = get_crawler("youtube", str(persona.id))
        contents = await crawler.crawl(channel_url="...")

        for content in contents:
            await VectorStore.add_content(session, ...)

if __name__ == "__main__":
    asyncio.run(build_persona())
```

---

## 🙏 Credits & References

### Inspiration

This project implements production RAG best practices from:
- **[Cole Medin's mcp-crawl4ai-rag](https://github.com/coleam00/mcp-crawl4ai-rag)** - Production RAG implementation with Supabase
- **[Cole Medin's YouTube Channel](https://www.youtube.com/@ColeMedin)** - AI Agents Masterclass

### Key Concepts Learned

- **Header-based chunking** for better semantic boundaries
- **Code block extraction** with LLM summaries for agentic RAG
- **CrossEncoder reranking** for post-retrieval quality improvement
- **3x fetch strategy** (retrieve more, rerank, return top K)
- **Batch processing** with exponential backoff for fault tolerance
- **Toggleable RAG strategies** for production flexibility

### Technologies

- **[pgvector](https://github.com/pgvector/pgvector)** - Vector similarity search for PostgreSQL
- **[PydanticAI](https://ai.pydantic.dev/)** - Type-safe AI agent framework
- **[sentence-transformers](https://www.sbert.net/)** - CrossEncoder models for reranking
- **[Anthropic Claude](https://www.anthropic.com/claude)** - LLM for agents and summaries
- **[OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)** - text-embedding-3-small

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/gpt153/ColeAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/gpt153/ColeAI/discussions)

---

**Built with ❤️ using Cole Medin's production RAG best practices**
