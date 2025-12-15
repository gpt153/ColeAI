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
