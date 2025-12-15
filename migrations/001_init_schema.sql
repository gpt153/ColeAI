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
