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
    meta = Column('metadata', JSON, default={})  # Renamed to avoid SQLAlchemy conflict
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
    meta = Column('metadata', JSON, default={})  # Renamed to avoid SQLAlchemy conflict
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
