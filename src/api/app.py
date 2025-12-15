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
