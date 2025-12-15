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
