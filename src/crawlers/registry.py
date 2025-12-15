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
