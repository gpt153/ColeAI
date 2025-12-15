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
