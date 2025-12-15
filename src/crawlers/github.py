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
        Crawl all public repositories for a GitHub user or organization.

        Args:
            username: GitHub username or organization name
            **kwargs:
                repo_filter: List[str] - Specific repos to crawl (default: all)
                include_code: bool - Extract code files (default: False)
                is_org: bool - Whether username is an organization (default: False)
                rate_limit_delay: float - Delay in seconds between repos (default: 0)
                recursive: bool - Recursively crawl all subdirectories (default: True)
                file_extensions: List[str] - File extensions to include (default: ['.md', '.py', '.ts', '.js'])

        Returns:
            List of content dictionaries
        """
        repo_filter = kwargs.get("repo_filter", None)
        include_code = kwargs.get("include_code", False)
        is_org = kwargs.get("is_org", False)
        rate_limit_delay = kwargs.get("rate_limit_delay", 0)
        recursive = kwargs.get("recursive", True)
        file_extensions = kwargs.get("file_extensions", ['.md', '.py', '.ts', '.js', '.tsx', '.jsx'])

        contents = []

        try:
            # Get repos from user or organization
            if is_org:
                org = self.github.get_organization(username)
                repos = org.get_repos()
                logger.info(f"Crawling organization: {username}")
            else:
                user = self.github.get_user(username)
                repos = user.get_repos()
                logger.info(f"Crawling user: {username}")

            import asyncio

            for repo in repos:
                # Skip if filtering and repo not in list
                if repo_filter and repo.name not in repo_filter:
                    continue

                logger.info(f"Crawling repository: {repo.full_name}")

                if recursive:
                    # Recursively extract all files
                    repo_contents = await self._extract_all_files(repo, file_extensions)
                    contents.extend(repo_contents)
                    logger.info(f"  ✓ Extracted {len(repo_contents)} files from {repo.name}")
                else:
                    # Extract only root README
                    readme_content = await self._get_readme(repo)
                    if readme_content:
                        contents.append(readme_content)

                # Rate limiting delay
                if rate_limit_delay > 0:
                    logger.info(f"Waiting {rate_limit_delay}s before next repo...")
                    await asyncio.sleep(rate_limit_delay)

            logger.info(f"Extracted {len(contents)} items from GitHub")
            return contents

        except Exception as e:
            logger.error(f"Failed to crawl GitHub {'org' if is_org else 'user'} {username}: {e}", exc_info=True)
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

    async def _extract_all_files(self, repo, file_extensions: List[str]) -> List[Dict[str, Any]]:
        """Recursively extract all files matching the given extensions from the repository."""
        contents = []
        processed = 0
        skipped = 0

        try:
            # Start traversing from root
            contents_list = repo.get_contents("")

            while contents_list:
                file_content = contents_list.pop(0)

                if file_content.type == "dir":
                    # If it's a directory, add its contents to the queue
                    try:
                        contents_list.extend(repo.get_contents(file_content.path))
                    except Exception as e:
                        logger.warning(f"Could not access directory {file_content.path}: {e}")
                        continue
                else:
                    processed += 1

                    # Only extract README.md and documentation markdown files
                    filename = file_content.path.split('/')[-1].lower()

                    # Prioritize README files and docs
                    is_readme = filename == 'readme.md'
                    is_docs = 'doc' in file_content.path.lower() or '/docs/' in file_content.path.lower()
                    is_md = file_content.path.endswith('.md')

                    # Skip non-markdown files unless it's in docs
                    if not is_md:
                        skipped += 1
                        continue

                    # Only process README files and files in docs directories
                    if not (is_readme or is_docs):
                        skipped += 1
                        continue

                    try:
                        # Get file content
                        file_data = repo.get_contents(file_content.path)

                        # Skip files larger than 500KB
                        if file_data.size > 500_000:
                            logger.warning(f"Skipping large file: {file_content.path} ({file_data.size} bytes)")
                            skipped += 1
                            continue

                        decoded_content = file_data.decoded_content.decode('utf-8')

                        contents.append({
                            'title': f"{repo.name} - {file_content.path}",
                            'content_text': decoded_content,
                            'source_url': file_data.html_url,
                            'metadata': {
                                'repo_name': repo.name,
                                'file_path': file_content.path,
                                'stars': repo.stargazers_count,
                                'language': repo.language,
                            }
                        })

                        logger.info(f"  📄 {file_content.path}")

                    except Exception as e:
                        logger.warning(f"Could not read file {file_content.path}: {e}")
                        skipped += 1
                        continue

                    # Progress logging every 50 files
                    if processed % 50 == 0:
                        logger.info(f"  Progress: {processed} files scanned, {len(contents)} extracted, {skipped} skipped")

            logger.info(f"  Final: {processed} files scanned, {len(contents)} extracted, {skipped} skipped")
            return contents

        except Exception as e:
            logger.error(f"Failed to extract files from {repo.name}: {e}")
            return []

    async def _extract_code_files(self, repo) -> List[Dict[str, Any]]:
        """Clone repo and extract Python/TypeScript/JavaScript files."""
        # This is now handled by _extract_all_files
        return []
