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
