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
                rate_limit_delay: float - Delay in seconds between videos (default: 0)

        Returns:
            List of video content dictionaries
        """
        max_videos = kwargs.get("max_videos", None)
        rate_limit_delay = kwargs.get("rate_limit_delay", 0)

        # Use channel /videos tab instead of uploads playlist to bypass API limits
        logger.info(f"Fetching channel info from {channel_url}")

        # Ensure we use the /videos tab URL which doesn't have the 100-item limit
        if '/videos' not in channel_url:
            videos_url = f"{channel_url.rstrip('/')}/videos"
        else:
            videos_url = channel_url

        logger.info(f"Using videos URL: {videos_url}")

        # Extract videos from the channel /videos tab
        ydl_opts_videos = {
            'quiet': True,
            'ignoreerrors': True,  # Continue on errors
            'skip_download': True,  # Don't download videos
            'no_warnings': True,
            'extract_flat': True,  # Use flat extraction for channel /videos tab
            'lazy_playlist': False,  # Force complete extraction
            'extractor_args': {
                'youtube:tab': {
                    'skip': ['authcheck'],  # Skip auth check for better pagination
                }
            },
        }

        if max_videos:
            ydl_opts_videos['playlistend'] = max_videos
        else:
            # For all videos, ensure we don't stop early
            ydl_opts_videos['playliststart'] = 1

        videos = []

        with yt_dlp.YoutubeDL(ydl_opts_videos) as ydl:
            try:
                channel_info = ydl.extract_info(videos_url, download=False)

                if 'entries' not in channel_info:
                    logger.warning(f"No videos found in channel")
                    return []

                # Filter out None entries
                video_entries = [e for e in channel_info['entries'] if e and e.get('id')]

                logger.info(f"Found {len(video_entries)} videos to process (total available: {channel_info.get('playlist_count', len(video_entries))})")

                # Extract transcripts for each video
                import asyncio
                for idx, entry in enumerate(video_entries, 1):
                    video_id = entry.get('id')
                    if not video_id:
                        continue

                    video_url = f"https://www.youtube.com/watch?v={video_id}"

                    try:
                        logger.info(f"[{idx}/{len(video_entries)}] Processing {entry.get('title', video_url)[:60]}...")
                        transcript_data = await self._get_video_transcript(video_url)
                        if transcript_data:
                            videos.append(transcript_data)
                        else:
                            logger.warning(f"No transcript for {entry.get('title', video_url)}")
                    except Exception as e:
                        logger.error(f"Failed to get transcript for {video_url}: {e}")
                        continue

                    # Rate limiting delay
                    if rate_limit_delay > 0 and idx < len(video_entries):
                        await asyncio.sleep(rate_limit_delay)

                logger.info(f"Successfully extracted {len(videos)} transcripts out of {len(video_entries)} videos")
                return videos

            except Exception as e:
                logger.error(f"Failed to crawl playlist: {e}", exc_info=True)
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

                # Parse subtitle format (JSON3 or VTT) and extract text
                transcript = self._parse_subtitle(subtitle_text)

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

    def _parse_subtitle(self, subtitle_text: str) -> str:
        """Parse subtitle format (JSON3 or VTT) and extract clean text."""
        # Try JSON3 format first (YouTube's automatic captions)
        if subtitle_text.strip().startswith('{'):
            try:
                import json
                data = json.loads(subtitle_text)

                # Extract text from JSON3 events
                text_parts = []
                events = data.get('events', [])

                for event in events:
                    segs = event.get('segs', [])
                    for seg in segs:
                        text = seg.get('utf8', '')
                        if text and text != '\n':
                            text_parts.append(text)

                return ' '.join(text_parts)
            except json.JSONDecodeError:
                logger.warning("Failed to parse as JSON3, falling back to VTT")

        # Fall back to VTT format
        lines = subtitle_text.split('\n')
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
