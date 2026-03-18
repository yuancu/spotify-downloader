"""
SoundCloud module for downloading and searching songs.

Patched for fuzzy search: SoundCloud is user-generated content with inconsistent
metadata, so the standard matching thresholds (artist 70%, name 60%, time 25%)
are too strict. This version overrides `search()` with its own relaxed matching
based on title similarity and duration tolerance.
"""

import logging
import re
import unicodedata
from difflib import SequenceMatcher
from itertools import islice
from typing import Any, Dict, List, Optional

from soundcloud import SoundCloud as SoundCloudClient
from soundcloud.resource.track import Track

from spotdl.providers.audio.base import AudioProvider
from spotdl.types.result import Result
from spotdl.types.song import Song
from spotdl.utils.formatter import create_song_title

__all__ = ["SoundCloud"]

logger = logging.getLogger(__name__)


def _normalize(s: str) -> str:
    """Lowercase, strip accents, collapse whitespace, remove punctuation."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _title_similarity(a: str, b: str) -> float:
    """Return 0-100 similarity between two normalized strings."""
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio() * 100


class SoundCloud(AudioProvider):
    """
    SoundCloud audio provider class with fuzzy matching.
    """

    SUPPORTS_ISRC = False
    GET_RESULTS_OPTS: List[Dict[str, Any]] = [{}]

    # Duration tolerance: allow ±30s or ±20% (whichever is larger)
    DURATION_TOLERANCE_SEC = 30
    DURATION_TOLERANCE_PCT = 0.20

    # Minimum title similarity score to consider a match (0-100)
    MIN_TITLE_SIMILARITY = 50

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.client = SoundCloudClient()

    @staticmethod
    def _strip_noise(name: str) -> str:
        """Remove common Spotify metadata noise from a song name."""
        noise = [
            r"\s*-\s*Spotify Singles$",
            r"\s*-\s*Single$",
            r"\s*[\(\[](Remaster(ed)?|Deluxe|Bonus|Live|Acoustic|Demo|Anniversary"
            r"|Edition|Version|Radio Edit|Extended|Original Mix)[^\)\]]*[\)\]]",
            r"\s*\(feat\.?\s*[^\)]*\)",
            r"\s*ft\.?\s+\S+",
        ]
        for pat in noise:
            name = re.sub(pat, "", name, flags=re.IGNORECASE).strip()
        return name

    @staticmethod
    def _fuzzy_search_terms(search_term: str) -> List[str]:
        """Generate progressively simpler queries from Spotify-style search term."""
        terms = [search_term]

        cleaned = SoundCloud._strip_noise(search_term)
        if cleaned != search_term:
            terms.append(cleaned)

        parts = cleaned.split(" - ", 1)
        if len(parts) == 2:
            artist, title = parts[0].strip(), parts[1].strip()
            terms.append(f"{title} {artist}")
            terms.append(title)

        return list(dict.fromkeys(terms))

    def _duration_ok(self, song_duration_ms: int, result_duration_ms: int) -> bool:
        """Check if durations are within tolerance."""
        if song_duration_ms <= 0 or result_duration_ms <= 0:
            return True  # can't check, allow it
        diff = abs(song_duration_ms - result_duration_ms) / 1000.0
        threshold = max(
            self.DURATION_TOLERANCE_SEC,
            song_duration_ms / 1000.0 * self.DURATION_TOLERANCE_PCT,
        )
        return diff <= threshold

    def _score_result(self, song: Song, track: Track) -> float:
        """
        Score a SoundCloud track against a Spotify song.
        Returns 0-100. Higher is better. Returns 0 if definitely wrong.
        """
        song_title = self._strip_noise(song.name).lower()
        track_title = track.title.lower()

        # Title similarity (most important for SoundCloud)
        title_sim = _title_similarity(song_title, track_title)

        # Bonus if artist name appears in track title or uploader name
        artist_bonus = 0
        for artist in song.artists:
            artist_lower = artist.lower()
            if artist_lower in track_title or artist_lower in track.user.username.lower():
                artist_bonus = 15
                break

        # Duration penalty
        duration_ok = self._duration_ok(song.duration, track.full_duration)
        duration_penalty = 0 if duration_ok else -20

        score = title_sim + artist_bonus + duration_penalty
        return max(0, min(100, score))

    def search(self, song: Song, *_args, **_kwargs) -> Optional[str]:
        """
        Search SoundCloud with fuzzy matching, bypassing the strict
        base class matching that doesn't work well with UGC platforms.
        """
        search_query = create_song_title(song.name, song.artists).lower()
        logger.debug("[%s] SoundCloud fuzzy searching for: %s", song.song_id, search_query)

        seen_ids: set = set()
        candidates: List[tuple] = []  # (score, track)

        for query in self._fuzzy_search_terms(search_query):
            for result in islice(self.client.search(query), 20):
                if not isinstance(result, Track) or result.id in seen_ids:
                    continue
                seen_ids.add(result.id)

                # Skip preview-only tracks
                try:
                    if "/preview/" in result.media.transcodings[0].url:
                        continue
                except (IndexError, AttributeError):
                    continue

                score = self._score_result(song, result)
                logger.debug(
                    "[%s] SC candidate: %.1f - %s by %s (%dms) %s",
                    song.song_id, score, result.title,
                    result.user.username, result.full_duration,
                    result.permalink_url,
                )

                if score >= self.MIN_TITLE_SIMILARITY:
                    candidates.append((score, result))

        if not candidates:
            logger.debug("[%s] SoundCloud: no candidates above threshold", song.song_id)
            return None

        # Sort by score desc, then by playback count desc (prefer popular)
        candidates.sort(key=lambda x: (x[0], x[1].playback_count or 0), reverse=True)
        best_score, best_track = candidates[0]

        logger.info(
            "[%s] SoundCloud matched: %.1f%% - %s by %s → %s",
            song.song_id, best_score, best_track.title,
            best_track.user.username, best_track.permalink_url,
        )
        return best_track.permalink_url

    def get_results(self, search_term: str, *_args, **_kwargs) -> List[Result]:
        """
        Get results from SoundCloud with progressive fuzzy search.
        (Still used if base class search() is called for some reason.)
        """
        seen_ids: set = set()
        results: List[Track] = []

        for query in self._fuzzy_search_terms(search_term):
            for result in islice(self.client.search(query), 20):
                if isinstance(result, Track) and result.id not in seen_ids:
                    seen_ids.add(result.id)
                    results.append(result)

        simplified_results = []
        for result in results:
            try:
                if "/preview/" in result.media.transcodings[0].url:
                    continue
            except (IndexError, AttributeError):
                continue

            album = self.client.get_track_albums(result.id)
            try:
                album_name = next(album).title
            except StopIteration:
                album_name = None

            simplified_results.append(
                Result(
                    source="soundcloud",
                    url=result.permalink_url,
                    name=result.title,
                    verified=result.user.verified,
                    duration=result.full_duration,
                    author=result.user.username,
                    result_id=str(result.id),
                    isrc_search=False,
                    search_query=search_term,
                    views=result.playback_count,
                    explicit=False,
                    album=album_name,
                )
            )

        return simplified_results
