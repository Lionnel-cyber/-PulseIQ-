"""Reddit connector for PulseIQ.

Fetches recent posts from financial distress subreddits using PRAW
(Python Reddit API Wrapper) and returns a tidy DataFrame ready for
the dbt staging layer.

Default subreddits:
    r/personalfinance  — personal finance questions and discussions
    r/povertyfinance   — discussions about managing finances in poverty
"""

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import praw
from pydantic import BaseModel

from src.connectors.base_connector import BaseConnector, http_retry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic validation model
# ---------------------------------------------------------------------------


class RedditPost(BaseModel):
    """A single Reddit post extracted from a PRAW Submission object.

    All fields are stored in their native types as returned by the PRAW API.
    Type conversion (e.g. ``created_utc`` → ``datetime``) happens during
    DataFrame construction, not here.

    Attributes:
        id: Reddit post ID (e.g. ``"abc123"``).
        title: Post title text.
        selftext: Post body text. Empty string ``""`` for link posts or
            posts without body text.
        score: Net upvote count at time of fetch.
        created_utc: Post creation time as a UTC Unix timestamp (float).
        subreddit: Subreddit display name (e.g. ``"personalfinance"``).
    """

    id: str
    title: str
    selftext: str
    score: int
    created_utc: float
    subreddit: str


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class RedditConnector(BaseConnector):
    """Fetches posts from financial subreddits via the Reddit API (PRAW).

    Inherits retry logic and raw data persistence from ``BaseConnector``.
    For each subreddit, the top 100 posts from the past 24 hours are fetched,
    validated with Pydantic, persisted as raw JSON, and returned as a
    tidy DataFrame.

    Args:
        raw_data_root: Root directory for raw data storage.
            Defaults to ``"data/raw"``. Pass ``tmp_path`` in tests.

    Raises:
        ValueError: At construction time if any required Reddit API
            credentials are not set.

    Example::

        connector = RedditConnector()
        df = connector.fetch()
        df = connector.fetch(["personalfinance", "financialindependence"])
    """

    DEFAULT_SUBREDDITS: list[str] = ["personalfinance", "povertyfinance"]

    def __init__(self, raw_data_root: str | Path = "data/raw") -> None:
        super().__init__(raw_data_root)

        self._client_id: str = os.getenv("REDDIT_CLIENT_ID") or ""
        self._client_secret: str = os.getenv("REDDIT_CLIENT_SECRET") or ""
        self._user_agent: str = os.getenv("REDDIT_USER_AGENT") or ""

        if (
            not self._client_id
            or not self._client_secret
            or not self._user_agent
        ):
            raise ValueError(
                "REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and "
                "REDDIT_USER_AGENT environment variables must be set. "
                "Add them to your .env file."
            )

        self._reddit = praw.Reddit(
            client_id=self._client_id,
            client_secret=self._client_secret,
            user_agent=self._user_agent,
        )

    # ------------------------------------------------------------------
    # Private fetch method — decorated with shared retry policy
    # ------------------------------------------------------------------

    @http_retry
    def _fetch_subreddit_posts(
        self, subreddit_name: str
    ) -> list[dict[str, Any]]:
        """Fetch the top 100 posts from the past 24 hours for one subreddit.

        The PRAW ``top()`` call returns a lazy generator; iterating it here
        inside the retry-decorated method ensures that any network error
        during iteration causes tenacity to retry the entire fetch.

        Args:
            subreddit_name: Subreddit display name without the ``r/`` prefix
                (e.g. ``"personalfinance"``).

        Returns:
            List of raw post dicts, each containing the fields required by
            ``RedditPost``.

        Raises:
            prawcore.exceptions.ResponseException: On API errors.
                Tenacity retries this up to 3 times before re-raising.
        """
        self._logger.debug(
            "Fetching top posts from r/%s (last 24 h)", subreddit_name
        )
        submissions = self._reddit.subreddit(subreddit_name).top(
            time_filter="day", limit=100
        )
        return [
            {
                "id": s.id,
                "title": s.title,
                "selftext": s.selftext,
                "score": s.score,
                "created_utc": s.created_utc,
                "subreddit": s.subreddit.display_name,
            }
            for s in submissions
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch(self, subreddits: list[str] | None = None) -> pd.DataFrame:
        """Fetch recent posts from one or more subreddits.

        For each subreddit this method:

        1. Fetches the top 100 posts from the past 24 hours via
           ``_fetch_subreddit_posts()``.
        2. Validates each post dict with ``RedditPost``.
        3. Persists all raw post data together via ``save_raw()``.
        4. Builds DataFrame rows:

           - ``date`` from ``created_utc`` (UTC Unix timestamp → naive datetime)
           - ``text`` from ``title + " " + selftext`` (empty selftext handled)

        Args:
            subreddits: List of subreddit display names to fetch from.
                Defaults to ``DEFAULT_SUBREDDITS``
                (``["personalfinance", "povertyfinance"]``) if ``None``.

        Returns:
            DataFrame with columns:

            - ``date``      — ``datetime64[ns]`` (UTC, timezone-naive)
            - ``subreddit`` — ``str``
            - ``post_id``   — ``str``
            - ``text``      — ``str`` (title and body concatenated)
            - ``score``     — ``int64``

        Raises:
            prawcore.exceptions.ResponseException: After 3 failed attempts.
            pydantic.ValidationError: If a post fails schema validation.
            ValueError: If required env vars are not set (raised at init).
        """
        names: list[str] = (
            subreddits if subreddits is not None else self.DEFAULT_SUBREDDITS
        )

        all_raw: dict[str, list[dict[str, Any]]] = {}
        all_posts: dict[str, list[RedditPost]] = {}

        for subreddit_name in names:
            raw_posts = self._fetch_subreddit_posts(subreddit_name)
            all_raw[subreddit_name] = raw_posts

            validated = [
                RedditPost.model_validate(post) for post in raw_posts
            ]
            all_posts[subreddit_name] = validated
            self._logger.info(
                "Fetched %d posts from r/%s", len(validated), subreddit_name
            )

        self.save_raw(all_raw, source_name="reddit")

        frames: list[pd.DataFrame] = []
        for subreddit_name, posts in all_posts.items():
            rows = [
                {
                    "date": pd.to_datetime(post.created_utc, unit="s"),
                    "subreddit": post.subreddit,
                    "post_id": post.id,
                    "text": " ".join(
                        filter(None, [post.title, post.selftext])
                    ),
                    "score": post.score,
                }
                for post in posts
            ]
            if rows:
                frames.append(pd.DataFrame(rows))

        if not frames:
            return pd.DataFrame(
                columns=["date", "subreddit", "post_id", "text", "score"]
            )

        return pd.concat(frames, ignore_index=True)
