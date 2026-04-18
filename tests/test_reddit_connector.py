"""Tests for src/connectors/reddit_connector.py.

PRAW is mocked with ``unittest.mock.MagicMock`` — no live Reddit API calls.
``monkeypatch`` injects dummy Reddit credentials for every test that
instantiates the connector. ``praw.Reddit`` is patched at the module level
inside the connector so that ``__init__`` receives the mock instance.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.connectors.reddit_connector import RedditConnector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PATCH_TARGET = "src.connectors.reddit_connector.praw.Reddit"

# Fixed UTC timestamp: 2024-01-01 00:00:00 UTC
_CREATED_UTC = 1704067200.0


def _make_mock_submission(
    post_id: str = "abc123",
    title: str = "Help with debt",
    selftext: str = "I have $5000 in credit card debt",
    score: int = 150,
    created_utc: float = _CREATED_UTC,
    subreddit_name: str = "personalfinance",
) -> MagicMock:
    """Build a MagicMock that mimics a PRAW Submission object."""
    m = MagicMock()
    m.id = post_id
    m.title = title
    m.selftext = selftext
    m.score = score
    m.created_utc = created_utc
    m.subreddit.display_name = subreddit_name
    return m


def _make_connector(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    submissions: list[MagicMock] | None = None,
) -> tuple["RedditConnector", MagicMock]:
    """Create a RedditConnector with PRAW fully mocked.

    Returns:
        ``(connector, mock_reddit_instance)`` so tests can inspect calls.
    """
    monkeypatch.setenv("REDDIT_CLIENT_ID", "test_id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "test_secret")
    monkeypatch.setenv("REDDIT_USER_AGENT", "pulseiq-test/1.0")

    with patch(_PATCH_TARGET) as MockReddit:
        mock_reddit = MockReddit.return_value
        mock_reddit.subreddit.return_value.top.return_value = (
            submissions if submissions is not None else []
        )
        connector = RedditConnector(raw_data_root=tmp_path)
        # Store mock on connector so the patched reddit persists after `with`
        connector._reddit = mock_reddit

    return connector, mock_reddit


# ---------------------------------------------------------------------------
# Tests: missing credentials
# ---------------------------------------------------------------------------


def test_raises_if_client_id_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ValueError raised at init if REDDIT_CLIENT_ID is not set."""
    saved_id = os.environ.pop("REDDIT_CLIENT_ID", None)
    saved_secret = os.environ.pop("REDDIT_CLIENT_SECRET", None)
    saved_user_agent = os.environ.pop("REDDIT_USER_AGENT", None)
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "test_secret")
    monkeypatch.setenv("REDDIT_USER_AGENT", "pulseiq-test/1.0")
    try:
        with pytest.raises(ValueError, match="REDDIT_CLIENT_ID"):
            RedditConnector()
    finally:
        if saved_id is not None:
            os.environ["REDDIT_CLIENT_ID"] = saved_id
        if saved_secret is not None:
            os.environ["REDDIT_CLIENT_SECRET"] = saved_secret
        if saved_user_agent is not None:
            os.environ["REDDIT_USER_AGENT"] = saved_user_agent


def test_raises_if_client_secret_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ValueError raised at init if REDDIT_CLIENT_SECRET is not set."""
    saved_id = os.environ.pop("REDDIT_CLIENT_ID", None)
    saved_secret = os.environ.pop("REDDIT_CLIENT_SECRET", None)
    saved_user_agent = os.environ.pop("REDDIT_USER_AGENT", None)
    monkeypatch.setenv("REDDIT_CLIENT_ID", "test_id")
    monkeypatch.setenv("REDDIT_USER_AGENT", "pulseiq-test/1.0")
    try:
        with pytest.raises(ValueError, match="REDDIT_CLIENT_SECRET"):
            RedditConnector()
    finally:
        if saved_id is not None:
            os.environ["REDDIT_CLIENT_ID"] = saved_id
        if saved_secret is not None:
            os.environ["REDDIT_CLIENT_SECRET"] = saved_secret
        if saved_user_agent is not None:
            os.environ["REDDIT_USER_AGENT"] = saved_user_agent


def test_raises_if_user_agent_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ValueError raised at init if REDDIT_USER_AGENT is not set."""
    saved_id = os.environ.pop("REDDIT_CLIENT_ID", None)
    saved_secret = os.environ.pop("REDDIT_CLIENT_SECRET", None)
    saved_user_agent = os.environ.pop("REDDIT_USER_AGENT", None)
    monkeypatch.setenv("REDDIT_CLIENT_ID", "test_id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "test_secret")
    try:
        with pytest.raises(ValueError, match="REDDIT_USER_AGENT"):
            RedditConnector()
    finally:
        if saved_id is not None:
            os.environ["REDDIT_CLIENT_ID"] = saved_id
        if saved_secret is not None:
            os.environ["REDDIT_CLIENT_SECRET"] = saved_secret
        if saved_user_agent is not None:
            os.environ["REDDIT_USER_AGENT"] = saved_user_agent


# ---------------------------------------------------------------------------
# Tests: DataFrame schema and correctness
# ---------------------------------------------------------------------------


def test_successful_fetch_returns_correct_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """fetch() returns a DataFrame with the documented column schema and dtypes."""
    submission = _make_mock_submission()
    connector, mock_reddit = _make_connector(monkeypatch, tmp_path, [submission])
    mock_reddit.subreddit.return_value.top.return_value = [submission]

    df = connector.fetch(["personalfinance"])

    assert list(df.columns) == ["date", "subreddit", "post_id", "text", "score"]
    assert len(df) == 1
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_integer_dtype(df["score"])
    assert df["subreddit"].iloc[0] == "personalfinance"
    assert df["post_id"].iloc[0] == "abc123"
    assert df["score"].iloc[0] == 150


def test_text_concatenation_title_and_selftext(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """text column = title + space + selftext when selftext is non-empty."""
    submission = _make_mock_submission(
        title="Need help", selftext="Lost my job last week"
    )
    connector, mock_reddit = _make_connector(monkeypatch, tmp_path, [submission])
    mock_reddit.subreddit.return_value.top.return_value = [submission]

    df = connector.fetch(["personalfinance"])

    assert df["text"].iloc[0] == "Need help Lost my job last week"


def test_text_concatenation_empty_selftext(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """text column = title only (no trailing space) when selftext is empty."""
    submission = _make_mock_submission(title="Title only", selftext="")
    connector, mock_reddit = _make_connector(monkeypatch, tmp_path, [submission])
    mock_reddit.subreddit.return_value.top.return_value = [submission]

    df = connector.fetch(["personalfinance"])

    assert df["text"].iloc[0] == "Title only"


def test_date_parsed_from_created_utc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """date column is correctly parsed from UTC Unix timestamp."""
    submission = _make_mock_submission(created_utc=_CREATED_UTC)
    connector, mock_reddit = _make_connector(monkeypatch, tmp_path, [submission])
    mock_reddit.subreddit.return_value.top.return_value = [submission]

    df = connector.fetch(["personalfinance"])

    assert df["date"].iloc[0] == pd.Timestamp("2024-01-01 00:00:00")


def test_fetches_multiple_subreddits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """fetch() collects posts from each subreddit and returns them all."""
    sub_pf = _make_mock_submission(post_id="pf1", subreddit_name="personalfinance")
    sub_pov = _make_mock_submission(post_id="pov1", subreddit_name="povertyfinance")

    monkeypatch.setenv("REDDIT_CLIENT_ID", "test_id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "test_secret")
    monkeypatch.setenv("REDDIT_USER_AGENT", "pulseiq-test/1.0")

    with patch(_PATCH_TARGET) as MockReddit:
        mock_reddit = MockReddit.return_value

        def top_side_effect(*_args, **kwargs):
            # Return the right submissions based on which subreddit was called
            return mock_reddit._current_submissions

        mock_reddit.subreddit.return_value.top.side_effect = top_side_effect
        connector = RedditConnector(raw_data_root=tmp_path)
        connector._reddit = mock_reddit

    # Configure side-effect to return different posts per subreddit
    call_count = [0]
    submissions_by_call = [[sub_pf], [sub_pov]]

    def top_by_call(**kwargs: object) -> list[MagicMock]:
        result = submissions_by_call[call_count[0]]
        call_count[0] += 1
        return result

    mock_reddit.subreddit.return_value.top.side_effect = top_by_call

    df = connector.fetch(["personalfinance", "povertyfinance"])

    assert len(df) == 2
    assert set(df["post_id"]) == {"pf1", "pov1"}
    assert set(df["subreddit"]) == {"personalfinance", "povertyfinance"}


def test_empty_subreddit_returns_empty_dataframe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """fetch() returns an empty DataFrame with correct columns when no posts exist."""
    connector, mock_reddit = _make_connector(monkeypatch, tmp_path, submissions=[])
    mock_reddit.subreddit.return_value.top.return_value = []

    df = connector.fetch(["personalfinance"])

    assert list(df.columns) == ["date", "subreddit", "post_id", "text", "score"]
    assert len(df) == 0
