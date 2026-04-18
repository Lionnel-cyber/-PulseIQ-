"""Abstract base class for all PulseIQ data source connectors.

Every connector inherits from BaseConnector and must implement fetch().
The module-level ``http_retry`` decorator should be applied to any method
that makes an outbound HTTP call.
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared retry decorator — import and apply to HTTP call methods in subclasses
# ---------------------------------------------------------------------------
http_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


class BaseConnector(ABC):
    """Abstract base class for all PulseIQ data source connectors.

    Subclasses must implement ``fetch()`` and should decorate any method
    that makes an HTTP call with ``@http_retry`` from this module.

    Args:
        raw_data_root: Root directory for raw data storage.
            Defaults to ``"data/raw"`` (relative to CWD). Override in tests
            by passing a ``tmp_path`` fixture value.

    Example::

        from src.connectors.base_connector import BaseConnector, http_retry

        class MyConnector(BaseConnector):
            @http_retry
            def _call_api(self) -> dict:
                ...

            def fetch(self) -> pd.DataFrame:
                raw = self._call_api()
                self.save_raw(raw, "my_source")
                ...
    """

    def __init__(self, raw_data_root: str | Path = "data/raw") -> None:
        self._raw_data_root = Path(raw_data_root)
        self._logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Fetch data from the source and return as a DataFrame.

        Implementations must:
        1. Call the external API (decorated with ``@http_retry``).
        2. Validate the raw response with a Pydantic model.
        3. Call ``save_raw()`` before returning.
        4. Return a ``pd.DataFrame`` with a fully documented column schema.

        Returns:
            DataFrame with source-specific columns documented in the subclass.

        Raises:
            tenacity.RetryError: Propagated after 3 failed HTTP attempts.
            pydantic.ValidationError: If the API response fails schema validation.
        """

    def save_raw(self, data: dict[str, Any], source_name: str) -> Path:
        """Persist a raw API response to the partitioned data lake.

        Writes the payload as JSON to::

            {raw_data_root}/YYYY/MM/DD/{source_name}/data.json

        Directories are created automatically. Existing files are overwritten
        so re-runs on the same day always reflect the latest fetch.

        Args:
            data: Raw response dict to serialize. Non-JSON-serialisable values
                (e.g. ``datetime`` objects) are coerced to strings via
                ``default=str``.
            source_name: Identifier for the data source (e.g. ``"bls"``,
                ``"fred"``). Used as the leaf directory name.

        Returns:
            Absolute ``Path`` to the written ``data.json`` file.

        Raises:
            OSError: If the target directory cannot be created or the file
                cannot be written.
        """
        today = date.today()
        output_dir = (
            self._raw_data_root
            / str(today.year)
            / f"{today.month:02d}"
            / f"{today.day:02d}"
            / source_name
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "data.json"
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)

        self._logger.info("Saved raw %s data → %s", source_name, output_path)
        return output_path
