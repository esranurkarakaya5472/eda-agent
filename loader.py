"""tools/loader.py — Dataset loading and basic validation."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads a CSV file and performs lightweight validation."""

    SUPPORTED_EXTENSIONS = {".csv"}

    def __init__(self, encoding: str = "utf-8") -> None:
        self._encoding = encoding

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, file_path: str | Path) -> pd.DataFrame:
        path = Path(file_path)
        self._validate_path(path)

        df = pd.read_csv(path, encoding=self._encoding, low_memory=False)
        self._validate_dataframe(df, path)

        logger.info("Dataset loaded: %s (%d rows × %d cols)", path.name, *df.shape)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_path(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() not in DatasetLoader.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame, path: Path) -> None:
        if df.empty:
            raise ValueError(f"Empty dataset: {path}")
        if df.columns.duplicated().any():
            dupes = df.columns[df.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate column names detected: {dupes}")
