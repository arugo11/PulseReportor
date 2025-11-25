from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SessionMetadata:
    path: Path
    n_samples: int
    duration_sec: float
    sample_rate_hz: float
    start_timestamp_ms: int
    end_timestamp_ms: int


def load_session(path: Path) -> tuple[pd.DataFrame, SessionMetadata]:
    raw_df = pd.read_csv(path, dtype=str)
    df = raw_df.copy()
    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    df["raw_value"] = pd.to_numeric(df["raw_value"], errors="coerce")
    df = df.dropna(subset=["timestamp_ms", "raw_value"])
    df = df[(df["timestamp_ms"] >= 0) & (df["timestamp_ms"] <= 1_000_000_000)]
    df = df[(df["raw_value"] >= 0) & (df["raw_value"] <= 1023)]
    if df.empty:
        raise ValueError("Session file contains no valid samples after cleaning")
    dropped = len(raw_df) - len(df)
    if dropped:
        logger.warning("Dropped %d malformed samples from %s", dropped, path)
    df["timestamp_ms"] = df["timestamp_ms"].astype(int)
    df["raw_value"] = df["raw_value"].astype(int)
    if df.empty:
        raise ValueError("Session file contains no samples")
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    start = int(df["timestamp_ms"].iloc[0])
    end = int(df["timestamp_ms"].iloc[-1])
    duration = max((end - start) / 1000.0, 1e-3)
    n_samples = len(df)
    diffs = df["timestamp_ms"].diff().dropna()
    median_gap = float(diffs.median()) if not diffs.empty else 0.0
    median_rate = 1000.0 / median_gap if median_gap > 0 else 0.0
    sample_rate = median_rate if median_rate > 0 else (n_samples / duration if duration > 0 else 0.0)
    meta = SessionMetadata(
        path=path,
        n_samples=n_samples,
        duration_sec=duration,
        sample_rate_hz=sample_rate,
        start_timestamp_ms=start,
        end_timestamp_ms=end,
    )
    return df, meta
