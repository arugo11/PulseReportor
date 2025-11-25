from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from heartpy import process as heartpy_process
from hrvanalysis import (
    get_frequency_domain_features,
    get_time_domain_features,
)

from .session_store import SessionMetadata


@dataclass
class HRMetrics:
    mean_hr_bpm: float
    min_hr_bpm: float
    max_hr_bpm: float
    hr_series_bpm: list[float]
    rr_intervals_ms: list[float]
    time_domain: dict[str, float]
    freq_domain: dict[str, float]
    artifact_percentage: float
    quality: str
    peak_indices: list[int] = field(default_factory=list)


@dataclass
class AnalysisResult:
    session: SessionMetadata
    hr: HRMetrics


class AnalysisError(Exception):
    pass


def _preprocess_signal(raw_values: Sequence[int]) -> np.ndarray:
    signal = np.asarray(raw_values, dtype=float)
    signal = signal - np.mean(signal)
    window = np.ones(5, dtype=float) / 5.0
    smooth = np.convolve(signal, window, mode="same")
    baseline = np.convolve(signal, np.ones(25, dtype=float) / 25.0, mode="same")
    processed = smooth - baseline
    if np.std(processed) > 0:
        processed = (processed - np.mean(processed)) / np.std(processed)
    return processed


def analyze_session(
    df,
    session_meta: SessionMetadata,
    expected_sample_rate_hz: float,
) -> AnalysisResult:
    raw = df["raw_value"].to_numpy(dtype=float)
    # 装着不良で0や1023付近に張り付いた区間を補間で埋める
    sat_low, sat_high = 5, 1018
    sat_mask = (raw <= sat_low) | (raw >= sat_high)
    masked_pct = 100.0 * sat_mask.sum() / max(1, len(raw))
    if sat_mask.all():
        raise AnalysisError("Signal is fully saturated")
    if sat_mask.any():
        idx = np.arange(len(raw))
        valid = ~sat_mask
        raw[sat_mask] = np.interp(idx[sat_mask], idx[valid], raw[valid])
    signal = _preprocess_signal(raw)
    sample_rate = session_meta.sample_rate_hz or expected_sample_rate_hz
    if expected_sample_rate_hz:
        if sample_rate < 0.5 * expected_sample_rate_hz or sample_rate > 1.5 * expected_sample_rate_hz:
            sample_rate = expected_sample_rate_hz
    try:
        working_data, measures = heartpy_process(signal, sample_rate)
    except Exception as exc:  # pragma: no cover - dependent on external signal
        raise AnalysisError("HeartPy failed to process signal") from exc

    rr_list = [
        float(rr) for rr in working_data.get("RR_list", [])
        if rr > 0 and 300.0 <= rr <= 2000.0  # 30–200 bpm plausible range
    ]
    if not rr_list:
        raise AnalysisError("RR intervals could not be extracted")

    hr_values = [60000.0 / rr for rr in rr_list]
    time_domain = get_time_domain_features(rr_list)
    freq_domain = get_frequency_domain_features(
        rr_list, sampling_frequency=max(4, int(expected_sample_rate_hz // 25) or 4)
    )

    artifact_pct = float(masked_pct)
    sample_rate_gap = abs(session_meta.sample_rate_hz - expected_sample_rate_hz)
    if expected_sample_rate_hz:
        drift_ratio = sample_rate_gap / expected_sample_rate_hz
    else:
        drift_ratio = 0.0
    poor_quality = artifact_pct > 10 or len(rr_list) < 5 or drift_ratio > 0.1
    quality = "poor" if poor_quality else "good"

    metrics = HRMetrics(
        mean_hr_bpm=float(time_domain.get("mean_hr", np.mean(hr_values))),
        min_hr_bpm=float(time_domain.get("min_hr", min(hr_values))),
        max_hr_bpm=float(time_domain.get("max_hr", max(hr_values))),
        hr_series_bpm=[float(hr) for hr in hr_values],
        rr_intervals_ms=rr_list,
        time_domain={k: float(v) for k, v in time_domain.items()},
        freq_domain={k: float(v) for k, v in freq_domain.items()},
        artifact_percentage=artifact_pct,
        quality=quality,
        peak_indices=[int(idx) for idx in working_data.get("peaklist", []) if isinstance(idx, int)],
    )
    result = AnalysisResult(session=session_meta, hr=metrics)
    return result
