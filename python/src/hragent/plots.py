from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .analysis import AnalysisResult


@dataclass
class PlotPaths:
    signal_plot: Path
    hr_plot: Path
    hrv_psd_plot: Path | None = None


def _ensure_plots_dir(output_dir: Path) -> Path:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _compute_psd(rr_intervals: Sequence[float]) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if len(rr_intervals) < 4:
        return None
    times = np.cumsum(rr_intervals) / 1000.0
    duration = times[-1]
    if duration <= 0:
        return None
    fs = 4.0
    target_times = np.arange(0, duration, 1 / fs)
    if len(target_times) < 2:
        return None
    uniform_rr = np.interp(target_times, times, rr_intervals)
    detrended = uniform_rr - np.mean(uniform_rr)
    psd = np.abs(np.fft.rfft(detrended)) ** 2
    freq = np.fft.rfftfreq(len(uniform_rr), d=1 / fs)
    return freq, psd


def generate_plots(
    df,
    analysis: AnalysisResult,
    output_dir: Path,
) -> PlotPaths:
    plots_dir = _ensure_plots_dir(output_dir)
    timestamps = df["timestamp_ms"].to_numpy()
    raw_values = df["raw_value"].to_numpy()
    signal_plot = plots_dir / "ppg.png"
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(timestamps, raw_values, color="tab:blue", label="PPG")
    peak_indices = [idx for idx in analysis.hr.peak_indices if 0 <= idx < len(timestamps)]
    if peak_indices:
        peak_times = timestamps[peak_indices]
        peak_values = raw_values[peak_indices]
        ax.scatter(peak_times, peak_values, color="tab:red", label="Peaks")
    ax.set_title("PPG Signal with Detected Peaks")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Raw Value")
    ax.grid(True)
    ax.legend(loc="upper right")
    _save_fig(fig, signal_plot)

    hr_plot = plots_dir / "hr.png"
    rr = np.asarray(analysis.hr.rr_intervals_ms)
    hr_series = np.asarray(analysis.hr.hr_series_bpm)
    fig_hr, ax_hr = plt.subplots(figsize=(8, 3))
    if rr.size and hr_series.size:
        hr_time = np.cumsum(rr) / 1000.0
        ax_hr.plot(hr_time, hr_series, marker="o", linestyle="-", color="tab:green")
    ax_hr.set_title("Heart Rate Over Time")
    ax_hr.set_xlabel("Time (s)")
    ax_hr.set_ylabel("bpm")
    ax_hr.grid(True)
    _save_fig(fig_hr, hr_plot)

    psd_result = _compute_psd(rr.tolist()) if rr.size else None
    hrv_psd_plot: Path | None = None
    if psd_result is not None:
        freq, psd = psd_result
        hrv_psd_plot = plots_dir / "hrv_psd.png"
        fig_psd, ax_psd = plt.subplots(figsize=(8, 3))
        ax_psd.plot(freq, psd, color="tab:purple")
        ax_psd.set_xlim(0, 0.5)
        ax_psd.set_xlabel("Frequency (Hz)")
        ax_psd.set_ylabel("Power")
        ax_psd.set_title("HRV Power Spectral Density")
        ax_psd.grid(True)
        _save_fig(fig_psd, hrv_psd_plot)

    return PlotPaths(signal_plot=signal_plot, hr_plot=hr_plot, hrv_psd_plot=hrv_psd_plot)
