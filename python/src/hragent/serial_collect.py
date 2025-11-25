from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import serial

from .config import SerialConfig

logger = logging.getLogger(__name__)

MAX_TIMESTAMP_MS = 1_000_000_000  # guard against corrupted concatenated numbers
MAX_RAW_VALUE = 1023
START_TIMESTAMP_MAX = 5_000  # discard stale buffered lines until we see a fresh reset (<5s)
PROGRESS_INTERVAL_SEC = 10.0

@dataclass
class RecordingResult:
    output_path: Path
    n_samples: int
    duration_sec: float
    sample_rate_hz: float


def record_session(
    serial_cfg: SerialConfig,
    duration_sec: float | None,
    output_path: Path,
) -> RecordingResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_monotonic = time.monotonic()
    last_progress_log = start_monotonic
    consecutive_timeouts = 0
    samples = 0
    started = False
    last_ts = None

    with serial.Serial(
        port=serial_cfg.port,
        baudrate=serial_cfg.baudrate,
        timeout=serial_cfg.timeout,
    ) as ser, output_path.open("w", newline="", encoding="utf-8") as file:
        ser.reset_input_buffer()
        writer = csv.writer(file)
        writer.writerow(["timestamp_ms", "raw_value"])
        try:
            while True:
                if duration_sec and (time.monotonic() - start_monotonic) >= duration_sec:
                    break
                raw_line = ser.readline()
                if not raw_line:
                    consecutive_timeouts += 1
                    logger.warning("Serial timeout %d/5", consecutive_timeouts)
                    if consecutive_timeouts >= 5:
                        raise RuntimeError("Too many serial read timeouts, aborting session")
                    continue
                consecutive_timeouts = 0
                try:
                    line = raw_line.decode("utf-8").strip()
                except UnicodeDecodeError:
                    line = raw_line.decode("latin1", errors="ignore").strip()
                if not line:
                    continue
                if "," not in line:
                    logger.warning("Malformed line (missing comma): %r", line)
                    continue
                timestamp_str, value_str = line.split(",", 1)
                try:
                    timestamp_ms = int(timestamp_str.strip())
                    raw_value = int(value_str.strip())
                except ValueError:
                    logger.warning("Malformed values at line %d: %r", samples + 1, line)
                    continue
                if not started:
                    # Skip any stale buffered lines from before the board reset.
                    if timestamp_ms > START_TIMESTAMP_MAX:
                        continue
                    started = True
                    start_monotonic = time.monotonic()
                if last_ts is not None and timestamp_ms < last_ts:
                    logger.warning("Timestamp went backwards at line %d: %r", samples + 1, line)
                    continue
                last_ts = timestamp_ms
                if not (0 <= raw_value <= MAX_RAW_VALUE):
                    logger.warning("Raw value out of range at line %d: %r", samples + 1, line)
                    continue
                if not (0 <= timestamp_ms <= MAX_TIMESTAMP_MS):
                    logger.warning("Timestamp out of range at line %d: %r", samples + 1, line)
                    continue
                writer.writerow([timestamp_ms, raw_value])
                samples += 1
                now = time.monotonic()
                if now - last_progress_log >= PROGRESS_INTERVAL_SEC:
                    elapsed = now - start_monotonic
                    rate = samples / elapsed if elapsed > 0 else 0.0
                    logger.info("Recording... %d samples (%.1f Hz)", samples, rate)
                    last_progress_log = now
        except KeyboardInterrupt:
            logger.info("Recording interrupted by user")
    duration = time.monotonic() - start_monotonic
    effective_rate = samples / duration if duration > 0 else 0.0
    return RecordingResult(
        output_path=output_path,
        n_samples=samples,
        duration_sec=duration,
        sample_rate_hz=effective_rate,
    )
