import numpy as np
import pandas as pd
from heartpy import load_exampledata
from pathlib import Path

from hragent.analysis import analyze_session
from hragent.session_store import SessionMetadata


def test_analyze_session_handles_example_data():
    data, _ = load_exampledata(0)
    length = len(data)
    timestamps = np.arange(length) * 10
    df = pd.DataFrame({"timestamp_ms": timestamps, "raw_value": data.astype(int)})
    meta = SessionMetadata(
        path=Path("example"),
        n_samples=length,
        duration_sec=(timestamps[-1] - timestamps[0]) / 1000.0,
        sample_rate_hz=100.0,
        start_timestamp_ms=int(timestamps[0]),
        end_timestamp_ms=int(timestamps[-1]),
    )
    result = analyze_session(df, meta, 100.0)
    assert 50 < result.hr.mean_hr_bpm < 70
    assert result.hr.quality == "poor"
    assert result.hr.artifact_percentage > 90
    assert "lf" in result.hr.freq_domain and result.hr.freq_domain["lf"] > 0
    assert "sdnn" in result.hr.time_domain
