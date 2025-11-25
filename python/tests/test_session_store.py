import pandas as pd
import pytest

from hragent.session_store import load_session


def test_load_session_metadata(tmp_path):
    rows = [
        {"timestamp_ms": 0, "raw_value": 512},
        {"timestamp_ms": 100, "raw_value": 520},
        {"timestamp_ms": 200, "raw_value": 500},
        {"timestamp_ms": 300, "raw_value": 530},
        {"timestamp_ms": 400, "raw_value": 515},
    ]
    path = tmp_path / "session.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    df, meta = load_session(path)
    assert len(df) == 5
    assert meta.n_samples == 5
    assert meta.start_timestamp_ms == 0
    assert meta.end_timestamp_ms == 400
    assert meta.duration_sec == pytest.approx(0.4)
    assert meta.sample_rate_hz == pytest.approx(5 / 0.4)
