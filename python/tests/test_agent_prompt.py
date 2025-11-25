import numpy as np
import pandas as pd
from heartpy import load_exampledata
from pathlib import Path

from hragent.agent import generate_report_for_session


class StubModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.calls.append(prompt)
        return "# 心拍解析レポート\n\n## 概要\n概要サマリ"


def test_agent_prompt_includes_summary(tmp_path, monkeypatch):
    base_dir = tmp_path / "python_base"
    (base_dir / "data" / "sessions").mkdir(parents=True)
    (base_dir / "data" / "reports").mkdir(parents=True)
    data, _ = load_exampledata(0)
    timestamps = np.arange(len(data)) * 10
    session_path = base_dir / "data" / "sessions" / "session_01.csv"
    pd.DataFrame({"timestamp_ms": timestamps, "raw_value": data.astype(int)}).to_csv(
        session_path, index=False
    )
    monkeypatch.setenv("HRA_BASE_DIR", str(base_dir))
    monkeypatch.setenv("HRA_SAMPLE_RATE_HZ", "100.0")
    stub_model = StubModel()
    report = generate_report_for_session(session_path, model=stub_model)
    assert report.startswith("# 心拍解析レポート")
    assert stub_model.calls
    assert "平均心拍数" in stub_model.calls[0]
    assert "プロット一覧" in stub_model.calls[0]
