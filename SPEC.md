
# Heart Rate Report Agent - SPEC

## 1. Overview

This project implements a one-shot heart rate analysis and reporting pipeline.

1. Arduino + Pulse Sensor acquires PPG signal during a measurement session.
2. Arduino streams timestamped samples via USB serial to a host PC.
3. A Python CLI tool records the stream into a CSV file per session.
4. A Python analysis module uses HeartPy and an HRV library to compute heart rate and HRV features.
5. An AI agent (LangGraph + LangChain + OpenAI API) generates a human readable report (Markdown, optionally PDF) from the analysis results.

The implementation should be modular so that:

- The Arduino sketch can be replaced without touching Python.
- The Python collector can be reused for different biosensors as long as they follow the same line format.
- The analysis and agent modules can be unit tested without actual hardware.

回路構成は Pulse Sensor と Arduino を `5V`/`GND`/`A0` で直接接続するだけの極力シンプルなものであり、信号処理はすべて Python 側 (analysis.py) で行う。

---

## 2. Requirements

### 2.1 Functional requirements

- Acquire PPG signal from a Pulse Sensor (SEN-11574 compatible) using Arduino Uno (or equivalent 5V board).
- Stream samples over serial in the format:

  - ASCII line: `<timestamp_ms>,<raw_value>\n`
  - `timestamp_ms`: unsigned 32bit integer from `millis()` since Arduino boot, in milliseconds.
  - `raw_value`: integer 0–1023 from `analogRead()`.
  - PC 側はポートオープン時にシリアルバッファをクリアし、タイムスタンプが 5 秒未満の行が来るまで古い行を破棄する。`raw_value` が 0–1023 以外、`timestamp_ms` が 0–1e9 以外、タイムスタンプ逆行、カンマ欠落行はスキップする。

- On the PC side, record a contiguous measurement session as a CSV file:

  - Header: `timestamp_ms,raw_value`
  - One row per sample.

- Analysis:

  - Detect beats and compute basic heart rate metrics:
    - Mean HR, min HR, max HR, HR time series.
  - Extract RR intervals.
  - Compute standard HRV features from RR intervals:
    - Time domain (mean RR, SDNN, RMSSD, pNN50, etc.).
    - Frequency domain (LF, HF, LF/HF if feasible).
  - Compute simple signal quality indicators (artifact ratio, dropped samples, etc.).

- Reporting:

  - Generate a Markdown report summarizing:
    - Measurement metadata (date, duration, sampling rate, sensor type).
    - Basic statistics (heart rate, HRV).
    - Short interpretation text, explicitly non medical diagnosis.
    - Plots: PPG with detected peaks, HR over time, optional HRV spectrum.
  - Optionally render Markdown to PDF (if a PDF backend is available).

- AI agent:

  - Implement an agent using LangGraph + LangChain that:
    - Accepts a session CSV path as input.
    - Calls analysis tools to compute metrics and generate plots.
    - Generates a structured report in Markdown using OpenAI API.
    - Optionally performs one reflection loop to refine the report.

### 2.2 Non functional requirements

- Python 3.11+.
- OS: Windows 11 or macOS, Linux should also work.
- Handle at least 10 minutes of recording at 100 Hz without data loss.
- Simple CLI interface, no GUI required.
- Code should be structured to allow unit tests for analysis and agent logic.

---

## 3. Project structure

Repository root (Python part):

```text
.
├── arduino/
│   └── pulse_logger/
│       └── pulse_logger.ino
├── python/
│   ├── pyproject.toml
│   ├── README.md
│   ├── SPEC.md
│   ├── .env.sample
│   ├── src/
│   │   └── hragent/
│   │       ├── __init__.py
│       │   ├── config.py
│       │   ├── serial_collect.py
│       │   ├── session_store.py
│       │   ├── analysis.py
│       │   ├── plots.py
│       │   ├── report_template.md
│       │   ├── agent.py
│       │   └── cli.py
│   ├── data/
│   │   ├── sessions/
│   │   └── reports/
│   └── tests/
│       ├── test_analysis.py
│       ├── test_session_store.py
│       └── test_agent_prompt.py
└── SPEC.md
```

`arduino/` 以下は Arduino IDE 用のスケッチ, `python/` 以下が Python パッケージ.

---

## 4. Hardware and Arduino specification

### 4.1 Hardware

- Microcontroller: Arduino Uno (ATmega328P) or compatible 5V board.
    
- Sensor: Switch-Science SKU 1135 (Pulse Sensor SEN-11574 compatible) heart rate sensor module.
    
- Connections: limited to the three wires below so the wiring stays minimal.
    
    - Sensor `+` → Arduino `5V`.
        
    - Sensor `-` → Arduino `GND`.
        
    - Sensor `S` → Arduino `A0` (Analog input 0).
        
- Do not add any external analog circuitry such as op-amps, RC filters, or comparators; the pulse sensor output is captured directly by the Arduino ADC.
    
- The hardware side does not perform any filtering or gain adjustments; the sensor signal is read as-is with `analogRead()` and any noise reduction, bandpass filtering, normalization, or smoothing is handled in Python (analysis.py).

### 4.2 Sampling parameters

- Target sampling interval: 10 ms (100 Hz).
    
- Timing:
    
    - Use `millis()` for timing, not `delay()` in a tight loop, to reduce jitter.
        
    - Sampling loop should try to keep constant interval (`sample_interval_ms`).
        

### 4.3 Arduino sketch responsibilities

The sketch is responsible only for sampling the pulse sensor and streaming timestamps; it must not perform BPM calculation, peak detection, filtering, or any other signal processing.
    
- Initialize serial at 115200 baud.
    
- Initialize sensor pin as analog input.
    
- In `loop()`:
    
    - Check elapsed time from the previous sample, and if `>= sample_interval_ms`:
        
        - Read the raw analog value via `analogRead()`.
            
        - Get `now_ms = millis()`.
            
        - Print `<now_ms>,<value>\n` to serial.
            
- The sketch must not block on long operations.
    

### 4.4 Arduino configurable parameters

In `pulse_logger.ino` define configuration constants:

```cpp
const int PULSE_SENSOR_PIN = A0;
const unsigned long SAMPLE_INTERVAL_MS = 10;  // 100Hz
const long SERIAL_BAUD_RATE = 115200;
```

Codex should implement them as `const` variables at the top of the sketch.

### 4.5 必要部品

- 必須部品:
    
    - Arduino Uno または互換の 5V 動作ボード.
        
    - Switch-Science SKU 1135 (Pulse Sensor SEN-11574 互換) モジュール.
        
    - ジャンパワイヤ（オス-オスなど）数本.
        
    - USB ケーブル（Arduino と PC 接続用）.
        
- 任意部品:
    
    - センサ取り付け用のバンドやテープなどの固定具 (電子部品ではなく物理的な補助).
        
---

## 5. Serial protocol and PC side assumptions

### 5.1 Serial settings

- Baud rate: 115200.
    
- Data bits: 8.
    
- Parity: None.
    
- Stop bits: 1.
    
- Flow control: None.
    

### 5.2 Line format

- Each sample is a single ASCII line:
    
    - Example: `123456,512\n`.
        
- No additional debug prints.
    
- PC side code must ignore empty lines and malformed lines but log them.
    

### 5.3 Port configuration

`python/src/hragent/config.py` should define:

```python
from pydantic import BaseModel

class SerialConfig(BaseModel):
    port: str  # e.g. "COM3" or "/dev/ttyACM0"
    baudrate: int = 115200
    timeout: float = 1.0  # seconds

class Paths(BaseModel):
    base_dir: Path
    sessions_dir: Path
    reports_dir: Path

class AppConfig(BaseModel):
    serial: SerialConfig
    paths: Paths
    sample_rate_hz: float = 100.0
```

Config should be loaded from:

- `.env` file using `python-dotenv` or
    
- default values plus command line overrides.
    

---

## 6. Python: data collection module

`serial_collect.py` implements serial recording logic.

### 6.1 External dependencies

- `pyserial` for serial communication.
    
- `pandas` for CSV writing (or standard `csv` module).
    
- `rich` or `logging` for console logging (optional).
    

### 6.2 Core API

Implement:

```python
def record_session(
    serial_cfg: SerialConfig,
    duration_sec: float | None,
    output_path: Path,
) -> None:
    """
    Open serial port and record heart rate samples into a CSV file.

    If duration_sec is None, record until user presses Ctrl+C.
    CSV format: timestamp_ms,raw_value
    """
```

Behavior:

- Open serial port with given configuration.
    
- Create output CSV and write header.
    
- Loop:
    
    - Read `line = ser.readline()`.
        
    - Decode as UTF-8 or ASCII, strip whitespace.
        
    - If line is empty, continue.
        
    - Split on `,` into `timestamp_ms`, `raw_value`.
        
    - Validate both as integers, skip malformed lines with a warning.
        
    - Append row to CSV.
        
- Stop when:
    
    - Elapsed real time >= `duration_sec`, or
        
    - User sends KeyboardInterrupt.
        
- Ensure serial port and file are closed via `with` context or `try/finally`.
Implementation notes:
- Port open後に `reset_input_buffer()` を呼び、タイムスタンプが 5 秒未満の行が来るまで古いバッファを破棄する。
- `raw_value` は 0–1023、`timestamp_ms` は 0–1e9 の範囲外、カンマ欠落、逆行タイムスタンプはスキップ。
- 10秒ごとにINFOログで進捗（サンプル数と実効Hz）を表示。
    

### 6.3 CLI behavior

Expose CLI via `cli.py` using `typer` or `argparse`:

```bash
python -m hragent.collect \
  --port COM3 \
  --duration-sec 300 \
  --output data/sessions/session_20251118_203000.csv
```

Options:

- `--port`: required.
    
- `--baudrate`: optional, default 115200.
    
- `--duration-sec`: optional, if omitted keep recording until Ctrl+C.
    
- `--output`: optional, if omitted generate filename with timestamp under `data/sessions/`.
    

CLI should print:

- Start time, estimated end time if duration specified.
    
- Number of samples recorded.
    
- Effective sampling rate at the end (for debugging).
    
Before recording starts, `cli.py` must ensure the storage directories exist:
    
    - When using `hragent.collect`, create `data/sessions/` if missing via `mkdir -p data/sessions` before opening the output file.
    
    - When running analysis/report flows (including `run_full_session`), ensure `data/reports/<session_id>/` exists by running `mkdir -p data/reports/<session_id>/` before writing reports or plots.
        
  This directory creation is part of the CLI command startup routine and must run prior to serial access or analysis.

---

## 7. Python: session storage

`session_store.py` handles loading and basic metadata.

### 7.1 Data model

Define:

```python
@dataclass
class SessionMetadata:
    path: Path
    n_samples: int
    duration_sec: float
    sample_rate_hz: float
    start_timestamp_ms: int
    end_timestamp_ms: int
```

### 7.2 Loader

Implement:

```python
def load_session(path: Path) -> tuple[pd.DataFrame, SessionMetadata]:
    """
    Load a session CSV into a DataFrame with columns:
    - timestamp_ms (int)
    - raw_value (int)

    Also compute metadata.
    """
```

Processing:

- Read CSV.
    
- Infer:
    
    - `n_samples = len(df)`.
        
    - `start_timestamp_ms = df.timestamp_ms.iloc[0]`.
        
    - `end_timestamp_ms = df.timestamp_ms.iloc[-1]`.
        
    - `duration_sec = (end - start) / 1000.0`.
        
    - `sample_rate_hz = n_samples / duration_sec` (for reporting and sanity check).
Notes:
- 読み込み時は文字列で取り込み、数値変換できない・範囲外の行をドロップする。
- 中央値のサンプル間隔から推定サンプリングレートを算出し、極端な外れ値の影響を抑える。
        

---

## 8. Python: heart rate and HRV analysis

`analysis.py` provides functions to compute metrics using HeartPy and an HRV library.

### 8.1 Dependencies

- `heartpy` for PPG and beat detection.
    
- `hrv-analysis` or `pyhrv` for HRV feature extraction.
    
- `numpy`, `pandas`.
    

### 8.2 Public API

```python
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

@dataclass
class AnalysisResult:
    session: SessionMetadata
    hr: HRMetrics
```

Main function:

```python
def analyze_session(
    df: pd.DataFrame,
    session_meta: SessionMetadata,
    expected_sample_rate_hz: float,
) -> AnalysisResult:
    """
    Run PPG based heart rate and HRV analysis on a session.

    Steps:
    1. Preprocess raw signal (detrend, normalise, optional bandpass).
    2. Run HeartPy to detect beats and compute heart rate.
    3. Extract RR intervals from HeartPy output.
    4. Run HRV library on RR series (time and frequency domain).
    5. Compute artifact ratio and basic quality flags.
    """

Hardware delivers raw PPG without analog filtering, so the preprocessing step must include bandpass filtering, smoothing, and normalization before passing data to HeartPy.

```

Implementation hints:

- Use `heartpy.process()` with `sample_rate_hz` from metadata or config.
    
- Construct RR list from HeartPy `working_data['RR_list']`.
    
- Use `hrv-analysis` functions:
    
    - preprocessing (artifact removal) on RR series.
        
    - `get_time_domain_features`, `get_frequency_domain_features`.
        
- Fill `HRMetrics` and `AnalysisResult`.
    
Additional robustness:

- 0/1023 近辺に張り付くサチュレーション区間は線形補間してから前処理を行う。
- RR間隔は 300〜2000 ms (30–200 bpm 相当) で外れ値を除去する。
- アーチファクト率は飽和割合も考慮して計算する。
    

### 8.3 Error handling

- If HeartPy fails due to too noisy data:
    
    - Raise a custom exception (e.g. `AnalysisError`) or return a result with a `quality` field set to `poor`.
        
    - Agent should be able to mention that analysis quality is low.
        

---

## 9. Python: plotting

`plots.py` creates plot images for the report.

### 9.1 Dependencies

- `matplotlib` (no seaborn).
    
- `numpy`.
    

### 9.2 API

```python
@dataclass
class PlotPaths:
    signal_plot: Path
    hr_plot: Path | None
    hrv_psd_plot: Path | None

def generate_plots(
    df: pd.DataFrame,
    analysis: AnalysisResult,
    output_dir: Path,
) -> PlotPaths:
    """
    Generate plots for:
    - PPG signal with detected peaks.
    - Heart rate over time.
    - HRV power spectral density (if available).
    """
```

Behavior:

- Treat `output_dir` as the directory where the Markdown report lives (e.g. `data/reports/<session_id>/`).
    
- Assume the CLI already created `output_dir`; `generate_plots()` only needs to ensure the nested `plots/` directory exists.
    
- Create `output_dir / "plots"` and save PNG files there with descriptive filenames.
    
- Save at least these files:
    
    - `plots/ppg.png`
    - `plots/hr.png`
    
    - `plots/hrv_psd.png` (when available).
        
- Return their **absolute paths** (or `Path` objects) in `PlotPaths` so other modules can reference them reliably.
    
- When the agent builds Markdown, it should link to images via the relative path `plots/<file>.png` from the report directory.

---

## 10. AI agent specification

`agent.py` implements a LangGraph based agent for report generation.

### 10.1 Dependencies

- `langchain-core`
    
- `langchain-openai`
    
- `langgraph`
    
- `pydantic`
    
- `python-dotenv` for loading OpenAI API key.
    
- OpenAI model: configurable via env, default `gpt-4o` or similar.
    

### 10.2 State definition

Agent state (TypedDict or Pydantic) should include:

- `messages`: conversation history.
    
- `analysis`: serialized `AnalysisResult` (e.g. JSON serializable dict).
    
- `plots`: list of plot paths or filenames.
    
- `report_markdown`: generated report string.
    
- `complete`: bool flag for termination.
    

Example:

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    analysis: dict | None
    plots: list[str]
    report_markdown: str
    complete: bool
```

### 10.3 Nodes

Implement at least these nodes in LangGraph:

1. `load_and_analyze_node`:
    
    - Input: path to session CSV (from initial human message).
        
    - Calls `load_session()`, `analyze_session()`, `generate_plots()`.
        
    - Stores results into `state["analysis"]` and `state["plots"]`.
        
2. `generate_report_node`:
    
    - Builds a system prompt with strict instructions:
        
        - Report in Japanese, plain text (no code fences).
            
        - Include sections: 概要, 測定条件, 心拍解析結果, HRV解析結果, 信号品質, 注意事項, まとめ（本セッションで読み取れること）.
            
        - Do not provide medical diagnosis; write for readers new to HR/HRV and add short explanations for technical terms.
            
    - Injects analysis metrics and plot filenames as context.
        
    - When including plot images in Markdown, refer to them via the relative path `plots/<filename>.png` assuming the report lives in `data/reports/<session_id>/`.
        
    - Calls OpenAI chat model via `langchain-openai` (`ChatOpenAI`).
        
    - Stores Markdown text in `state["report_markdown"]`.
        
    - Sets `state["complete"] = True`.
        
3. Optional `reflection_node`:
    
    - Takes current `report_markdown` and `analysis`.
        
    - Asks model to review and improve clarity, consistency, and remove speculation.
        
    - Updates `report_markdown`.
        

### 10.4 Graph structure

- Start → `load_and_analyze_node` → `generate_report_node` → optional `reflection_node` → End.
    
- Conditional edges:
    
    - Use a `should_continue` function to loop `reflection_node` at most `MAX_REFLECTIONS` times.
        

### 10.5 Public API

```python
def generate_report_for_session(
    session_path: Path,
) -> str:
    """
    Run the agent end-to-end and return a Markdown report string.
    """
```

---

## 11. Report format

Template expectations (informal, agent will generate text):

- Top level title: `# 心拍解析レポート`.
    
- Sections:
    
    1. `## 概要`
        
    2. `## 測定条件`
        
        - 測定日時, 計測時間, 推定サンプリング周波数, 使用センサ.
            
    3. `## 心拍解析結果`
        
        - 平均心拍数, 最小/最大, 推移の概要.
            
    4. `## HRV解析結果`
        
        - 主な指標 (SDNN, RMSSD, LF, HF, LF/HF).
            
        - 解釈はあくまで一般的な傾向にとどめる.
            
    5. `## 信号品質`
        
        - アーチファクト率, 注意点.
        
    6. `## 注意事項`
        
        - 医療機器ではないこと, 診断目的で使用しないこと.
            
    7. `## まとめ（本セッションで読み取れること）`
        
        - 数値に基づいた2〜4行の簡潔な要約。一般的な目安との比較は一言にとどめ、個人差と非医療であることを強調。
            

The agent should embed plot filenames as markdown image references (relative paths), for example:

```markdown
![PPG波形](plots/session_20251118_ppg.png)
```

Actual file existence is guaranteed by `generate_plots()`.

### 11.1 Report and plot storage

- Each session gets a dedicated directory under `data/reports/` (e.g. `data/reports/session_20251118_203000/`).
    
- Place the Markdown report inside that directory, for example `data/reports/<session_id>/report.md`; the filename can vary as long as it stays within the session directory.
    
- Store plots in `data/reports/<session_id>/plots/` beneath the report directory, and refer to them within Markdown using relative paths such as `plots/ppg.png`.
    
- When `generate_plots()` runs, it must treat its `output_dir` argument as the report directory, create `output_dir / "plots"`, save images there, and return absolute `Path` objects pointing into `output_dir / "plots"`.


---

## 12. CLI for full pipeline

`cli.py` should also expose a command to run analysis and report in one shot after recording:

```bash
# 1) Record 5 minutes, 2) analyze, 3) generate report
python -m hragent.run_full_session \
  --port COM3 \
  --duration-sec 300 \
  --session-name test_session_1
```

Expected behavior:

- `run_full_session` first ensures `data/sessions/` and `data/reports/<session-name>/` exist (via `mkdir -p`), then proceeds.
- Logging: 記録中は10秒ごとにINFOログでサンプル数/実効Hzを表示し、解析開始・レポート完了もINFOで出力する。

1. Record data into `data/sessions/<session-name>.csv`.
    
2. Run analysis and generate plots.
    
3. Run agent to create Markdown report.
    
4. Save report into `data/reports/<session-name>.md`.
    
5. Print report path to stdout.
    

---

## 13. Error handling expectations

- **Serial connection errors:**
    
    - If the requested port cannot be opened, CLI must print `Failed to open serial port <port>. Check connection and port name.` and exit without retries.
        
    - Automatic retries are not performed; relaunching the CLI is required for another attempt.
        
- **Read errors during recording:**
    
    - `readline()` timeouts should emit warnings and continue, allowing up to five consecutive timeouts before aborting the session, closing files/ports, and reporting failure.
        
    - Malformed lines (missing comma, non-numeric content) must be skipped with a warning that includes the raw line or line number.
        
- **Analysis quality issues:**
    
    - HeartPy/HRV failures due to poor signal quality should set `quality: poor` or raise `artifact_percentage` flags in `AnalysisResult` while allowing the pipeline to continue.
        
    - The agent must note low signal quality in the report and avoid overstating conclusions; partial metrics may be included with a disclaimer.
        
---

## 14. Environment and configuration

- Python dependencies specified in `pyproject.toml`:
    
    - `heartpy`
        
    - `hrvanalysis` or `pyhrv` (choose one and wire it)
        
    - `pyserial`
        
    - `pandas`
        
    - `numpy`
        
    - `matplotlib`
        
    - `typer` or `argparse` (if using typer, add `typer`).
        
    - `langchain-core`, `langchain-openai`, `langgraph`
        
    - `pydantic`
        
    - `python-dotenv`
        
- Astral の Python 用ツールチェーンである `uv` を使い、依存管理・仮想環境・実行を一元化する。
    
    - 依存インストール例:

      ```bash
      uv sync
      ```
    
    - CLI 実行例:

      ```bash
      uv run python -m hragent.collect --port COM3 --duration-sec 300
      ```
    
    - `uv` が使えない環境では、通常の `venv` + `pip install -r requirements.txt`（`pyproject.toml` の依存を）で代替して構わない旨を明記する。
- `.env.sample` example:
    

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o
HRA_SERIAL_PORT=COM3
HRA_SAMPLE_RATE_HZ=100.0
```

---

## 15. Testing

Write tests for:

- `analysis.py`:
    
    - Use synthetic PPG like signals (e.g. sine wave) to ensure HR estimation is in expected range.
        
    - Test behavior with small and noisy sample arrays.
        
- `session_store.py`:
    
    - Correct duration and sample rate computation for known CSV fixtures.
        
- `agent.py`:
    
    - Use a stub LLM or small local model to test prompt construction and that `analysis` dict is correctly embedded into the messages.
        
    - Do not call real OpenAI API in unit tests, use mocks.
        

---

## 16. Implementation notes and constraints

- All medical wording in the report must include a disclaimer that the system is not a medical device and the output is not a diagnosis.
    
- The agent must not invent metrics that are not present in `AnalysisResult`.
    
- Sampling jitter is expected at a small level, but if inferred sample rate deviates more than, for example, ±10 % from config, the analysis should still run but mark `artifact_percentage` or `quality` as degraded.
  
