# Heart Rate Report Agent

Pulse Sensor + Arduinoで記録したPPGをPythonで解析し、AIが日本語レポートを生成する一括パイプラインです。ハード（Arduinoスケッチ）とソフト（Python収録/解析/レポート）が疎結合になっています。

## 構成
- `arduino/` : Arduino Uno向けスケッチ。`pulse_logger`がA0を100Hzで読み取り、`<timestamp_ms>,<raw_value>`をシリアル出力。
- `python/` : 収録・解析・レポート生成。LangGraph + ChatOpenAIで初心者向けの解説付きレポートを生成。
- `SPEC.md` : 詳細仕様（ハード/ソフト両方）。

## クイックスタート（Python側）
1. `cd python`
2. 依存取得: `uv sync`（または `pip install -e .[test]`）
3. 環境変数: `.env.sample` を `.env` にコピーし、`OPENAI_API_KEY`（必要なら `OPENAI_ORG`/`OPENAI_PROJECT`）と `HRA_SERIAL_PORT` を設定。
4. 一括実行例（60秒計測）:  
   `make full PORT=/dev/ttyACM1 SESSION=test6 DURATION=60`

## 主な機能（Python）
- 記録: `python -m hragent.collect`（ポート指定、タイムアウト等も指定可）
- 一括: `python -m hragent.run_full_session`（記録→解析→レポート）
- 解析: 0/1023付近のサチュレーション補間、RR 300–2000msの外れ値除去、HeartPy+hrvanalysis
- レポート: ChatOpenAIを使用し、初心者向けの平易な解説＋「まとめ」セクションを含む。プロットは相対パスで埋め込み。
- ログ: 記録中10秒ごとにINFOでサンプル数/実効Hzを表示。古いシリアルバッファを破棄し、5秒未満のタイムスタンプから収録開始。範囲外・欠損行・逆行タイムスタンプはスキップ。

## Makefile（簡易デモ用, `python/`直下で実行）
- 例: `make full PORT=/dev/ttyACM1 SESSION=test6 DURATION=60`
- `make collect` 記録のみ / `make report SESSION=...` 既存CSVからレポートのみ / `make monitor` でシリアルモニタ
- `PORT` は `HRA_SERIAL_PORT` を優先、未指定なら `/dev/ttyACM0`

## Arduino側
- スケッチ: `arduino/pulse_logger/pulse_logger.ino`（A0, 115200bps, 10ms間隔）。`millis()`ベースの非ブロッキングループ。
- 接続: `5V/GND/A0` にPulse Sensorを直結。外付けフィルタ等は不要、Python側で処理。

## テスト（Python）

```bash
pip install -e .[test]
pytest
```
