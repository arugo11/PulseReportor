# Heart Rate Report Agent (Python)

このディレクトリには心拍データ収集から解析、レポート生成までを実装したPythonパッケージがあります。

## セットアップ

1. [`uv`](https://github.com/astral-sh/uv) が利用できるなら `uv sync` で依存を取り込みます。
2. 使えない場合は通常の仮想環境で `pip install -e .[test]` を実行してください。
3. `.env.sample` をコピーして `.env` を作成し、`OPENAI_API_KEY` などを設定します（必要に応じて `OPENAI_ORG`/`OPENAI_PROJECT` を追加）。

## 主な機能

- `python -m hragent.collect` で Pulse Sensor からシリアル生データを記録。
- `python -m hragent.run_full_session` で記録→解析→レポート生成を一括実行。
- `hragent.analysis`, `hragent.plots` で信号処理とプロットを行い、`hragent.agent` で LangGraph + LangChain を使ったAIレポート。
- `python/data/sessions` と `python/data/reports` に結果を保存。
- 10秒ごとに記録進捗（サンプル数/実効Hz）をINFOログで表示。
- 記録時はシリアルの古いバッファを破棄し、タイムスタンプが5秒未満の行から開始。値の範囲外や欠損行はスキップ。
- 解析時は0/1023付近に張り付いた区間を補間し、RR間隔を300〜2000msに制限して外れ値を除去。
- レポートはChatOpenAIを使用し、初心者向けの平易な解説と「まとめ」セクションを含む。

## Makefile（簡易デモ用）

- 例: `make full PORT=/dev/ttyACM1 SESSION=test6 DURATION=60`
- `make collect` 記録のみ、`make report SESSION=...` 既存CSVからレポートのみ、`make monitor` でシリアルモニタ。
- `PORT` は `HRA_SERIAL_PORT` を拾うか、指定がない場合 `/dev/ttyACM0` を使用。

## テスト

```bash
pip install -e .[test]
pytest
```
