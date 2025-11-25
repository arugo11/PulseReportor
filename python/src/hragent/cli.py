from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import typer
from serial import SerialException

from .agent import generate_report_for_session
from .config import ensure_session_storage, get_report_dir, load_config
from .serial_collect import record_session

app = typer.Typer()


def _format_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def _prepare_config(port: str, baudrate: int | None, timeout: float | None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return load_config(port=port, baudrate=baudrate, timeout=timeout)


@app.command("collect")
def collect(
    port: str = typer.Option(..., help="シリアルポート名"),
    baudrate: int = typer.Option(115200, help="ボーレート"),
    duration_sec: float | None = typer.Option(None, help="記録時間（秒）") ,
    output: Path | None = typer.Option(None, help="出力ファイルパス"),
) -> None:
    """Pulse Sensor からシリアルで生データを記録する"""
    try:
        config = _prepare_config(port, baudrate, None)
    except ValueError as exc:
        typer.echo(f"設定読み込みに失敗しました: {exc}")
        raise typer.Exit(code=1)
    ensure_session_storage(config.paths)
    session_dir = config.paths.sessions_dir
    if output:
        output_path = output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = session_dir / f"session_{timestamp}.csv"
    try:
        typer.echo(f"記録開始: {datetime.now():%Y-%m-%d %H:%M:%S}")
        if duration_sec:
            typer.echo(f"推定終了: {datetime.now() + timedelta(seconds=duration_sec):%Y-%m-%d %H:%M:%S}")
        result = record_session(config.serial, duration_sec, output_path)
    except SerialException:
        typer.echo(f"Failed to open serial port {port}. Check connection and port name.")
        raise typer.Exit(code=1)
    except RuntimeError as exc:
        typer.echo(f"記録に失敗しました: {exc}")
        raise typer.Exit(code=1)
    typer.echo(f"記録完了: {result.output_path}")
    typer.echo(f"サンプル数: {result.n_samples}")
    typer.echo(f"実効サンプリングレート: {result.sample_rate_hz:.1f} Hz")


@app.command("run_full_session")
def run_full_session(
    port: str = typer.Option(..., help="シリアルポート名"),
    duration_sec: float | None = typer.Option(None, help="記録時間（秒）"),
    session_name: str = typer.Option(..., help="セッション名"),
) -> None:
    """記録→解析→レポート生成を一括実行"""
    config = _prepare_config(port, None, None)
    ensure_session_storage(config.paths)
    report_dir = get_report_dir(config.paths, session_name)
    session_path = config.paths.sessions_dir / f"{session_name}.csv"

    try:
        typer.echo("データ記録中...")
        result = record_session(config.serial, duration_sec, session_path)
    except SerialException:
        typer.echo(f"Failed to open serial port {port}. Check connection and port name.")
        raise typer.Exit(code=1)
    except RuntimeError as exc:
        typer.echo(f"記録に失敗しました: {exc}")
        raise typer.Exit(code=1)

    typer.echo(f"記録完了: {session_path}")
    typer.echo(f"サンプル数: {result.n_samples}")
    typer.echo(f"実効サンプリングレート: {result.sample_rate_hz:.1f} Hz")
    typer.echo("解析とレポート生成")
    try:
        report = generate_report_for_session(session_path)
    except Exception as exc:
        typer.echo(f"レポート生成に失敗しました: {exc}")
        raise typer.Exit(code=1)
    report_file = report_dir / "report.md"
    report_file.write_text(report, encoding="utf-8")
    typer.echo(f"レポート保存: {report_file}")
