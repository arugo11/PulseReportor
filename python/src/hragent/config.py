from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv, dotenv_values
from pydantic import BaseModel, Field, root_validator


class SerialConfig(BaseModel):
    port: str = Field("", description="Serial port name, e.g. COM3 or /dev/ttyACM0")
    baudrate: int = 115200
    timeout: float = 1.0


class Paths(BaseModel):
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    sessions_dir: Path | None = None
    reports_dir: Path | None = None

    @root_validator(pre=True)
    def _populate_paths(cls, values: dict[str, Any]) -> dict[str, Any]:
        base = values.get("base_dir") or Path(__file__).resolve().parents[2]
        values.setdefault("base_dir", base)
        values.setdefault("sessions_dir", base / "data" / "sessions")
        values.setdefault("reports_dir", base / "data" / "reports")
        return values


class AppConfig(BaseModel):
    serial: SerialConfig
    paths: Paths
    sample_rate_hz: float = 100.0
    openai_model: str = "gpt-4o"


def _load_env(base_dir: Path) -> dict[str, str]:
    dotenv_path = base_dir / ".env"
    load_dotenv(dotenv_path)
    merged = {**dotenv_values(dotenv_path)}
    merged.update({k: v for k, v in os.environ.items() if v is not None})
    return merged


def _cast(value: str | None, cast: type, default: Any) -> Any:
    if value is None:
        return default
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def load_config(
    *,
    base_dir: Path | None = None,
    port: str | None = None,
    baudrate: int | None = None,
    timeout: float | None = None,
    sample_rate_hz: float | None = None,
    openai_model: str | None = None,
) -> AppConfig:
    env_override = os.environ.get("HRA_BASE_DIR")
    base_dir = Path(env_override) if env_override else base_dir or Path(__file__).resolve().parents[2]
    env = _load_env(base_dir)
    if (env_base := env.get("HRA_BASE_DIR")):
        base_dir = Path(env_base)
        env = _load_env(base_dir)
    serial_kwargs = {
        "port": port or env.get("HRA_SERIAL_PORT") or "",
        "baudrate": baudrate or _cast(env.get("HRA_SERIAL_BAUDRATE"), int, 115200),
        "timeout": timeout or _cast(env.get("HRA_SERIAL_TIMEOUT"), float, 1.0),
    }
    cfg = AppConfig(
        serial=SerialConfig(**serial_kwargs),
        paths=Paths(base_dir=base_dir),
        sample_rate_hz=sample_rate_hz or _cast(env.get("HRA_SAMPLE_RATE_HZ"), float, 100.0),
        openai_model=openai_model or env.get("OPENAI_MODEL", "gpt-4o"),
    )
    return cfg


def get_report_dir(paths: Paths, session_name: str) -> Path:
    report_dir = paths.reports_dir / session_name
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def ensure_session_storage(paths: Paths) -> None:
    paths.sessions_dir.mkdir(parents=True, exist_ok=True)
