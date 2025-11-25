PORT ?= $(HRA_SERIAL_PORT)
PORT ?= /dev/ttyACM0
SESSION ?= demo
DURATION ?= 60
CSV := data/sessions/$(SESSION).csv
REPORT := data/reports/$(SESSION)/report.md

.PHONY: help setup collect full report monitor

help:
	@echo "make setup           # uv sync"
	@echo "make collect         # record only (SESSION, DURATION, PORT configurable)"
	@echo "make full            # record -> analyze -> report (SESSION, DURATION, PORT configurable)"
	@echo "make report SESSION=... # generate report from existing CSV"
	@echo "make monitor         # open serial monitor (arduino-cli) for quick check"

setup:
	uv sync

collect:
	@if [ -z "$(PORT)" ]; then echo "PORT is required (set PORT or HRA_SERIAL_PORT)"; exit 1; fi
	uv run python -m hragent.collect --port $(PORT) --duration-sec $(DURATION) --output $(CSV)

full:
	@if [ -z "$(PORT)" ]; then echo "PORT is required (set PORT or HRA_SERIAL_PORT)"; exit 1; fi
	uv run python -m hragent.run_full_session --port $(PORT) --duration-sec $(DURATION) --session-name $(SESSION)

report:
	@if [ -z "$(SESSION)" ]; then echo "SESSION is required"; exit 1; fi
	uv run python - <<'PY'
	from pathlib import Path
	from hragent.agent import generate_report_for_session
	session = Path("data/sessions/$(SESSION).csv")
	print(generate_report_for_session(session))
	PY

monitor:
	@if [ -z "$(PORT)" ]; then echo "PORT is required (set PORT or HRA_SERIAL_PORT)"; exit 1; fi
	arduino-cli monitor -p $(PORT) -b arduino:avr:uno
