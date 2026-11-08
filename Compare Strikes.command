#!/bin/bash
set -e
cd "$(dirname "$0")"
.venv/bin/python detect_shinai.py --post-hit-frames 6 --output outputs/strike_followthrough
.venv/bin/python detect_shinai.py --events strike_events_second.json --post-hit-frames 6 --output outputs/strike_followthrough_second
.venv/bin/python compare_individual_strikes.py
read -r -p 'Press Return to close.'
