#!/bin/bash
set -e
cd "$(dirname "$0")"
.venv/bin/python detect_shinai.py --events strike_events_second.json --output outputs/strike_analysis_second
.venv/bin/python plot_strikes.py --events strike_events_second.json --output outputs/strike_analysis_second
.venv/bin/python average_strikes.py --input outputs/strike_analysis_second/aligned_poses.json --output outputs/average_strike_second --max-time-span-ms 10
.venv/bin/python average_strikes.py
.venv/bin/python compare_average_strikes.py
read -r -p 'Press Return to close.'
