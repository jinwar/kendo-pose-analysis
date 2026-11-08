#!/bin/bash
cd "$(dirname "$0")" || exit 1
.venv/bin/python detect_shinai.py && .venv/bin/python plot_strikes.py
read -r -p 'Press Return to close.'
