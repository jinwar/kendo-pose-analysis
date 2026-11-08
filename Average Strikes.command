#!/bin/bash
cd "$(dirname "$0")" || exit 1
.venv/bin/python average_strikes.py
read -r -p 'Press Return to close.'
