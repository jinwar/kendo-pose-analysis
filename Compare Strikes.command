#!/bin/bash
set -e
cd "$(dirname "$0")"
.venv/bin/python compare_individual_strikes.py
read -r -p 'Press Return to close.'
