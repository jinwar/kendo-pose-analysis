#!/bin/bash
cd "$(dirname "$0")" || exit 1
if [ -f outputs/index.html ]; then
  open outputs/index.html
else
  echo 'No completed results yet. Run a comparison first.'
  read -r -p 'Press Return to close.'
fi
