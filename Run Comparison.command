#!/bin/bash
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo 'First follow the setup instructions in README.md.'
  read -r -p 'Press Return to close.'
  exit 1
fi
echo 'Choose a video to process (the full clip will take several minutes):'
select video in videos/*.mp4; do
  if [ -n "$video" ]; then
    .venv/bin/python compare.py "$video"
    break
  fi
done
read -r -p 'Press Return to close.'
