#!/usr/bin/env bash
# Start the landslide susceptibility app.
# Nothing here needs credentials: elevation comes from a public SRTM endpoint.
set -e

python -m pip install -r requirements.txt
[ -f models/landslide_model.joblib ] || python src/train.py

# The district map is optional but it is the headline view. Building it takes a few minutes the
# first time, because every cell needs its own elevation window from a shared service.
if [ ! -f data/susceptibility_grid.json ]; then
  echo "Building the district susceptibility map. This runs once and takes a few minutes."
  python src/susceptibility_map.py
fi

python web/app.py
