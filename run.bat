@echo off
REM Start the landslide susceptibility app.
REM Nothing here needs credentials: elevation comes from a public SRTM endpoint.

python -m pip install -r requirements.txt
if not exist "models\landslide_model.joblib" python "src\train.py"

REM The district map is optional but it is the headline view. Building it takes a few minutes
REM the first time, because every cell needs its own elevation window from a shared service.
if not exist "data\susceptibility_grid.json" (
  echo Building the district susceptibility map. This runs once and takes a few minutes.
  python "src\susceptibility_map.py"
)

python "web\app.py"
