# GeminiNet for MeteoNet NW

This subproject forecasts north-west France radar rainfall fields together with aligned regional weather-state trajectories derived from ground-station observations.

## Data

Official source: https://meteonet.umr-cnrm.fr/dataset/

```bash
python datasets/download.py
python datasets/preprocess.py --force
```

The downloader retrieves all rainfall and ground-station archives used by this project. The preprocessor aggregates radar observations and station variables to a shared hourly timeline under `datasets/MeteoNet_NW/processed`.

## Training

```bash
python -m pip install -r requirements.txt
python run.py
```

Outputs are written under `experiments/meteonet_nw`.