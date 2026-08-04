# GeminiNet for SEVIR IR107

This subproject forecasts event-level SEVIR IR107 infrared fields together with image-derived cloud-state and texture trajectories.

## Data

Official source: https://sevir.s3.amazonaws.com

```bash
python datasets/download.py
python datasets/preprocess.py --force
```

The downloader reads the official catalog and retrieves every IR107 file unless an optional limit is supplied. The preprocessor preserves event boundaries, resizes infrared frames when needed, and stores aligned visual and numerical sequences under `datasets/SEVIR_IR107_192/processed`.

## Training

```bash
python -m pip install -r requirements.txt
python run.py
```

Outputs are written under `experiments/sevir_ir107_192`.