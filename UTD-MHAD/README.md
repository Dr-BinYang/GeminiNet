# GeminiNet for UTD-MHAD

This subproject forecasts human-action video frames together with wearable inertial trajectories. RGB is the default visual modality; depth can be selected during preprocessing.

## Data

Official source: https://personal.utdallas.edu/~kehtar/UTD-MHAD.html

```bash
python datasets/download.py
python datasets/preprocess.py --force
```

The official site may require manual agreement or direct archive URLs. The downloader accepts repeated `--url` options when automatic discovery is unavailable. The preprocessor pairs visual sequences with inertial measurements and processes all available pairs by default.

## Training

```bash
python -m pip install -r requirements.txt
python run.py
```

Outputs are written under `experiments/utd_mhad_rgb_inertial`.