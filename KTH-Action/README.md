# GeminiNet for KTH Action

This subproject forecasts grayscale human-action video frames and video-derived motion-state trajectories. Motion energy, active-region geometry, centroid, and spread provide the numerical modality.

## Data

Official source: https://www.csc.kth.se/cvap/actions/

```bash
python datasets/download.py
python datasets/preprocess.py --force
```

The downloader retrieves every official action archive. The preprocessor decodes and temporally resamples the videos, derives motion features, and stores the aligned arrays under `datasets/KTH_Action/processed`.

## Training

```bash
python -m pip install -r requirements.txt
python run.py
```

Outputs are written under `experiments/kth_action_motion_stats`.