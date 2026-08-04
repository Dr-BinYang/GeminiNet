# BAIR Robot Pushing

Official source: http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar

Download endpoint: `http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar`

```bash
python Bair-Robot-Pushing/download.py
python Bair-Robot-Pushing/preprocess.py --force
```

The downloader stores the archive and extracted TFRecord files in `Bair-Robot-Pushing/raw/`. The preprocessing step reads robot pushing trajectories, builds fixed input and prediction windows, keeps the configured camera stream as the image modality, and writes arrays plus metadata to `Bair-Robot-Pushing/processed/`.