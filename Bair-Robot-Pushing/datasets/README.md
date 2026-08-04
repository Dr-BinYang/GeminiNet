# BAIR Robot Pushing data

Official information: https://www.tensorflow.org/datasets/catalog/bair_robot_pushing_small

From the project root:

```bash
python datasets/download.py
python datasets/preprocess.py --force
```

The downloader retrieves and extracts the original TFRecord archive. The preprocessor converts RGB frames, end-effector positions, and actions into memory-mapped NumPy arrays. Raw and processed data are ignored by Git.