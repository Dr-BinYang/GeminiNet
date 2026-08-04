# GeminiNet for BAIR Robot Pushing

This subproject forecasts robot-pushing video frames together with the aligned robot state and action trajectory. The visual modality uses the main RGB camera. The numerical modality is built from end-effector position and control signals.

## Data

Official dataset information: https://www.tensorflow.org/datasets/catalog/bair_robot_pushing_small

```bash
python datasets/download.py
python datasets/preprocess.py --force
```

The preprocessor reads the original TFRecords and writes memory-mapped training and test arrays under `datasets/BAIR_Robot_Pushing_Small/processed`.

## Training

```bash
python -m pip install -r requirements.txt
python run.py
```

The default run trains the image tendency model, the numerical tendency model, and GeminiNet in sequence. Checkpoints, metrics, and visualizations are written under `experiments/bair_robot_pushing_small`.