# GeminiNet for Moving MNIST

This subproject forecasts moving handwritten-digit frames and their aligned motion-state trajectories. The numerical modality describes position, velocity, size, and digit identity.

## Data

MNIST source: https://www.kaggle.com/datasets/hojjatk/mnist-dataset

```bash
python datasets/download.py
python datasets/preprocess.py --download_mnist --overwrite
```

The generator uses the complete MNIST source pool and writes configurable moving-digit training, validation, and test sequences under `datasets/Moving_MNIST_State`.

## Training

```bash
python -m pip install -r requirements.txt
python run.py
```

Outputs are written under `experiments/moving_mnist_state`. 