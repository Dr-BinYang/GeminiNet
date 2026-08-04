# Moving MNIST

Official source: https://www.kaggle.com/datasets/hojjatk/mnist-dataset

Download endpoint: torchvision MNIST mirrors, with the Kaggle page above as a dataset reference.

```bash
python Moving-MNIST/download.py
python Moving-MNIST/preprocess.py --download_mnist --overwrite
```

The downloader stores MNIST files in `Moving-MNIST/raw/`. The preprocessing step samples digit images, simulates motion on a square canvas, records frame tensors and motion-state variables, and writes split files plus metadata to `Moving-MNIST/processed/`.