# Moving MNIST data

MNIST source: https://www.kaggle.com/datasets/hojjatk/mnist-dataset

From the project root:

```bash
python datasets/download.py
python datasets/preprocess.py --download_mnist --overwrite
```

The source MNIST digits are downloaded through torchvision. The generator produces moving-digit image sequences and their aligned motion-state trajectories. Generated data are ignored by Git.