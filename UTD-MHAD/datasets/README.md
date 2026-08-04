# UTD-MHAD data

Official source: https://personal.utdallas.edu/~kehtar/UTD-MHAD.html

From the project root:

```bash
python datasets/download.py
python datasets/preprocess.py --force
```

The official site may require manual agreement or direct archive URLs. The downloader accepts repeated `--url` options when automatic link discovery is unavailable. The preprocessor pairs video or depth sequences with inertial measurements and processes all available pairs by default.