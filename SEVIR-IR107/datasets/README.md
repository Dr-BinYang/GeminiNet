# SEVIR IR107 data

Official source: https://sevir.s3.amazonaws.com

From the project root:

```bash
python datasets/download.py
python datasets/preprocess.py --force
```

The downloader reads the SEVIR catalog and retrieves all IR107 files unless an optional limit is supplied. The preprocessor stores event-level infrared sequences and derives cloud-state and texture variables. Raw and processed data are ignored by Git.