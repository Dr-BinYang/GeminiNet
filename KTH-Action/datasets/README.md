# KTH Action data

Official source: https://www.csc.kth.se/cvap/actions/

From the project root:

```bash
python datasets/download.py
python datasets/preprocess.py --force
```

The downloader retrieves every official action archive. The preprocessor decodes the videos, resamples the frame sequences, and derives motion-state features. Raw and processed data are ignored by Git.