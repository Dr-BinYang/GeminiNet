# KTH Action

Official source: https://www.csc.kth.se/cvap/actions/

Download endpoint: `https://www.csc.kth.se/cvap/actions/<action>.zip`, where `<action>` is one of walking, jogging, running, boxing, handwaving, and handclapping.

```bash
python KTH-Action/download.py
python KTH-Action/preprocess.py --force
```

The downloader stores ZIP archives and extracted videos in `KTH-Action/raw/`. The preprocessing step discovers videos by action folder, decodes frames, resamples each video to a fixed temporal length, computes motion-based numerical features, and writes arrays plus metadata to `KTH-Action/processed/`.