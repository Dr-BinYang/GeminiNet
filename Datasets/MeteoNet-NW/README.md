# MeteoNet NW

Official source: https://meteonet.umr-cnrm.fr/dataset/

Download endpoints: `https://meteonet.umr-cnrm.fr/dataset/data/NW/ground_stations/` and `https://meteonet.umr-cnrm.fr/dataset/data/NW/radar/rainfall/`.

```bash
python MeteoNet-NW/download.py
python MeteoNet-NW/preprocess.py --max_hours 0 --force
```

The downloader stores archives and extracted files in `MeteoNet-NW/raw/`. The preprocessing step recursively discovers station CSV files and rainfall NPZ fields, aligns records by timestamp, downsamples radar grids, computes tabular weather features, and writes arrays plus metadata to `MeteoNet-NW/processed/`.