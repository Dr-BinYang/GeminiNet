# SEVIR IR107

Official source: https://sevir.s3.amazonaws.com

Download endpoints: `https://sevir.s3.amazonaws.com/CATALOG.csv` and IR107 HDF5 files listed in the catalog under `https://sevir.s3.amazonaws.com`.

```bash
python SEVIR-IR107/download.py --max_raw_gb 0 --max_files 0
python SEVIR-IR107/preprocess.py --max_raw_gb 0 --max_files 0 --max_events 0 --force
```

The downloader stores the catalog and HDF5 files in `SEVIR-IR107/raw/`. The preprocessing step reads the catalog, selects IR107 files, downloads missing HDF5 files, extracts event sequences, resizes infrared frames, computes image-level features, and writes arrays plus metadata to `SEVIR-IR107/processed/`.