# UTD-MHAD

Official source: https://personal.utdallas.edu/~kehtar/UTD-MHAD.html

Download endpoint: the downloader scrapes the official page above and also includes direct fallback archive URLs for UTD-MHAD depth, inertial, and skeleton files.

```bash
python UTD-MHAD/download.py --max_total_gb 0
python UTD-MHAD/preprocess.py --max_sequences 0 --force
```

The downloader stores archives and extracted files in `UTD-MHAD/raw/`. The preprocessing step extracts archive files when needed, matches modalities by sequence identity, decodes the configured image source, loads inertial or skeleton features, resamples sequences, and writes arrays plus labels and metadata to `UTD-MHAD/processed/`.