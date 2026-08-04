# ERA5-Land

Official source: https://cds.climate.copernicus.eu/

Download endpoint: Copernicus CDS API dataset `reanalysis-era5-land`.

```bash
python ECMWF-ERA5/download.py --start_year START_YEAR --end_year END_YEAR
python ECMWF-ERA5/preprocess.py --max_files 0 --force
```

The downloader stores monthly NetCDF or ZIP files in `ECMWF-ERA5/raw/`. The preprocessing step opens all selected raw files, extracts the configured regional grid, derives gridded meteorological fields and numerical features, and writes arrays plus metadata to `ECMWF-ERA5/processed/`. CDS credentials must be configured locally before downloading.