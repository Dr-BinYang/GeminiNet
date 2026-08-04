# NOAA GOES-16 ABI Channel 13

Official source: https://www.ncei.noaa.gov/products/goes-terrestrial-weather-abi-glm

Download endpoint: NOAA public S3 bucket `https://noaa-goes16.s3.amazonaws.com`, product `ABI-L2-CMIPC`, channel `13`.

```bash
python NOAA-GOES16/download.py --start_time START_ISO_UTC --end_time END_ISO_UTC --max_files 0
python NOAA-GOES16/preprocess.py --start_time START_ISO_UTC --end_time END_ISO_UTC --max_files 0 --force
```

The downloader stores NetCDF files in `NOAA-GOES16/raw/`. The preprocessing step queries the requested UTC interval, downloads missing files, reads brightness-temperature grids, resizes infrared images, computes cloud and temperature features, and writes arrays plus metadata to `NOAA-GOES16/processed/`.