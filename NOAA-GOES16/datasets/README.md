# GOES-16 ABI data

Official information: https://www.ncei.noaa.gov/products/goes-terrestrial-weather-abi-glm

Specify an explicit UTC interval:

```bash
python datasets/download.py --start_time START --end_time END
python datasets/preprocess.py --start_time START --end_time END --force
```

The downloader uses the public NOAA GOES-16 S3 archive. The preprocessor samples ABI Channel-13 CONUS imagery, converts it to brightness temperature, and derives cloud-state statistics. No fixed project year or file-count cap is used.