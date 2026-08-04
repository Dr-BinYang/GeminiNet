# ERA5-Land data

Dataset access: https://console.cloud.google.com/storage/browser/weatherbench2/datasets/era5

The project preprocessor uses ERA5-Land through the Copernicus Data Store API. Configure `cdsapi`, then choose the desired available interval:

```bash
python datasets/download.py --start_year START --end_year END
python datasets/preprocess.py --force
```

The preprocessor extracts regional temperature fields and aligned regional-mean atmospheric and land-surface variables. No fixed year range or file-count cap is imposed by the project.