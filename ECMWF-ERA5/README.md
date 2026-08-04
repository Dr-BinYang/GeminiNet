# GeminiNet for ERA5-Land

This subproject forecasts a regional two-metre temperature field together with aligned regional-mean atmospheric and land-surface variables.

## Data

ERA5 access information: https://console.cloud.google.com/storage/browser/weatherbench2/datasets/era5

The included downloader uses the Copernicus Data Store API. Configure `cdsapi` and select any available interval without a project-imposed file cap:

```bash
python datasets/download.py --start_year START --end_year END
python datasets/preprocess.py --force
```

The preprocessor converts monthly NetCDF files into aligned `images.npy`, `numbers.npy`, timestamps, coordinates, and metadata under `datasets/ERA5_Land_ChengYu/processed`.

## Training

```bash
python -m pip install -r requirements.txt
python run.py
```

Outputs are written under `experiments/era5_land_chengyu`.