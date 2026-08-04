# GeminiNet for GOES-16 ABI

This subproject forecasts GOES-16 ABI Channel-13 CONUS brightness-temperature fields together with image-derived cloud-state trajectories.

## Data

Official information: https://www.ncei.noaa.gov/products/goes-terrestrial-weather-abi-glm

Select an explicit UTC interval from the public archive:

```bash
python datasets/download.py --start_time START --end_time END
python datasets/preprocess.py --start_time START --end_time END --force
```

No fixed project year or file-count cap is used. The preprocessor samples the selected interval, converts the infrared field to brightness temperature, and derives cloud fraction and texture variables under `datasets/GOES16_ABI_C13_CONUS/processed`.

## Training

```bash
python -m pip install -r requirements.txt
python run.py
```

Outputs are written under `experiments/goes16_abi_c13_conus`.