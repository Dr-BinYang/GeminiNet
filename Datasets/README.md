# Dataset access

This directory contains standalone dataset download and preprocessing workflows for the GeminiNet experiments. Each dataset folder includes:

- `download.py`: downloads raw files from the official source or public data service.
- `preprocess.py`: converts raw files into arrays and metadata used by the corresponding experiment.
- `README.md`: summarizes the source, command examples, and processing steps.

The scripts can be used from the full repository or from a copied `Datasets` folder. By default, each workflow writes raw files to `<dataset>/raw/` and processed files to `<dataset>/processed/`.

Large raw files, processed arrays, checkpoints, and experiment outputs are not included in this repository.

## Typical usage

Run commands from the repository root:

```bash
python Datasets/KTH-Action/download.py
python Datasets/KTH-Action/preprocess.py --force
```

Or run them from inside a copied `Datasets` directory:

```bash
python KTH-Action/download.py
python KTH-Action/preprocess.py --force
```

For weather and climate datasets, pass the time range required by the experiment at runtime. From the repository root, keep the `Datasets/` prefix:

```bash
python Datasets/ECMWF-ERA5/download.py --start_year START_YEAR --end_year END_YEAR
python Datasets/NOAA-GOES16/preprocess.py --start_time START_ISO_UTC --end_time END_ISO_UTC --max_files 0 --force
```

## Dataset index

| Dataset | Workflow | Download endpoint |
| --- | --- | --- |
| Moving MNIST | `Moving-MNIST/` | MNIST via torchvision; source reference: https://www.kaggle.com/datasets/hojjatk/mnist-dataset |
| BAIR Robot Pushing | `Bair-Robot-Pushing/` | http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar |
| UTD-MHAD | `UTD-MHAD/` | https://personal.utdallas.edu/~kehtar/UTD-MHAD.html |
| KTH Action | `KTH-Action/` | https://www.csc.kth.se/cvap/actions/<action>.zip |
| MeteoNet NW | `MeteoNet-NW/` | https://meteonet.umr-cnrm.fr/dataset/data/NW/ |
| ERA5-Land | `ECMWF-ERA5/` | Copernicus CDS API dataset `reanalysis-era5-land`: https://cds.climate.copernicus.eu/ |
| GOES-16 ABI | `NOAA-GOES16/` | NOAA public S3 bucket: https://noaa-goes16.s3.amazonaws.com |
| SEVIR IR107 | `SEVIR-IR107/` | https://sevir.s3.amazonaws.com/CATALOG.csv and IR107 HDF5 files under https://sevir.s3.amazonaws.com |