# GeminiNet

> **⚠️ Naming Note:** *GeminiNet* refers to the "twin" relationship between image and numerical modalities in multimodal forecasting, inspired by the Gemini constellation. **It is unrelated to Google's Gemini large language models.**

GeminiNet is a multimodal forecasting framework for paired visual and numerical time series. Two unimodal predictors first estimate modality-specific evolutionary tendencies. The main network then conditions historical observations on those tendencies, estimates their reliability, performs cross-modal interaction, and forecasts both future modalities.

## Dataset projects

Each directory is a self-contained experiment with the same model family and a dataset-specific input pipeline.

- `Moving-MNIST`
- `Bair-Robot-Pushing`
- `UTD-MHAD`
- `KTH-Action`
- `MeteoNet-NW`
- `ECMWF-ERA5`
- `NOAA-GOES16`
- `SEVIR-IR107`

The centralized [Datasets](Datasets/README.md) directory provides dataset download and preprocessing workflows.

## Usage

Choose a subproject and install its dependencies:

```bash
cd Moving-MNIST
python -m pip install -r requirements.txt
python datasets/download.py
python datasets/preprocess.py --download_mnist --overwrite
python run.py
```

All runners support command-line overrides. Boolean options use paired flags such as `--train_gemininet` and `--no-train_gemininet`.

