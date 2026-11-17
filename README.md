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


## Repository Structure

```text
GeminiNet/
├── Bair-Robot-Pushing/
├── ECMWF-ERA5/
├── KTH-Action/
├── MeteoNet-NW/
├── Moving-MNIST/
├── NOAA-GOES16/
├── SEVIR-IR107/
├── UTD-MHAD/
│
├── Datasets/
│   ├── Bair-Robot-Pushing/
│   ├── ECMWF-ERA5/
│   ├── KTH-Action/
│   ├── MeteoNet-NW/
│   ├── Moving-MNIST/
│   ├── NOAA-GOES16/
│   ├── SEVIR-IR107/
│   ├── UTD-MHAD/
│   └── README.md
│
└── README.md
```

Each dataset project is self-contained and follows a consistent structure:

```text
<dataset-project>/
├── data_provider/
│   ├── __init__.py
│   ├── <dataset-specific-loader>.py
│   └── normalizer.py
│
├── datasets/
│   ├── README.md
│   ├── download.py
│   └── preprocess.py
│
├── models/
│   ├── __init__.py
│   ├── encoders.py
│   ├── gemininet.py
│   ├── heads.py
│   ├── interaction.py
│   ├── reliability.py
│   └── unimodal.py
│
├── plot/
│   ├── plot_metrics.py
│   └── plot_training_overview.py
│
├── scripts/
│   └── <dataset-specific-script>.py
│
├── utils/
│   ├── __init__.py
│   ├── auto_plot.py
│   ├── checkpoint.py
│   ├── common.py
│   ├── logger.py
│   ├── metrics.py
│   ├── model_summary.py
│   └── visualization.py
│
├── README.md
├── requirements.txt
├── run.py
└── trainer.py
```

The main directories serve the following purposes:

| Directory / File | Description |
|---|---|
| `data_provider/` | Dataset-specific data loading and normalization |
| `datasets/` | Dataset download and preprocessing utilities |
| `models/` | GeminiNet architecture and model components |
| `plot/` | Training and evaluation visualization |
| `scripts/` | Dataset-specific preprocessing and preparation scripts |
| `utils/` | Training utilities, checkpoint management, metrics, and visualization |
| `trainer.py` | Model training and evaluation workflow |
| `run.py` | Main experiment entry point |
| `requirements.txt` | Project dependencies |
| `README.md` | Dataset-specific documentation |

The `Datasets/` directory centralizes dataset download and preprocessing resources, while each dataset project maintains its own implementation, configuration, and execution pipeline.


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

