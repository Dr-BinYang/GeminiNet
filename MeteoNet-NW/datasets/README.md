# MeteoNet north-west data

Official source: https://meteonet.umr-cnrm.fr/dataset/

From the project root:

```bash
python datasets/download.py
python datasets/preprocess.py --force
```

The downloader retrieves all available north-west rainfall and ground-station archives used by this project. The preprocessor aggregates radar frames and station observations to aligned hourly sequences. Raw and processed data are ignored by Git.