# DL-Wine: Predicting Wine Quality from Weather Patterns

This repository contains a set of scripts and notebooks used to predict
wine quality from historical weather observations. Weather data from Météo‑Franc
e is merged with wine ratings scraped from Vivino to train deep learning models
capable of anticipating whether a vintage will be good given the year's climate.

---

## Repository Contents

- **Data collection and preprocessing** scripts
- **Deep learning models** for tabular data
- **Jupyter notebooks** for analysis and experimentation

  ***

## Installation

First install the Python dependencies:

```bash
$env:PYTHONPATH = (Get-Location).Path
pip install -r requirements.txt
```

The main packages used are `pandas`, `numpy`,`optuna`,`polars`,`plotly` and `torch`.

---

## Data Overview

### Weather

Weather observations are organised by French department and cover 1950–2 025. The raw and processed CSV/Parquet files can be found at:
[Météo France](https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes/)
Weather observations are organised by French department and cover 1950–2 025. The raw and processed CSV/Parquet files are located in:

```
data/weather/
data/weather_by_year/
data/weather_by_year_cleaned/
```

### Wine

Wine ratings scraped from Vivino are stored in `data/Wine/`. The region
metadata is provided in `data/Wine/regions_corrected.csv`.
An interactive map of the wine regions can be found [🗺️ here](https://lucasponcet.github.io/DL_Project/wine_map.html) !

---

## Quick Start

Load the datasets with `pandas`:

```
python
import pandas as pd

wine = pd.read_csv('data/out/vivino_wines_with_weather_AOC.csv')
weather = pd.read_parquet('data/weather_by_year/weather_all_stations_2010.parquet')

print(wine.head())
print(weather.head())
```

---

## Project Structure

```
DL_Project
├─ data/                   # Raw and processed datasets
├─ src/
│  ├─ model/               # Training scripts and model classes
│  ├─ preprocessing/       # Notebooks and scripts to prepare data
│  └─ scrapper/            # Wine rating scrapers
├─ models/                 # Saved model weights
└─ docs/                   #  HTML map
```

---

## Workflow

### Data Preprocessing

Generate weather features:

```bash
cd src/preprocessing/Weather
jupyter notebook features_weather.ipynb
```

Merge Wine and Weather data:

```bash
cd src/preprocessing/"Wine & Weather"
python build_wines_coord.py
```

### Model Training

Train and evaluate the predictive model:

```bash
cd src/model
python MLP.py
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/new-analysis`
3. Commit your changes: `git commit -m 'Add new analysis'`
4. Push your branch: `git push origin feature/new-analysis`
5. Create a Pull Request

---

## License

This project is licensed under the MIT License. See the `License` file for details.

---

Enjoy predicting wine quality!
