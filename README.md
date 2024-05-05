# Predict energy behavior of prosumers

This project was based on the __[Kaggle](https://www.kaggle.com)__ competition __[Enefit - Predict Energy Behavior of Prosumer](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/overview)__, sponsored by __[Eesti Energia](https://www.energia.ee/en/avaleht)__. It was developed during December 2023-XXXX 2024 by __[William Lima](https://www.linkedin.com/in/william-lima-363475160/)__, __[Marco Pichl](https://www.linkedin.com/in/marco-pichl-2b8300226/)__, __[Jannis Weber](https://www.linkedin.com/in/jannis-weber/)__ and __[Rakib Rahman](https://www.linkedin.com/in/rakibur-rahman-phd-18877446/)__. 

A preliminary version of this project was presented at __[Spiced Academy](https://www.spiced-academy.com/en/program/data-science)__ Data Science Bootcamp, Berlin, as the final project of three of us (Lima, Pichl and Weber) on 22 January 2024.

## Business problem

The number of prosumers is rapidly increasing, and solving the problems of energy imbalance and their rising costs is vital. If left unaddressed, this could lead to increased operational costs, potential grid instability, and inefficient use of energy resources. If this problem were effectively solved, it would significantly reduce the imbalance costs, improve the reliability of the grid, and make the integration of prosumers into the energy system more efficient and sustainable. Moreover, it could potentially incentivize more consumers to become prosumers, knowing that their energy behavior can be adequately managed, thus promoting renewable energy production and use.

## Objective

The main goal of this project was to predict the amount of electricity produced and consumed by Estonian energy customers who have installed solar panels (prosumer) from weather, energy prices and installed photovoltaic capacity data.

## Repo organisation

**`app`: App for the model**

- `enefit_prediction.py` : Streamlit app for the prediction model.

**`eda_notebooks`: Exploratory Data Analysis (EDA) notebooks**

- `EDA_target_part1.ipynb` : exploration for the target data, containing correlations analysis and data cleaning for summer time missing values and outliers.
- `EDA_target_part2.ipynb` : further exploration for the target data, containing analysis of the target distribution w.r.t. the client profile and in the log scale.
- `EDA_weather.ipynb` : exploration of the weather data and imputation of missing values.
- `EDA_client.ipynb` : exploration of the client data, with analysis of the distribution and correlations of the client count and solar panel installed capacity.
- `EDA_electricity_gas` : exploration of the energy prices data, with analysis of the summer time missing values, distribution and correlation of lagged values with target. 

**`modelling`: Modelling notebooks**

- `FeatureEngineering.ipynb`: use the modules `data_loading` and `featuring_engineering` to load and process the features, see below.
- `ModelProduction.ipynb`: model building for solar energy production, with the analysis of the correlations between features and features and target and feture importance for a CatBoost model.   
- `ModelConsumption.ipynb`: model building for electricity consumption, with the analysis of the correlations between features and features and target and feture importance for a CatBoost model.

**`models`: Hyperparameter tunning notebooks**

- `ConsumptionBusinessModelParameterTuing.ipynb` : hyperparameter tuning of a CatBoost model for business electricity consumption.
- `ConsumptionPrivateModelParameterTuing.ipynb` : hyperparameter tuning of a CatBoost model for private electricity consumption.
- `ProductionBusinessModelParameterTuing.ipynb` : hyperparameter tuning of a CatBoost model for business solar energy production.
- `ProductionPrivateModelParameterTuing.ipynb` : hyperparameter tuning of a CatBoost model for private solar energy production.

**`modules`: Model architecture**

- `data_loading.py` : module that loads the feature and target data.
- `featuring_engineering.py` : module that create new features relevant to our model (see model description below) and join the various dataset.
- `model_building.py` : module responsible for training our model.
- `mock_api.py` : module that simulates the competition API. It loads new unseen data used to the prediction part of the model.

## Requirements:

- pyenv with Python: 3.9.4

### Environment setup for `macOS`

```BASH
make setup

#or

pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Environment setup for `WindowsOS`

For `Git-bash` CLI:
```
pyenv local 3.11.3
python -m venv .venv
source .venv/Scripts/activate
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` file contains the libraries needed for EDA and modelling.

## Data

The raw data used in this project and there detailed documentation can be found in the competition __[website](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/data)__.

TO DO

## Model

TO DO

## Results

TO DO
