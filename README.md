# Predict energy behavior of prosumers

This project was based on the __[Kaggle](https://www.kaggle.com)__ competition __[Enefit - Predict Energy Behavior of Prosumer](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/overview)__, sponsored by __[Eesti Energia](https://www.energia.ee/en/avaleht)__. It was developed during December 2023-Feb 2024 by __[William Lima](https://www.linkedin.com/in/william-lima-363475160/)__, __[Marco Pichl](https://www.linkedin.com/in/marco-pichl-2b8300226/)__, __[Jannis Weber](https://www.linkedin.com/in/jannis-weber/)__ and __[Rakib Rahman](https://www.linkedin.com/in/rakibur-rahman-phd-18877446/)__. 

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
- `featuring_engineering.py` : module that cleans the data and creates new features relevant to our model (see model description below) and join the various dataset.
- `model_building.py` : module responsible for training our model.
- `mock_api.py` : module that simulates the competition API. It loads new unseen data used to the prediction part of the model.

## Data

The raw data used in this project and its detailed documentation can be found in the Kaggle competition __[website](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/data)__. The provided dataset contains target, client, historical weather, forecasted weather, gas prices and electricity prices. Below we give a brief overview of the target and client datasets.

### Target data

The target dataset consists of the electricity consumption and solar energy production hourly time series of 69 `prediction_unit_id`'s. Each `prediction_unit_id` is a set of prosumers aggregated according to the `county_id`, `product_type` and `is_business` client features. For the complete dictionary between `prediction_unit_id` and these features, see [here](https://github.com/wcclima/enefit-project/blob/main/data/prediction_unit_id_dictionary.csv).
![cons_prod_series](https://github.com/user-attachments/assets/466a2ce7-8f8e-4372-b2a7-c8b5284a4569)
*<p align="center"> Plot of the energy consumption (cons) and production (prod) hourly time series for each `prediction_unit_id`. </p>*

The plot of the target distribution aggregated by `prediction_unit_id`'s for energy consumption and energy production shows that both distributions are highly skewed.
![cons_prod_target_dist](https://github.com/user-attachments/assets/abb9f10e-a494-4581-9a08-992565130870)
*<p align="center"> Plot of the target distribution for energy consumption and production. </p>*

### Client data

For the client data we highlight the `installed_capacity` and `eic_count` features. The former corresponds to the daily time series for the photovoltaic installed capacity while the latter is the __[Electricity Identification Code](https://en.wikipedia.org/wiki/Energy_Identification_Code)__ (EIC) count, i.e. the number of prosumers, for each `prediction_unit_id` value. Aggregating according to `prediction_unit_id` and discriminating accordinf to the client feature `is_business`, we see that these quantities are strongly correlated, as expected. Hence in the modelling part we should pick one or the other in order to vaid overfitting.
![installed_capacity_vs_eic_count](https://github.com/user-attachments/assets/015f90b0-771f-4175-b026-c82b7794899e)
*<p align="center"> Plot of installed capacity versus EIC count for all units. </p>*

We notice that for solar energy production, and useful quantity is the __[*capacity factor*](https://en.wikipedia.org/wiki/Capacity_factor)__ which is the ratio of the actual energy output for a certain period of time and the photovoltaic installed capacity. Here 1 hour is the relevant period of time, and the relevant capacity factor definition is

$$
\textrm{capacity factor} = \frac{\textrm{energy output}}{(\textrm{installed capacity})\times (\textrm{1 hour})}.
$$



## Model

TO DO

## Results

TO DO
