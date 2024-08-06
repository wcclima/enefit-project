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

The raw data used in this project and its detailed documentation can be found in the Kaggle competition __[website](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/data)__. The provided dataset contains target, client, historical weather, forecasted weather, gas prices and electricity prices. (*Note: all provided hourly data are given in EET/EEST timezones, except for the historical weather data which is given in the UTC+3 timezone without DST.*) We note that not all data is available at the same time of when the forecasts are made. This is capture in each dataset by the `data_block_id` feature. Below we give a brief overview of the target and client datasets.

### Target data

The target dataset consists of the electricity consumption and solar energy production hourly time series of 69 `prediction_unit_id`'s. Each `prediction_unit_id` is a set of prosumers aggregated according to the `county_id`, `product_type` and `is_business` client features. For the complete dictionary between `prediction_unit_id` and these features, see [here](https://github.com/wcclima/enefit-project/blob/main/data/prediction_unit_id_dictionary.csv).
![cons_prod_series](https://github.com/user-attachments/assets/466a2ce7-8f8e-4372-b2a7-c8b5284a4569)
*<p align="center"> Plot of the energy consumption (cons) and production (prod) hourly time series for each `prediction_unit_id`. </p>*

The plot of the target distribution for energy consumption and energy production of both business and private prosumers shows that these distributions are highly skewed.
![cons_prod_target_dist](https://github.com/user-attachments/assets/402afa75-2ed5-4859-bc7c-eb2f7a7736fe)
*<p align="center"> Plot of the target distribution for energy consumption and production. </p>*

### Client data

For the client data we highlight the `installed_capacity` and `eic_count` features. The former corresponds to the daily time series for the installed photovoltaic capacity while the latter is the __[Electricity Identification Code](https://en.wikipedia.org/wiki/Energy_Identification_Code)__ (EIC) count, i.e. the number of prosumers, for each `prediction_unit_id` value.
![installed_capacity_vs_eic_count](https://github.com/user-attachments/assets/5f61d1bb-a8eb-47c6-afb0-87600c2b32c3)
*<p align="center"> Plot of installed capacity versus EIC count for all units. </p>*

Aggregating according to `prediction_unit_id` and discriminating according to the client feature `is_business`, we see that these quantities are strongly correlated, as expected. Hence in the modelling part we should pick one or the other in order to avoid overfitting. Moreover, from the higher slope we see that business prosumers have a higher installed photovoltaic capacity per prosumer on average than private prosumers, as expected.

The client dataset is available with a delay of two days with respect to the target data due to the lack synchroneity of part of the data at the time of the forecasting mentioned above and has to be inputted, see next section. 

## Model

 - **Method:** We approached the problem of forecasting the hourly electricity consumption and solar energy production of the prosumers as a cross-sectional problem, i.e. we take the target time series as a function of the various feature time series, such as the photovoltaic installed capacity, EIC count, electricity prices, atmospheric temperature, etc. We note that energy consumption and production are very different processes and that consumption and production of businesses and private happen at different scales. For these reasons we choose to model to build four models for each of the combinations of the features `is_consumption` and `is_business`.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; We note that not all data is available at the same time at the time of the forecasting. 

 - **Target preprocessing:** We notice that for solar energy production, and useful quantity is the __[*capacity factor*](https://en.wikipedia.org/wiki/Capacity_factor)__ which is the ratio of the actual energy output for a certain period of time and the installed photovoltaic capacity. Here 1 hour is the relevant period of time, and the relevant capacity factor definition is

$$
\textrm{capacity factor} = \frac{\textrm{hourly energy production}}{(\textrm{installed capacity})\times (\textrm{1 hour})}.
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Experimentation has also shown that in the case of energy consumption the models for both business and private prosumers perform\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; better if we take consumption per EIC count as the target:

$$
\textrm{consumption per EIC count} = \frac{\textrm{hourly energy consumption}}{\textrm{daily EIC count}}.
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Moreover, since energy consumption and production for both busines and private prosumers are highly skewed, it is convenient have the\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; target in a log scale.

- **Feature Engineering:** We create number of new features which are listed below and input the client and weather data that are missing at the time of the forecasting.
   - Solar azimuth and altitude: we use the __[suncal](https://pypi.org/project/suncalc/)__ library to create `solar_azimuth` and `solar_altitude` features, relevant for the solar energy production.
   - Categorical time features: we create `hour` (24 categories), `month` (12 categories) and `is_weekend` (`True`/`False`) as categorical features, to capture seasonal effects in energy consumption.
   - Numerical time features: we create the `hour_number` and `day_number` features counting the number of days and hours to capture seasonal effects in energy consumption.
   - Client features: we input two days of data for the `installed_capacity` and `eic_count` features by interpolating the corresponding daily time series.
   - Holiday features: we create the `is_holiday` and `is_school_holiday` features to flag national holidays and school holidays days in Estonia, for the former we used the __[holidays](https://pypi.org/project/holidays/)__ library and __[this](https://www.holidays-info.com/estonia/school-holidays/)__ reference for the latter.
   - Lagged electricity prices: we create `euros_per_mwh_{weeks}_weeks_lag` with weekly lag of `weeks` = 4, 8, 12, 16 and drop `euros_per_mwh` as we assume that the electricity prices influences comsumption and production with some delay.
   - Forecasted weather features: we use the forecasted weather data to input the missig historical weather data at the time of the energy consumption and production forecast. 
     - The all features are averaged over all the weather stations within each county and for each timestamp.
     - We create `rain`, `windspeed_10m`, `winddirection_10m`, `shortwave_radiation` and `diffuse_radiation` features and change the scale of the `snowfall` and `cloudcover_[high/medium/low]` features so the historical and forecasted weather datasets are comparable.
   - Historical weather features: we use the historical weather dataset as features to our model and the forecasted weather dataset whenever inputting is needed. 
     - The all features are averaged over all the weather stations within each county and for each timestamp.
   - Irradiation-temperature feature: we create the `irradiance_over_temperature` feature, relevant for solar energy production, based on this __[thread](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/discussion/468654)__.
   - Cell temperature feature: we create the `cell_temperature_low_efficiency`, `cell_temperature_medium_efficiency` and `cell_temperature_high_efficiency` to model the actual temperature at the solar cell for different efficiencies, see __[here](`cell_temperature_low_efficiency`)__.
   - Demographic feature: we create the `is population_over_100k` to flag counties with higher population, relevant for both energy consumption and production.
   - Lagged target features: we create `target_{day_lag}_days_lag` and `target_{day_lag}_days_lag_flip_is_cons` with daily lag of `day_lag` = 5, 6, 7, 14, 28, 42; the former is just the lagged target while the latter is the lagged target with the `is_consumption` flag flipped.
   - We do not use the gas prices data. 

- **Missing data input:** Both the client and historical weather datasets are available with some delay with respect to the the target data and have missing values when the target forecast is made.
  - *Client data missing values*: the `installed_capacity` and `eic_count` features are delayed by 2 days with respect to the target data we simply interpolate the missing data.
  - *Historical weather data missing values*: the weather features are delayed by 37 hours with repect to the target; we adopt two strategies to input these values using the forecasted weather data.
    - We choose to input the missing values for the `temperature`, `dewpoint`, `rain`, `snowfall`, `windspeed_10m` and `winddirection_10m` features from the forecasted weather dataset.
    - We choose to train a __[XGBoost](https://xgboost.readthedocs.io/en/stable/)__ model having the forecasted weather dataset as independent variables and the historical `surface_pressure`, `cloudcover_[low/mid/high/total]`, `shortwave_radiation`, `direct_solar_radiation` and `diffuse_radiation` features as dependent variables and then use the the XGBoost model's prediction to inpute the missing values of those historical weather features.  

- **Model Architecture:** we train four independent __[CatBoost](https://catboost.ai/)__, one for each of the combinations of energy consumption/production and business/private prosumer. 

- **Metrics:**
  - The metric considered for the competition is the *mean absolute error*.
  - We use the *root mean squared error* as the loss function for each one of the four CatBoost models.
  - We use the __[Optuna](https://optuna.org/)__ hyperparameter optimisation package with the *mean absolute error* adjusted to a target in a log scale as our evaluation metric.  

## Results

TO DO
