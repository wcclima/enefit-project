import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, Pool


class ModelBuilding:
    def __init__(self):
        self.name = "CatBoostRegressorModelsForEnergyProsumption"
        self.is_fitted = False

        self.cat_features = [
            "county",
            "is_business",
            "product_type",
            "is_consumption",
            "hour",
            "month",
            "is_weekend",
            "is_holiday",
            "is_school_holiday",
            "is_population_over_100k"
        ]
    
        self.num_features = [
            'temperature',
            'dewpoint',
            'rain',
            'snowfall',
            'surface_pressure',
            'cloudcover_total',
            'cloudcover_low',
            'cloudcover_mid',
            'cloudcover_high',
            'windspeed_10m',
            'winddirection_10m',
            'shortwave_radiation',
            'direct_solar_radiation',
            'diffuse_radiation',
            'installed_capacity',
            'euros_per_mwh_electricity',
            'lowest_price_per_mwh_gas',
            'highest_price_per_mwh_gas',
            'mean_price_per_mwh_gas',
            'variation_price_per_mwh_gas',
            'solar_altitude',
            'solar_azimuth',
            'day_number',
            'hour_number',
            'eic_count'
        ]
        
        self.cat_num_features = self.cat_features + self.num_features


        self.model_parameters = {
            "params_business_consumption" : {
                "loss_function" : "RMSE",
                "n_estimators": 1032, 
                "learning_rate": 0.06766662731954433, 
                "depth": 9, 
                "ignored_features": ["shortwave_radiation", 
                                     "eic_count", 
                                     "lowest_price_per_mwh_gas", 
                                     "highest_price_per_mwh_gas"], 
                "cat_features": self.cat_features
            },
            "params_business_production" : {
                "loss_function" : "RMSE",
                "n_estimators": 1135, 
                "learning_rate": 0.05658844144431656, 
                "depth": 9, 
                "ignored_features": ["shortwave_radiation", 
                                     "installed_capacity", 
                                     "hour_number", 
                                     "lowest_price_per_mwh_gas", 
                                     "highest_price_per_mwh_gas"], 
                "cat_features": self.cat_features
            },
            "params_private_consumption" : {
                "loss_function" : "RMSE",
                "n_estimators": 1126, 
                "learning_rate": 0.04354171311602304, 
                "depth": 9, 
                "ignored_features": ["shortwave_radiation", 
                                     "eic_count", 
                                     "hour_number", 
                                     "lowest_price_per_mwh_gas", 
                                     "highest_price_per_mwh_gas"], 
                "cat_features": self.cat_features
            },
            "params_private_production" : {
                "loss_function" : "RMSE",
                "n_estimators": 1134, 
                "learning_rate": 0.029888171355150536, 
                "depth": 7, 
                "ignored_features": ["shortwave_radiation", 
                                     "installed_capacity", 
                                     "lowest_price_per_mwh_gas", 
                                     "highest_price_per_mwh_gas"
                                     ], 
                "cat_features": self.cat_features
            }
        }

        self.model_business_consumption = CatBoostRegressor(
            **self.model_parameters.get(
                'params_business_consumption'
            ), 
            silent = True
        ) 
        self.model_business_production = CatBoostRegressor(
            **self.model_parameters.get(
                'params_private_consumption'
            ), 
            silent = True
        )
        self.model_private_consumption = CatBoostRegressor(
            **self.model_parameters.get(
                'params_business_production'
            ), 
            silent = True
        )
        self.model_private_production = CatBoostRegressor(
            **self.model_parameters.get(
                'params_private_production'
            ), 
            silent = True
        )

    def fit(self, df):
        
        mask_business_consumption = (
            (df_train_features.is_business == 1)&(df_train_features.is_consumption == 1)
        )
        y_business_consumption = (
            np.log1p(
                df_train_features[mask_business_consumption].target/df_train_features[mask_business_consumption].eic_count
            )
        )
        data_business_consumption = Pool(
            data = df_train_features[mask_business_consumption][self.cat_num_features], 
            label = y_business_consumption, 
            cat_features = self.cat_features
        )
        self.model_business_consumption.fit(
            data_business_consumption
        )
        

        mask_business_production = (
            (df_train_features.is_business == 1)&(df_train_features.is_consumption == 0)
        )
        y_business_production = (
            np.log1p(
                df_train_features[mask_business_production].target/df_train_features[mask_business_production].installed_capacity
            )
        )
        data_business_production = Pool(
            data = df_train_features[mask_business_production][self.cat_num_features],
            label = y_business_production,
            cat_features = self.cat_features
        )
        self.model_business_production.fit(
            data_business_production
        )


        mask_private_consumption = (
            (df_train_features.is_business == 0)&(df_train_features.is_consumption == 1)
        )
        y_private_consumption = (
            np.log1p(
                df_train_features[mask_private_consumption].target/df_train_features[mask_private_consumption].eic_count
            )
        )
        data_private_consumption = Pool(
            data = df_train_features[mask_private_consumption][self.cat_num_features],
            label = y_private_consumption,
            cat_features = self.cat_features
        )
        self.model_private_consumption.fit(
            data_private_consumption
        )


        mask_private_production = (
            (df_train_features.is_business == 0)&(df_train_features.is_consumption == 0)
        )
        y_private_production = (
            np.log1p(
                df_train_features[mask_private_production].target/df_train_features[mask_private_production].installed_capacity
            )
        )
        data_private_production = Pool(
            data = df_train_features[mask_private_production][self.cat_num_features],
            label = y_private_production,
            cat_features = self.cat_features 
        )
        self.model_private_production.fit(
            data_private_production
        )

        self.is_fitted = True

    def predict(self, df):

        predictions = np.zeros(len(df_features))

        mask_business_consumption = (
            (df_features.is_business == 1)&(df_features.is_consumption == 1)
        )
        pred_business_consumption = (
            self.model_business_consumption.predict(
                df_features[mask_business_consumption][self.cat_num_features]
            )
        )

        mask_business_production = (
            (df_features.is_business == 1)&(df_features.is_consumption == 0)
        )
        pred_business_production = (
            self.model_business_production.predict(
                df_features[mask_business_production][self.cat_num_features]
            )
        )

        mask_private_consumption = (
            (df_features.is_business == 0)&(df_features.is_consumption == 1)
        )
        pred_private_consumption = (
            self.model_private_consumption.predict(
                df_features[mask_private_consumption][self.cat_num_features]
            )
        )

        mask_private_production = (
            (df_features.is_business == 0)&(df_features.is_consumption == 0)
        )
        pred_private_production = (
            self.model_private_production.predict(
                df_features[mask_private_production][self.cat_num_features]
            )
        )

        eic_count_business_consumption = (
            df_features[mask_business_consumption].eic_count.to_numpy()
        )
        eic_count_private_consumption = (
            df_features[mask_private_consumption].eic_count.to_numpy()
        )

        installed_capacity_business_production = (
            df_features[mask_business_production].installed_capacity.to_numpy()
        )
        installed_capacity_private_production = (
            df_features[mask_private_production].installed_capacity.to_numpy()
        )

        predictions[mask_business_consumption.values] = (
            np.expm1(
                pred_business_consumption
            )*eic_count_business_consumption
        )
        predictions[mask_business_production.values] = (
            np.expm1(
                pred_business_production
            )*installed_capacity_business_production
        )
        predictions[mask_private_consumption.values] = (
            np.expm1(
                pred_private_consumption
            )*eic_count_private_consumption
        ) 
        predictions[mask_private_production.values] = (
            np.expm1(
                pred_private_production
            )*installed_capacity_private_production
        )

        return predictions

