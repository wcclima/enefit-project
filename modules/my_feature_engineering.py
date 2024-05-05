import numpy as np
import pandas as pd
from suncalc import get_position
import holidays
from datetime import datetime
from xgboost import XGBRegressor

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)



class FeatureEngineering:
    
    def __init__(self, data_loading):
        self.data = data_loading

       
    def fill_dst_nan(self, df_, limit = 1):
        cols_to_fill = df_.dtypes[
            df_.dtypes == "float64"
        ].index.to_list()
        
        df_.loc[df_.index, cols_to_fill] = 0.5*(
            df_[cols_to_fill].ffill(limit=limit) + df_[cols_to_fill].bfill(limit=limit)
        )
        
        return df_
    
    
    def get_tallinn_timezone(self,df_):
        start_date = df_.datetime.min()
        end_date = df_.datetime.max()
        dt = pd.date_range(
            start = start_date, 
            end = end_date, 
            freq = "h", 
            tz = "Europe/Tallinn"
        )
        datetime_df = pd.DataFrame(
            data={'datetime_new' : dt.tz_localize(None)}
        )
        df_ = df_.merge(
            datetime_df, 
            how = 'right', 
            left_on = "datetime", 
            right_on = 'datetime_new'
        )
        
        df_.drop(
            columns = "datetime", 
            inplace = True
        )
        df_ = df_.rename(
            columns = {'datetime_new' : "datetime"}
        )
        
        if "prediction_unit_id" in df_.columns:
            df_ = df_.sort_values(
                by = ["is_consumption", 
                      "prediction_unit_id", 
                      "datetime"
                     ]
            )
            df_["datetime"] = (
                df_.datetime.dt.tz_localize(
                    "Europe/Tallinn", 
                    ambiguous = "infer"
                )
            )
            df_ = self.fill_dst_nan(df_, 2)
            df_ = df_.sort_values(
                by = "row_id"
            )
        elif "county" in df_.columns:
            df_ = df_.sort_values(
                by = ["county", 
                      "datetime"
                     ]
            )
            df_["datetime"] = (
                df_.datetime.dt.tz_localize(
                    "Europe/Tallinn", 
                    ambiguous = "infer"
                )
            )
            df_ = df_.sort_values(
                by = ["datetime", 
                      "county"
                     ]
            )
        else:
            df_["datetime"] = (
                df_.datetime.dt.tz_localize(
                    "Europe/Tallinn", 
                    ambiguous = "infer"
                )
            )

        
        return df_
    
    def convert_to_tallinn_tz(self, df_):
        df_["datetime"] = (
            df_.datetime.dt.tz_localize(
                "Europe/Moscow"
            )
        )
        df_["datetime"] = (
            df_.datetime.dt.tz_convert(
                "Europe/Tallinn"
            )
        )
        
        df_ = (
            self.fill_dst_nan(df_)
        )
        
        return df_
    
    
    def get_county_loc(self, county):
        return self.data.county_loc.get(county)
    
    
    def get_solar_position(self, df_, in_degrees = True):
        df_county_loc = pd.DataFrame(
            self.data.county_loc
        ).T.rename(
            columns = {0: "latitude", 
                       1: "longitude"
                      }
        ).reset_index(
            names = "county"
        )
        df_sol_pos = df_[["datetime", "county"]].copy()
        df_sol_pos = df_sol_pos.merge(
            df_county_loc, 
            how = "left", 
            on="county"
        )
        df_sol_pos[["solar_azimuth", 
                    "solar_altitude"
                   ]] = pd.DataFrame(
            get_position(
                df_sol_pos.datetime.dt.tz_convert("UTC"), 
                df_sol_pos.longitude, 
                df_sol_pos.latitude
            )
        )
        df_sol_pos.drop(
            columns = ["latitude", 
                       "longitude"
                      ], 
            inplace = True
        )
        
        if in_degrees:
            df_sol_pos = df_sol_pos.assign(
                solar_azimuth = np.degrees(df_sol_pos.solar_azimuth), 
                solar_altitude = np.degrees(df_sol_pos.solar_altitude)
            )
        
        return df_sol_pos
        
    
    def _add_solar_features(self, df_feat):
        
        solar_pos_cols = ["solar_azimuth", 
                          "solar_altitude"]
        df_feat = pd.concat(
            [df_feat,
             self.get_solar_position(
                 df_feat
             )[solar_pos_cols]
            ], 
            axis = 1
        )        
                        
        return df_feat
    
    
    def _add_cat_time_features(self, df_feat):
                
        df_feat = df_feat.assign(hour = df_feat.datetime.dt.hour,
                                 month = df_feat.datetime.dt.month,
                                 weekday = df_feat.datetime.dt.weekday
                                )
        
        df_feat["is_weekend"] = df_feat["weekday"].apply(
            lambda x: 1 if x in [6,7] else 0
        )
        
        df_feat = df_feat.drop(columns = "weekday")
        df_feat["hour"] = df_feat.hour.astype('int64')
        df_feat["month"] = df_feat.month.astype('int64')
        df_feat["is_weekend"] = df_feat.is_weekend.astype('int64')
        
        return df_feat
    
    
    def _add_num_time_features(self, df_feat):
        
        start_time = df_feat.datetime.min()
        delta_t = df_feat["datetime"].copy() - start_time
        df_feat["day_number"] = [x.days for x in delta_t]
        df_feat["hour_number"] = [(x.days*24 + x.seconds//3600) for x in delta_t]

        df_feat["day_number"] = df_feat.day_number.astype('int64')
        df_feat["hour_number"] = df_feat.hour_number.astype('int64')
        
        return df_feat

    
    def _add_client_features(self, df_feat):
        df_client = self.data.df_client
        
        df_feat = df_feat.merge(
            df_client,
            on=["county", 
                "is_business", 
                "product_type", 
                "date"
               ],
            how="left"
        )
        
        pred_ids = pd.unique(
            df_feat.prediction_unit_id
        ).tolist()
        
        for id_ in pred_ids:
            mask_id = df_feat.prediction_unit_id == id_
            for cons in [0, 1]:
                mask_cons = df_feat.is_consumption == cons
                mask = mask_id&mask_cons
                for col in ["eic_count", "installed_capacity"]:
                    aux_series = (
                        df_feat[mask].set_index(
                            "datetime"
                        )[col].interpolate(
                            method="time"
                        )
                    )
                    aux_series.index = df_feat[mask].index
                    df_feat.loc[mask, col] = aux_series
        
        return df_feat
    
    
    def get_new_gas_features(self, df_):
        for weeks in range(1,25):
            lag = 7*weeks
            df_[f"mean_price_per_mwh_{weeks}_weeks_lag"] = 0.5*(
                df_.lowest_price_per_mwh.shift(
                    periods=lag
                ) 
                + df_.highest_price_per_mwh.shift(
                    periods=lag
                )
            )
        
        return df_
    
    
    def _add_gas_features(self, df_feat):
        df_gas_prices = self.data.df_gas_prices
        df_gas_prices = df_gas_prices.rename(
            columns = {"forecast_date": "date"}
        )
        df_dates = pd.DataFrame(
            {"date" : pd.date_range(
                start=df_feat.date.min(), 
                end = df_feat.date.max()
            )
            }
        )
        df_gas_prices = df_gas_prices.merge(
            df_dates, 
            how= "right", 
            left_on = "date", 
            right_on = "date", 
        )
        df_gas_prices = self.get_new_gas_features(df_gas_prices)

        df_feat = df_feat.merge(
            df_gas_prices,
            on="date",
            how="left"
        )
                
        return df_feat
    
    
    def _add_electricity_features(self, df_feat):
        df_electricity_prices = self.data.df_electricity_prices.rename(
            columns = {"forecast_date": "datetime"}
        )
        df_electricity_prices["datetime"] = (
            df_electricity_prices.datetime + pd.Timedelta(1, "h")
        )
        
        df_electricity_prices = (
            self.get_tallinn_timezone(
                df_electricity_prices
            )
        )
        df_dates = pd.DataFrame(
            {"datetime" : pd.date_range(
                start=df_feat.datetime.min(), 
                end = df_feat.datetime.max(),
                freq = "h",
                tz = "Europe/Tallinn"
            )
            }
        )
        df_electricity_prices = df_electricity_prices.merge(
            df_dates, 
            how= "right", 
            left_on = "datetime", 
            right_on = "datetime", 
        )
        for weeks in [4, 8, 12, 16]:
            lag = 24*7*weeks
            df_electricity_prices[f"euros_per_mwh_{weeks}_weeks_lag"] = (
                df_electricity_prices.euros_per_mwh.shift(
                    periods=lag
                )
            )
            
        df_feat = df_feat.merge(
            df_electricity_prices,
            on="datetime",
            how="left"
        )
        
        return df_feat
    
    
    def get_county(self, lat, lon):
        for k,v in self.data.station_to_county.items():
            if (lat,lon)==k:
                return int(v)
    
    
    def get_countywise_weather(self, df_):
        df_["county"] = df_.apply(
            lambda x: self.get_county(x.latitude,x.longitude),
            axis=1
        )
        df_ = df_.groupby(
            by = ["county","datetime"], 
            as_index=False
        ).mean().round(2)
        df_.reset_index(drop = True, inplace=True)
        df_["county"] = df_["county"].astype('int64')

        return df_
    
    
    def get_new_forecast_features(self, df_):
        
        df_["rain"] = df_.total_precipitation*1e3 - df_.snowfall*1e3
        df_["snowfall"] = df_.snowfall*1e2
        df_["windspeed_10m"] = np.sqrt(df_["10_metre_u_wind_component"]**2 
                                       + df_["10_metre_v_wind_component"]**2
                                      )
        df_["winddirection_10m"] = (np.degrees(
            np.arctan2(df_["10_metre_u_wind_component"], 
                       df_["10_metre_v_wind_component"]
                      )
        ) + 180.
                                   )
        df_["cloudcover_high"] = 100.*df_.cloudcover_high
        df_["cloudcover_mid"] = 100.*df_.cloudcover_mid
        df_["cloudcover_low"] = 100.*df_.cloudcover_low
        df_["cloudcover_total"] = 100.*df_.cloudcover_total
        
        df_ = pd.concat(
            [df_,
             self.get_solar_position(
                 df_
             )[["solar_altitude"]]
            ], 
            axis = 1
        )
        
        df_["solar_altitude"] = np.radians(df_.solar_altitude)
        
        df_["diffuse_radiation"] = (
            df_.surface_solar_radiation_downwards 
            - df_.direct_solar_radiation*np.sin(
                df_.solar_altitude
            )
        )

        df_["shortwave_radiation"] = (
            df_.diffuse_radiation 
            + df_.direct_solar_radiation
        )*np.sin(df_.solar_altitude)
        
        df_.drop(columns=["total_precipitation", 
                          "10_metre_u_wind_component", 
                          "10_metre_v_wind_component", 
                          "surface_solar_radiation_downwards", 
                          "solar_altitude"
                         ], 
                 inplace = True
                )
        
        return df_
    
    def input_weather_model(self, df_, X_cols, Y_cols):
        
        for county in range(16):
            county_mask = df_.county==county
            X = df_[county_mask].dropna()[X_cols]

            for col in Y_cols:
                y = df_[df_.county==county].dropna()[col]
                min_val = y.min()
                y = np.log1p(y - min_val)
            
                model = XGBRegressor(max_depth = 2)
                model_fitted = model.fit(X, y)
    
                nan_mask = df_[col].isnull()
                X_hat = df_[county_mask&nan_mask][X_cols]
                y_hat = model_fitted.predict(X_hat)    
    
                df_.loc[county_mask&nan_mask,col] = (
                    np.expm1(y_hat) + min_val
                ).round(2)
        
        return df_
    
    
    def input_weather_forecast(self, df_, Y_cols):
        for county in range(16):
            county_mask = df_.county==county

            for col in Y_cols:
                nan_mask = df_[col].isnull()
                df_.loc[county_mask&nan_mask,col] = (
                    df_[county_mask&nan_mask][col + "_forecast"]
                )
                
        return df_

    
    def get_weather_data(self):
        df_historical_weather = self.get_countywise_weather(
            self.data.df_historical_weather
            ).drop(
            columns = ["latitude", 
                       "longitude"
                      ]
        )

        df_historical_weather = (
            self.convert_to_tallinn_tz(
                df_historical_weather
            )
        )
        
        df_forecast_weather = self.get_countywise_weather(
            self.data.df_forecast_weather.rename(
                columns={"forecast_datetime" : "datetime"}
            )
        ).drop(
            columns = ["latitude", 
                       "longitude"
                      ]
        )
        
        df_forecast_weather = (
            df_forecast_weather.groupby(
                ["county", 
                 "datetime"
                ], 
                as_index = False
            ).mean()
        )
        
        df_forecast_weather = (
            self.get_tallinn_timezone(
                df_forecast_weather
            )
        )
        
        df_forecast_weather = (self.get_new_forecast_features(
            df_forecast_weather
        )
                              )
        
        df_weather_data = df_forecast_weather.merge(
            df_historical_weather, 
            how = "outer", 
            on = ["county", 
                  "datetime"
                 ], 
            suffixes = ("_forecast", None)
        )
        
        cols_to_input = [
            "temperature",
            "dewpoint",
            "rain",
            "snowfall",
            "windspeed_10m",
            "winddirection_10m"
        ]
        
        cols_to_predict = [
            "surface_pressure",
            "cloudcover_total",
            "cloudcover_low",
            "cloudcover_mid",
            "cloudcover_high",
            "shortwave_radiation",
            "direct_solar_radiation",
            "diffuse_radiation"]
        
        forecast_cols = [elem + "_forecast" for elem in df_forecast_weather.drop(
            columns = ["county",
                       "datetime"
                      ]
        ).columns
                 ]
        
        df_weather_data = (
            self.input_weather_model(
                df_weather_data, 
                forecast_cols, 
                cols_to_predict
            )
        )
        
        df_weather_data = (
            self.input_weather_forecast(
                df_weather_data, 
                cols_to_input
            )
        )
        
        return df_weather_data[df_historical_weather.columns.to_list()] 
        
    
    def _add_weather_features(self, df_feat):
        
        df_weather_data = self.get_weather_data()
        df_feat = df_feat.merge(
            df_weather_data,
            on=[
                "datetime",
                "county"
               ],
            how="left"
        )
        
        # for hours_lag in [24, 48, 72]:
        #     df_feat = df_feat.merge(
        #         df_weather_data.assign(
        #             datetime = df_weather_data.datetime + pd.Timedelta(hours_lag, "h")
        #         ), 
        #         on = [
        #             "datetime", 
        #             "county"
        #         ], how = "left",
        #         suffixes = (None, f"_{hours_lag}hs_lag")
        #     )
        
        return df_feat
    
    
    def _add_irradiance_over_temperature_feature(self, df_feat):
        df_feat["irradiance_over_temperature"] = (
            df_feat.installed_capacity*(
                df_feat.diffuse_radiation + df_feat.direct_solar_radiation*np.sin(
                    np.radians(df_feat.solar_altitude)
                )
            )/(df_feat.temperature + 273.15)
        )
                
        return df_feat
    
    
    def _add_cell_temperature_features(self, df_feat):
        df_feat["cell_temperature_low_efficience"] = (
            df_feat.temperature + 13./800.*(
                df_feat.diffuse_radiation + df_feat.direct_solar_radiation*np.sin(
                    np.radians(df_feat.solar_altitude)
                )
                                          )
        )
        df_feat["cell_temperature_medium_efficience"] = (
            df_feat.temperature + 28./800.*(
                df_feat.diffuse_radiation + df_feat.direct_solar_radiation*np.sin(
                    np.radians(df_feat.solar_altitude)
                )
                                           )
        )
        df_feat["cell_temperature_high_efficience"] = (
            df_feat.temperature + 38./800.*(
                df_feat.diffuse_radiation + df_feat.direct_solar_radiation*np.sin(
                    np.radians(df_feat.solar_altitude)
                )
                                           )
        )
                
        return df_feat
    
     
    def _add_demographic_features(self, df_feat):
        df_is_pop_100k = pd.DataFrame({"county": [i for i in range(16)]})
        df_is_pop_100k["is_population_over_100k"] = df_is_pop_100k.county.apply(
            lambda x: 1 if x in [0, 2, 11] else 0
        )
        
        df_feat = df_feat.merge(
            df_is_pop_100k, 
            how = "left", 
            on = "county"
        )
        
        return df_feat
    
    
    def _add_holiday_features(self, df_feat):
        estonia_holidays = list(
            holidays.country_holidays("EE", years=range(2021, 2026)).keys()
        )
        estonia_holidays = [pd.to_datetime(x) for x in estonia_holidays]
        df_estonia_holidays = pd.DataFrame(
            {"date" : pd.date_range(start = "2021-01-01", end = "2026-12-31")
            }
        )
        df_estonia_holidays["is_holiday"] = df_estonia_holidays.date.apply(
            lambda x: 1 if x in estonia_holidays else 0
        )
        df_estonia_holidays["is_holiday"] = (
            df_estonia_holidays.is_holiday.astype('int64')
        )
        
        school_holidays = [pd.to_datetime(x) for x in self.data.school_holidays]
        df_school_holidays = pd.DataFrame(
            {"date" : pd.date_range(start = "2021-01-01", end = "2026-12-31")
            }
        )
        df_school_holidays["is_school_holiday"] = df_school_holidays.date.apply(
            lambda x: 1 if x in school_holidays else 0
        )
        df_school_holidays["is_school_holiday"] = (
            df_school_holidays.is_school_holiday.astype('int64')
        )
        
        df_feat = df_feat.merge(
            df_estonia_holidays, 
            how = "left", 
            on = "date"
        )
        df_feat = df_feat.merge(
            df_school_holidays, 
            how = "left", 
            on = "date"
        )
        
        return df_feat
    
    
    def _add_target_features(self, df_feat):
        df_train = self.get_tallinn_timezone(
            self.data.df_data
        ).drop(
            columns = [
                "prediction_unit_id", 
                "row_id"
            ]
        )
        
        for day_lag in [5, 6, 7, 14, 28, 42]:
            df_feat = df_feat.merge(
                df_train.assign(
                    datetime = df_train.datetime + pd.Timedelta(day_lag, "d")
                ).rename(
                    columns = {"target" : f"target_{day_lag}_days_lag"}
                ), 
                on = [
                    "datetime", 
                    "county", 
                    "is_business", 
                    "product_type", 
                    "is_consumption"
                ], how = "left"
            )
            
            df_feat = df_feat.merge(
                df_train.assign(
                    datetime = df_train.datetime + pd.Timedelta(day_lag, "d"),
                    is_consumption = np.cos(df_train.is_consumption).astype("int64")
                ).rename(
                    columns = {"target" : f"target_{day_lag}_days_lag_flip_is_cons"}
                ), 
                on = [
                    "datetime", 
                    "county", 
                    "is_business", 
                    "product_type", 
                    "is_consumption"
                ], how = "left"
            )
                        
        return df_feat
        
        
    
    
    def _drop_columns(self, df_feat):
        df_feat = df_feat.drop(
            columns = ["date", "datetime"]
            )
        
        return df_feat

    
    def _join_target(self, df_feat, y):
        cat_cols = [
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
        if y is not None:
            df_feat = pd.concat([df_feat, y], axis=1)

        df_feat[cat_cols] = df_feat[cat_cols].astype("category")
        
        if "row_id" in df_feat.columns:
            df_feat = df_feat.drop(columns = "row_id")
        if "prediction_unit_id" in df_feat.columns:
            df_feat = df_feat.drop(columns = "prediction_unit_id")
            
        df_feat = df_feat.dropna()

        return df_feat
    
    
    def generate_features(self, df_prediction_items, verbose = True):
        self.verbose = verbose

        df_prediction_items = self.get_tallinn_timezone(
            df_prediction_items
        )
        df_prediction_items
        
        if "target" in df_prediction_items.columns:
            df_prediction_items, y = (
                df_prediction_items.drop(columns = "target"),
                df_prediction_items[["target"]],
            )
        else:
            y = None

        df_feat = df_prediction_items
        df_feat["date"] = pd.to_datetime(
            df_feat.datetime.dt.date
        )
        if self.verbose:
            list_added_feats = [
                "solar features", 
                "categorical time features",
                "numerical time features",
                "client features",
#                "gas features", 
                "electricity features", 
                "weather features",
                "irradiance_over_temperature",
                "cell_temperatue",
                "demographics feature", 
                "holiday features",
                "lagged target features"
                # "datetime columns"
            ]
        for i, add_features in enumerate(
            [
           self._add_solar_features,
           self._add_cat_time_features,
           self._add_num_time_features,
           self._add_client_features,
#           self._add_gas_features,
           self._add_electricity_features,
           self._add_weather_features,
           self._add_irradiance_over_temperature_feature,
           self._add_cell_temperature_features,
           self._add_demographic_features,
           self._add_holiday_features,
           self._add_target_features     
           # self._drop_columns,
        ]
        ):
            if self.verbose:
                start_time = datetime.now()
                if i in range(9):
                    print(f"[{start_time.time()}]: Adding " + list_added_feats[i] + "...")
                else:
                    print(f"[{start_time.time()}]: Removing " + list_added_feats[i] + "...")
            df_feat = add_features(df_feat)
            if self.verbose:
                end_time = datetime.now()
                time_elapsed = end_time - start_time
                if i in range(9):
                    print(f"[{end_time.time()}]: ... " + list_added_feats[i] + " added.")
                else:
                    print(f"[{end_time.time()}]: ... " + list_added_feats[i] + " removed.")
                    
        if self.verbose:
            start_time = datetime.now()
            print(f"[{start_time.time()}]: Adding target...")
            
        df_feat = self._join_target(df_feat, y)
        
        if self.verbose:
            end_time = datetime.now()
            print(f"[{end_time.time()}]: ... target added.")
        
        return df_feat