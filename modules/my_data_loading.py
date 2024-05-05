import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

class DataLoading:
    """
    Data Loading Module
    ===================
    
    This module loads the target, client, gas_prices, electricity_prices, 
    forecast_weather, historical_weather, weather_station_to_county_mapping
    and school_holidays from either .csv files into the pandas dataframe or 
    json format. The dates in the .csv files are parsed into the pandas 
    datetime format. For the data documentation, see
    www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/data.
    The school holidays in Estonian were taken from
    www.holidays-info.com/estonia/school-holidays/.
    --------------------------------------------------------------------------
    parameters:
    
    
    root_path (str): 
        The path for the folder with the data. 
    
    --------------------------------------------------------------------------
    atributes:
    
    
    df_data: 
        Dataframe corresponding to the full train dataset.
    
    df_client: 
        Dataframe corresponding to the client dataset containing the
        columns product_type, county, eic_count, installed_capacity,
        is_business, date.
    
    df_gas_prices: 
        Dataframe corresponding to the gas_prices dataset containing
        the columns forecast_date, lowest_price_per_mwh,
        highest_price_per_mwh.
    
    df_electricity_prices: 
        Dataframe corresponding to the electricty_prices dataset
        containing the columns forecast_date, euros_per_mwh.
    
    df_forecast_weather: 
        Dataframe corresponding to the forecast_weather dataset
        containing the columns latitude, longitude, origin_datetime,
        hours_ahead, temperature, dewpoint, cloudcover_high,
        cloudcover_low, cloudcover_mid, cloudcover_total,
        10_metre_u_wind_component, 10_metre_v_wind_component,
        forecast_datetime, direct_solar_radiation, 
        surface_solar_radiation_downwards, snowfall,
        total_precipitation.
    
    df_historical_weather: 
        Dataframe corresponding to the historical_weather dataset 
        containing the columns datetime, temperature, dewpoint,
        rain, snowfall, surface_pressure, cloudcover_total,
        cloudcover_low, cloudcover_mid, cloudcover_high,
        windspeed_10m, winddirection_10m, shortwave_radiation,
        direct_solar_radiation, diffuse_radiation, latitude,
        longitude.
 
    df_target: 
        Dataframe corresponding to the target dataset containing the
        columns target, county, is_business, product_type, 
        is_consumption, datetime.

    
    df_weather_station_to_county_mapping: 
        Dataframe corresponding to the df_weather_station_to_county_mapping
        dataset.
        
    --------------------------------------------------------------------------
    methods:
    
    
    update_with_new_data([df_new_client, df_new_gas_prices, 
                          df_new_electricity_prices, df_new_forecast_weather,
                          df_new_historical_weather, df_new_target]):
        Returns the new dataframes concatenated along the row axis with the
        corresponding loaded dataframe.
        
    preprocess_test(df_test):
        Returns the df_test dataframe with the column datetime renames as
        prediction_datetime.
    
    """

    # set the relevant columns for each dataset
    # as class member variables
    data_cols = [
        "target",
        "county",
        "is_business",
        "product_type",
        "is_consumption",
        "datetime",
        "row_id",
        "prediction_unit_id",
    ]
    client_cols = [
        "product_type",
        "county",
        "eic_count",
        "installed_capacity",
        "is_business",
        "date",
    ]
    gas_prices_cols = [
        "forecast_date",
        "lowest_price_per_mwh",
        "highest_price_per_mwh"
        ]
    electricity_prices_cols = [
        "forecast_date",
        "euros_per_mwh"
        ]
    forecast_weather_cols = [
        "latitude",
        "longitude",
#        "origin_datetime",
#        "hours_ahead",
        "forecast_datetime",
        "temperature",
        "dewpoint",
        "cloudcover_high",
        "cloudcover_low",
        "cloudcover_mid",
        "cloudcover_total",
        "10_metre_u_wind_component",
        "10_metre_v_wind_component",
        "direct_solar_radiation",
        "surface_solar_radiation_downwards",
        "snowfall",
        "total_precipitation"
    ]
    historical_weather_cols = [
        "datetime",
        "temperature",
        "dewpoint",
        "rain",
        "snowfall",
        "surface_pressure",
        "cloudcover_total",
        "cloudcover_low",
        "cloudcover_mid",
        "cloudcover_high",
        "windspeed_10m",
        "winddirection_10m",
        "shortwave_radiation",
        "direct_solar_radiation",
        "diffuse_radiation",
        "latitude",
        "longitude",
    ]
    location_cols = [
        "longitude", 
        "latitude", 
        "county"
    ]
    target_cols = [
        "target",
        "county",
        "is_business",
        "product_type",
        "is_consumption",
        "datetime",
    ]

    def __init__(self, root_path, verbose = True):
        self.root_path = root_path
        self.verbose = verbose
        
        # the data is loaded into a dataframe
        # from the repo in repo_path using the
        # columns given by the class member 
        # variables above
        if self.verbose:
            start_time = datetime.now()
            print(f"[{start_time.time()}]: Loading train.csv file...")
        self.df_data = pd.read_csv(
            os.path.join(self.root_path, "train.csv"), 
            usecols=self.data_cols,
            parse_dates=["datetime"],
        )
        if self.verbose:
            end_time = datetime.now()
            print(f"[{end_time.time()}]: ... train.csv file loaded.")        
     
    
        if self.verbose:
            start_time = datetime.now()
            print(f"[{start_time.time()}]: Loading client.csv file...")        
        self.df_client = pd.read_csv(
            os.path.join(self.root_path, "client.csv"),
            usecols=self.client_cols,
            parse_dates=["date"],
        )
        if self.verbose:
            end_time = datetime.now()
            print(f"[{end_time.time()}]: ... client.csv file loaded.")        
        

        if self.verbose:
            start_time = datetime.now()
            print(f"[{start_time.time()}]: Loading gas_prices.csv file...")
        self.df_gas_prices = pd.read_csv(
            os.path.join(self.root_path, "gas_prices.csv"),
            usecols=self.gas_prices_cols,
            parse_dates=["forecast_date"],
        )
        if self.verbose:
            end_time = datetime.now()
            print(f"[{end_time.time()}]: ... gas_prices.csv file loaded.")        

        
        if self.verbose:
            start_time = datetime.now()
            print(f"[{start_time.time()}]: Loading electricity_prices.csv file...")        
        self.df_electricity_prices = pd.read_csv(
            os.path.join(self.root_path, "electricity_prices.csv"),
            usecols=self.electricity_prices_cols,
            parse_dates=["forecast_date"],
        )
        if self.verbose:
            end_time = datetime.now()
            print(f"[{end_time.time()}]: ... electricity_prices.csv file loaded.")        

            
        if self.verbose:
            start_time = datetime.now()
            print(f"[{start_time.time()}]: Loading forecast_weather.csv file...")    
        self.df_forecast_weather = pd.read_csv(
            os.path.join(self.root_path, "forecast_weather.csv"),
            usecols=self.forecast_weather_cols,
            parse_dates=["forecast_datetime"],
        )
        if self.verbose:
            end_time = datetime.now()
            print(f"[{end_time.time()}]: ... forecast_weather.csv file loaded.")        
        

        if self.verbose:
            start_time = datetime.now()
            print(f"[{start_time.time()}]: Loading historical_weather.csv file...")            
        self.df_historical_weather = pd.read_csv(
            os.path.join(self.root_path, "historical_weather.csv"),
            usecols=self.historical_weather_cols,
            parse_dates=["datetime"],
        )
        if self.verbose:
            end_time = datetime.now()
            print(f"[{end_time.time()}]: ... historical_weather.csv file loaded.")        

            
        if self.verbose:
            start_time = datetime.now()
            print(f"[{start_time.time()}]: Loading weather_station_to_county_mapping.csv file...")    
        self.df_weather_station_to_county_mapping = pd.read_csv(
            os.path.join(self.root_path, "weather_station_to_county_mapping.csv"),
            usecols=self.location_cols
        )
        if self.verbose:
            end_time = datetime.now()
            print(f"[{end_time.time()}]: ... weather_station_to_county_mapping.csv file loaded.")        


        self.df_target = self.df_data[self.target_cols]

        #load the dataset corresponding to the map between the 
        #geocoordinates of the weather stations and the county 
        #they are located at

        if self.verbose:
            start_time = datetime.now()
            print(f"[{start_time.time()}]: Loading weather_station_to_county_dictionary.json file...")    
        FILE_PATH = os.path.join(self.root_path, "weather_station_to_county_dictionary.json")
        with open(FILE_PATH) as json_data:
            self.station_to_county = json.load(json_data)
        if self.verbose:
            end_time = datetime.now()
            print(f"[{end_time.time()}]: ... weather_station_to_county_dictionary.json file loaded.")        


        if self.verbose:
            start_time = datetime.now()
            print(f"[{start_time.time()}]: Building the station-to-county dictionary...")    
        self.station_to_county = {
            (round(lat,1),round(lon,1)) : county for lat, lon, county in zip(
                self.station_to_county.get('latitude'), 
                self.station_to_county.get('longitude'), 
                self.station_to_county.get('county')
            )
        }
        if self.verbose:
            end_time = datetime.now()
            print(f"[{end_time.time()}]: ... station-to-county dictionary built.")        

        
        #load the dataset corresponding to the geocoordinates of
        #each estonian county

        if self.verbose:
            start_time = datetime.now()
            print(f"[{start_time.time()}]: Loading county_location.json file...")    
        FILE_PATH = os.path.join(self.root_path, "county_location.json")
        with open(FILE_PATH) as json_data:
            self.county_loc = json.load(json_data)
        self.county_loc = {int(k):v for k,v in self.county_loc.items()}
        if self.verbose:
            end_time = datetime.now()
            print(f"[{end_time.time()}]: ... county_location.json file loaded.")        

        
        #load the dataset corresponding to the school holiday dates
        #in Estonia

        if self.verbose:
            start_time = datetime.now()
            print(f"[{start_time.time()}]: Loading school_holidays.json file...")    
        FILE_PATH = os.path.join(self.root_path, "school_holidays.json")
        with open(FILE_PATH) as json_data:
            self.school_holidays = json.load(json_data)
        if self.verbose:
            end_time = datetime.now()
            print(f"[{end_time.time()}]: ... school_holidays.json file loaded.")        

        
    def update_with_new_data(
        self,
        df_new_client,
        df_new_gas_prices,
        df_new_electricity_prices,
        df_new_forecast_weather,
        df_new_historical_weather,
        df_new_target,
    ):
        
        #concatenate the new datasets with the already
        #loaded dataframes along the row axis and remove
        #duplicated rows
        self.df_client = pd.concat(
            [self.df_client, df_new_client]
        ).drop_duplicates(
            ["date", "county", "is_business", "product_type"]
        )
        
        self.df_gas_prices = pd.concat(
            [self.df_gas_prices, df_new_gas_prices]
        ).drop_duplicates(
            ["forecast_date"]
        )
        
        self.df_electricity_prices = pd.concat(
            [self.df_electricity_prices, df_new_electricity_prices]
        ).drop_duplicates(
            ["forecast_date"]
        )
        
        self.df_forecast_weather = pd.concat(
            [self.df_forecast_weather, df_new_forecast_weather]
        ).drop_duplicates(
            ["forecast_datetime", "latitude", "longitude", "hours_ahead"]
        )
        
        self.df_historical_weather = pd.concat(
            [self.df_historical_weather, df_new_historical_weather]
        ).drop_duplicates(
            ["datetime", "latitude", "longitude"]
        )
        
        self.df_target = pd.concat(
            [self.df_target, df_new_target]
        ).drop_duplicates(
            ["datetime", "county", "is_business", "product_type", "is_consumption"]
        )

    def preprocess_test(self, df_test):
        #rename the column datetime as prediction_datetime
        df_test = df_test.rename(columns={"prediction_datetime": "datetime"})
        
        return df_test
