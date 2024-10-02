#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 08 2023

@author: lpoulain
"""

import cdsapi
import numpy as np
import pandas as pd
import pickle
import argparse

folder_path = '/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/'
with open(folder_path+'valid_dates_1980_00_06_12_18.pkl','rb') as f:
    valid_dates = pickle.load(f)

parser = argparse.ArgumentParser(description='Download ERA5 data')

parser.add_argument('--min_year', type=int, default=1985,
                    help='Minimum year to download')
parser.add_argument('--min_month', type=int, default=12,
                    help='Minimum month to download')

min_year = parser.parse_args().min_year
min_month = parser.parse_args().min_month

print(f"Starting from {min_year}-{min_month}")

# load client interface
client = cdsapi.Client()


#Single Pressure
data_origin = 'reanalysis-era5-pressure-levels'

datavars = [ #'vorticity'
              'geopotential', 
              'relative_humidity',
              'temperature', 
              'u_component_of_wind', 
              'v_component_of_wind',
              'vertical_velocity',
              #'divergence', 
              'specific_humidity',
           ]
       
"""plevels = {'geopotential':[1000, 925, 850, 700, 600, 500, 
                           400, 300, 250, 200, 150, 100, 50
                           ],
           'relative_humidity':[1000, 925, 850, 700, 600, 500, 
                           400, 300, 250, 200, 150, 100, 50
                           ],
           'temperature':[1000, 925, 850, 700, 600, 500, 
                           400, 300, 250, 200, 150, 100, 50
                           ],
           'u_component_of_wind':[1000, 925, 850, 700, 600, 500, 
                           400, 300, 250, 200, 150, 100, 50
                           ],
           'v_component_of_wind':[1000, 925, 850, 700, 600, 500, 
                           400, 300, 250, 200, 150, 100, 50
                           ],
           'vertical_velocity':[1000, 925, 850, 700, 600, 500, 
                           400, 300, 250, 200, 150, 100, 50
                           ],
           'divergence':[1000, 925, 850, 700, 600, 500, 
                           400, 300, 250, 200, 150, 100, 50
                           ],
           'specific_humidity':[1000, 925, 850, 700, 600, 500, 
                           400, 300, 250, 200, 150, 100, 50
                           ]
          }"""
          
plevels = [1000, 925, 850, 700, 600, 500, 
            400, 300, 250, 200, 150, 100, 50
            ]

times = ['00:00', '06:00', 
         '12:00', '18:00',
        ]

for year in list(valid_dates.keys()):
    if int(year) >= min_year and int(year) <= min_year+3:
        for month in list(valid_dates[year].keys()):
            data_params = {
                    'product_type': 'reanalysis',
                    'format': 'grib',
                    'variable': datavars,
                    'pressure_level': plevels, 
                    'year': f'{year}',
                    'month': f'{month}',
                    'day': list(valid_dates[year][month]),
                    'time': times,
                }
            
            if not (int(year)==min_year and int(month)<=min_month): 
                target_path = f'{folder_path}ERA5_{year}_{month}_upper.grib'
                client.retrieve(name=data_origin, request=data_params, target=target_path)
                
print(f"Done.")