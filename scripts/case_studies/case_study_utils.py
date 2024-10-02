import numpy as np
import sys

from utils.main_utils import get_start_date_nc, get_lead_time
from utils.cut_region import haversine


def find_trajectory_point(data, lat_truth, lon_truth, centroid_size=4, fcnv2=False):
    
    if "latitude" not in data.coords:
        data = data.rename({"lat":"latitude", "lon":"longitude"})
    
    # compute the closest grid point to the True eye position
    lats, lons = data.latitude.values, data.longitude.values
    lats, lons = np.meshgrid(lats, lons)

    idxs = np.unravel_index(haversine(lat_truth, lon_truth, lats, lons).T[np.newaxis, :, :].argmin(), (lats.shape[0], lons.shape[0]))
    eye_lat, eye_lon = data.latitude.values[idxs[0]], data.longitude.values[idxs[1]]
    
    tmp_lat, tmp_lon = np.arange(eye_lat-0.25*(centroid_size//2), eye_lat+0.25*(centroid_size//2), 0.25),\
                        np.arange(eye_lon-0.25*(centroid_size//2), eye_lon+0.25*(centroid_size//2), 0.25)
    tmp_lat, tmp_lon = [l for l in tmp_lat if l in lats], [l for l in tmp_lon if l in lons]
    
    data_centroid = data.sel(latitude=tmp_lat, longitude=tmp_lon)
    
    if not fcnv2:
        min_pres = data_centroid.msl.values.min(axis=(0, 1))
        
        idxs = np.unravel_index(data_centroid.msl.values.argmin(), (data_centroid.latitude.shape[0], data_centroid.longitude.shape[0]))

        pred_lat, pred_lon = data_centroid.latitude.values[idxs[0]], data_centroid.longitude.values[idxs[1]]

        max_wind = np.sqrt(data_centroid.u10.values**2+data_centroid.v10.values**2).max(axis=(0, 1))
    else:
        fcn_var = '__xarray_dataarray_variable__'
        msl_idx = list(data_centroid.channel).index('msl')
        min_pres = data_centroid[fcn_var].values[0, msl_idx].min(axis=(0, 1))
        
        idxs = np.unravel_index(data_centroid[fcn_var].values[0, msl_idx].argmin(), (data_centroid.latitude.shape[0], data_centroid.longitude.shape[0]))
        pred_lat, pred_lon = data_centroid.latitude.values[idxs[0]], data_centroid.longitude.values[idxs[1]]
        
        u10_idx = list(data_centroid.channel).index('u10m')
        v10_idx = list(data_centroid.channel).index('v10m')
        
        max_wind = np.sqrt(data_centroid[fcn_var].values[0, u10_idx]**2+data_centroid[fcn_var].values[0, v10_idx]**2).max(axis=(0, 1))
        
    return max_wind, min_pres, pred_lat, pred_lon


def remove_duplicates(data_list):
    unique_start_dates = list(set([get_start_date_nc(p) for p in data_list]))
    
    new_data_list = []
    for start_date in unique_start_dates:
        tmp_list = [p for p in data_list if get_start_date_nc(p)==start_date]
        tmp_list = sorted(tmp_list, key=lambda x: get_lead_time(x))
        new_data_list.append(tmp_list[-1]) # keep only data with highest ldt
    key = lambda x: get_start_date_nc(x)
    new_data_list = sorted(new_data_list, key=key)
    return new_data_list


            
            
        
    