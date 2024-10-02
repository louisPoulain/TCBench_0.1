import xarray as xr

def select_localisation(ds: xr.Dataset, lat_min: float, lat_max: float, 
                        lon_min: float, lon_max: float) -> xr.Dataset:
    ds_new = ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)) # lat organised by decreasing order
    return ds_new



def min_MSLP(ds: xr.Dataset) -> xr.DataArray:
    return ds['msl'].min().values



def min_region_MLSP(ds, precision_percent=1) -> xr.DataArray:
    # return the region where MLSP is min up to precision_percent margin
    m = ds['msl'].min().values
    ds_new = ds.where(ds['msl'] <= (1+precision_percent/100) * m, drop=True)
    return ds_new