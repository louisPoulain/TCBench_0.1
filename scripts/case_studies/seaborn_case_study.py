import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import utils.main_utils as ut
import glob

## KATRINA 2005 - 2005236N23285


def resize(ds, lat_min, lat_max, lon_min, lon_max):
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    if lat_name=="lat":
        ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    else:
        ds = ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
    return ds


def get_max_wind(ds):
    return np.sqrt((ds.u10**2+ds.v10**2).values).max().max()

def get_min_pres(ds):
    return ds.msl.values.min().min()



def main(tc="Katrina"):
    
    assert tc in ["Katrina", "Kai-Tak"], "Only Katrina and Kai-Tak are available"
    sids = {"Katrina":"2001298S07098", "Kai-Tak": "2000185N15117"}
    months = {"Katrina":"08", "Kai-Tak":"07"}
    month = months[tc]
    sid = sids[tc]
    year = sid[:4]
    agencies = {"Katrina":"USA", "Kai-Tak":"TOKYO"}
    agency = agencies[tc]
    
    
    df = pd.read_csv("/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered"\
                    +"_1980_00_06_12_18.csv", dtype="string", na_filter=False)


    gc_path = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/graphcast/"
    pangu_path = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/panguweather/"
    fcn_path = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/fourcastnetv2/"
    df = df[df["SID"]==sid]

    lat, lon = df[["LAT", "LON"]].values[0]
    lon = float(lon)+360 if float(lon)<0 else float(lon)
    closest_lon = ut.closest_longitude(lat, lon, np.arange(0, 360, 0.25))
    closest_lat = ut.closest_lat(lat, closest_lon, np.arange(-90, 90, 0.25))

    key = lambda x: (ut.get_start_date_nc(x), ut.get_lead_time(x))
    list_of_pred_gc = sorted(glob.glob(gc_path+f"graphcast_{year}*ldt_6.nc"), key=key)
    list_of_pred_pangu = sorted(glob.glob(pangu_path+f"pangu_{year}*ldt_6.nc"), key=key)
    list_of_pred_fc = sorted(glob.glob(fcn_path+f"fourcastnetv2_{year}*ldt_6.nc"), key=key)
    era5 = xr.open_dataset(f"/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/ERA5_{year}_{month}_surface.grib")
    
    
    first_iso, last_iso = ut.get_start_date_nc(list_of_pred_gc[0])+np.timedelta64(6,'h'), \
                            ut.get_start_date_nc(list_of_pred_gc[-1])+np.timedelta64(6,'h')
                            
    era5_valid_times = era5.valid_time.values
    era5_idxs = [i for i in range(len(era5_valid_times)) if era5_valid_times[i] in [first_iso, last_iso]]
    era5 = era5.isel(step=slice(era5_idxs[0], era5_idxs[1]+1))


    ## GRAPHCAST
    print("Graphcast")
    wind_gc = []
    pres_gc = []
    for i, file in enumerate(list_of_pred_gc):
        ds_gc = xr.open_dataset(file).isel(time=0)
        ds_gc = resize(ds_gc, closest_lat-30, closest_lat+30, closest_lon-30, closest_lon+30)
        max_wind = get_max_wind(ds_gc)
        min_pres = get_min_pres(ds_gc)
        wind_gc.append(max_wind)
        pres_gc.append(min_pres)
    min_pres_gc, max_pres_gc = min(pres_gc), max(pres_gc)
    min_wind_gc, max_wind_gc = min(wind_gc), max(wind_gc)
    

    ## PANGU
    print("Pangu")
    wind_pg = []
    pres_pg = []
    
    for i, file in enumerate(list_of_pred_pangu):
        ds_pangu = xr.open_dataset(file).isel(time=0)
        ds_pangu["lat"] = -ds_pangu["lat"]
        ds_pangu = resize(ds_pangu, closest_lat+30, closest_lat-30, closest_lon-30, closest_lon+30)
        max_wind = get_max_wind(ds_pangu)
        min_pres = get_min_pres(ds_pangu)
        
        wind_pg.append(max_wind)
        pres_pg.append(min_pres)
    
    min_pres_pg, max_pres_pg = min(pres_pg), max(pres_pg)
    min_wind_pg, max_wind_pg = min(wind_pg), max(wind_pg)
    
    
    ## FOURCASTNET
    print("Fourcastnet")
    
    wind_fc = []
    pres_fc = []
    
    for i, file in enumerate(list_of_pred_fc):
        ds_fcn = xr.open_dataset(file).isel(time=1)
        ds_fcn = resize(ds_fcn, closest_lat+30, closest_lat-30, closest_lon-30, closest_lon+30)
        max_wind = np.sqrt(ds_fcn["__xarray_dataarray_variable__"][0, 0].values**2 +\
                            ds_fcn["__xarray_dataarray_variable__"][0, 1].values**2).max().max()
        min_pres = ds_fcn["__xarray_dataarray_variable__"][0, 6].values.min().min()
        
        wind_fc.append(max_wind)
        pres_fc.append(min_pres)
        
    min_pres_fc, max_pres_fc = min(pres_fc), max(pres_fc)
    min_wind_fc, max_wind_fc = min(wind_fc), max(wind_fc)
        
    ## ERA5
    print("ERA5")
    wind_era5 = []
    pres_era5 = []
    
    for i in range(era5.step.values.shape[0]):
        ds_era5 = era5.isel(step=i)
        ds_era5 = resize(ds_era5, closest_lat-30, closest_lat+30, closest_lon-30, closest_lon+30)
        max_wind = get_max_wind(ds_era5)
        min_pres = get_min_pres(ds_era5)
        
        wind_era5.append(max_wind)
        pres_era5.append(min_pres)
    
    min_pres_era5, max_pres_era5 = min(pres_era5), max(pres_era5)
    min_wind_era5, max_wind_era5 = min(wind_era5), max(wind_era5)
        
    ## TARGET 
    print("Target")
    times = era5.valid_time.values
    wind_target = []
    pres_target = []
    
    for t in times:
        wind = df[df["ISO_TIME"].astype("datetime64[ns]")==t]["USA_WIND"].values[0]
        pres = df[df["ISO_TIME"].astype("datetime64[ns]")==t][f"{agency}_PRES"].values[0]
        
        # wind is in knots, pressure in mb so we have to convert
        wind_target.append(float(wind) * 0.5144444444444444)
        if pres in ['', ' ']: # when not reported assume by default 1000mb
            pres = 1000
        pres_target.append(float(pres) * 100)
    
    min_pres_target, max_pres_target = min(pres_target), max(pres_target)
    min_wind_target, max_wind_target = min(wind_target), max(wind_target)
    
    wind_dic = {"gc":[min_wind_gc, max_wind_gc], "pg":[min_wind_pg, max_wind_pg], 'fc':[min_wind_fc, max_wind_fc],
                "era5":[min_wind_era5, max_wind_era5], "target":[min_wind_target, max_wind_target]}
    pres_dic = {"gc":[min_pres_gc, max_pres_gc], "pg":[min_pres_pg, max_pres_pg], 'fc':[min_pres_fc, max_pres_fc],
                "era5":[min_pres_era5, max_pres_era5], "target":[min_pres_target, max_pres_target]}

    df_wind = pd.DataFrame({'Max wnd speed (graphcast, m/s)': np.array(wind_gc),
                            'Max wnd speed (pangu, m/s)': np.array(wind_pg),
                            'Max wnd speed (FCNv2, m/s)': np.array(wind_fc),
                            'Max wnd speed (ERA5, m/s)': np.array(wind_era5),
                            'Max wnd speed (target, m/s)': np.array(wind_target),
                            })
    df_pres = pd.DataFrame({'Min pres (graphcast, Pa)': np.array(pres_gc),
                            'Min pres (pangu, Pa)': np.array(pres_pg),
                            'Min pres (FCNv2, Pa)': np.array(pres_fc),
                            'Min pres (ERA5, Pa)': np.array(pres_era5),
                            'Min pres (target, Pa)': np.array(pres_target),
                            })

    return df_wind, df_pres, wind_dic, pres_dic

if __name__ == "__main__":
    
    tc = "Katrina"
    years = {"Katrina":"2005", "Kai-Tak":"2000"}
    year = years[tc]
    print(tc, year)
    
    df_wind, df_pres, wind_dic, pres_dic = main(tc)
    
    plot_path = "/users/lpoulain/louis/plots/seaborn_case_study/"
    ## wind speed
    
    #lim = [min(wind_dic["era5"][0], wind_dic["target"][0]), max(wind_dic["era5"][1], wind_dic["target"][1])]
    g = sns.jointplot(data=df_wind, x="Max wnd speed (ERA5, m/s)", y="Max wnd speed (target, m/s)", 
                      kind='hex', gridsize=20, color='b', marginal_kws=dict(bins=20)) 
    g.fig.suptitle(f"{tc} {year} - Max wind speed (ERA5 vs target)")#bins=np.linspace(lim[0], lim[1], 40)
    #g.savefig(plot_path + f"wind_speed_era5_target_{tc}.png")
    
    #lim = [min(wind_dic["gc"][0], wind_dic["era5"][0]), max(wind_dic["gc"][1], wind_dic["era5"][1])]
    g = sns.jointplot(data=df_wind, x="Max wnd speed (graphcast, m/s)", y="Max wnd speed (ERA5, m/s)", 
                      kind='hex', gridsize=20, color='b', marginal_kws=dict(bins=20))
    g.fig.suptitle(f"{tc} {year} - Max wind speed (graphcast vs ERA5, 6h lead time)")
    #g.savefig(plot_path + f"wind_speed_gc_era5_{tc}.png")
    
    #lim = [min(wind_dic["gc"][0], wind_dic["target"][0]), max(wind_dic["gc"][1], wind_dic["target"][1])]
    g = sns.jointplot(data=df_wind, x="Max wnd speed (graphcast, m/s)", y="Max wnd speed (target, m/s)", 
                      kind='hex', gridsize=20, color='b', marginal_kws=dict(bins=20))
    g.fig.suptitle(f"{tc} {year} - Max wind speed (graphcast vs target, 6h lead time)")
    #g.savefig(plot_path + f"wind_speed_gc_target_{tc}.png")
    
    #lim = [min(wind_dic["pg"][0], wind_dic["target"][0]), max(wind_dic["pg"][1], wind_dic["target"][1])]
    g = sns.jointplot(data=df_wind, x="Max wnd speed (pangu, m/s)", y="Max wnd speed (target, m/s)",
                        kind='hex', gridsize=20, color='b', marginal_kws=dict(bins=20))
    g.fig.suptitle(f"{tc} {year} - Max wind speed (pangu vs target, 6h lead time)")
    #g.savefig(plot_path + f"wind_speed_pangu_target_{tc}.png")
    
    #lim = [min(wind_dic["pg"][0], wind_dic["era5"][0]), max(wind_dic["pg"][1], wind_dic["era5"][1])]
    g = sns.jointplot(data=df_wind, x="Max wnd speed (pangu, m/s)", y="Max wnd speed (ERA5, m/s)",
                        kind='hex', gridsize=20, color='b', marginal_kws=dict(bins=20))
    g.fig.suptitle(f"{tc} {year} - Max wind speed (pangu vs ERA5, 6h lead time)")
    #g.savefig(plot_path + f"wind_speed_pangu_era5_{tc}.png")
    
    #lim = [min(wind_dic["fc"][0], wind_dic["target"][0]), max(wind_dic["fc"][1], wind_dic["target"][1])]
    g = sns.jointplot(data=df_wind, x="Max wnd speed (FCNv2, m/s)", y="Max wnd speed (target, m/s)",
                        kind='hex', gridsize=20, color='b', marginal_kws=dict(bins=20))
    g.fig.suptitle(f"{tc} {year} - Max wind speed (FCNv2 vs target, 6h lead time)")
    #g.savefig(plot_path + f"wind_speed_fcn_target_{tc}.png")
    
    #lim = [min(wind_dic["fc"][0], wind_dic["era5"][0]), max(wind_dic["fc"][1], wind_dic["era5"][1])]
    g = sns.jointplot(data=df_wind, x="Max wnd speed (FCNv2, m/s)", y="Max wnd speed (ERA5, m/s)",
                        kind='hex', gridsize=20, color='b', marginal_kws=dict(bins=20))
    g.fig.suptitle(f"{tc} {year} - Max wind speed (FCNv2 vs ERA5, 6h lead time)")
    #g.savefig(plot_path + f"wind_speed_fcn_era5_{tc}.png")
    
    
    ## pressure
    #lim = [min(pres_dic["era5"][0], pres_dic["target"][0]), max(pres_dic["era5"][1], pres_dic["target"][1])]
    g = sns.jointplot(data=df_pres, x="Min pres (ERA5, Pa)", y="Min pres (target, Pa)", 
                      kind='hex', gridsize=20, color='b', marginal_kws=dict(bins=20))
    g.fig.suptitle(f"{tc} {year} - Min pressure (ERA5 vs target)")
    #g.savefig(plot_path + f"pressure_era5_target_{tc}.png")
    
    #lim = [min(pres_dic["gc"][0], pres_dic["era5"][0]), max(pres_dic["gc"][1], pres_dic["era5"][1])]
    g = sns.jointplot(data=df_pres, x="Min pres (graphcast, Pa)", y="Min pres (ERA5, Pa)",
                        kind='hex', gridsize=20, color='b', marginal_kws=dict(bins=20))
    g.fig.suptitle(f"{tc} {year} - Min pressure (graphcast vs ERA5, 6h lead time)")
    #g.savefig(plot_path + f"pressure_gc_era5_{tc}.png")
    
    #lim = [min(pres_dic["gc"][0], pres_dic["target"][0]), max(pres_dic["gc"][1], pres_dic["target"][1])]
    g = sns.jointplot(data=df_pres, x="Min pres (graphcast, Pa)", y="Min pres (target, Pa)",
                        kind='hex', gridsize=20, color='b', marginal_kws=dict(bins=20))
    g.fig.suptitle(f"{tc} {year} - Min pressure (graphcast vs target, 6h lead time)")
    #g.savefig(plot_path + f"pressure_gc_target_{tc}.png")
    
    #lim = [min(pres_dic["pg"][0], pres_dic["target"][0]), max(pres_dic["pg"][1], pres_dic["target"][1])]
    g = sns.jointplot(data=df_pres, x="Min pres (pangu, Pa)", y="Min pres (target, Pa)",
                        kind='hex', gridsize=20, color='b', marginal_kws=dict(bins=20))
    g.fig.suptitle(f"{tc} {year} - Min pressure (pangu vs target, 6h lead time)")
    #g.savefig(plot_path + f"pressure_pangu_target_{tc}.png")
    
    #lim = [min(pres_dic["pg"][0], pres_dic["era5"][0]), max(pres_dic["pg"][1], pres_dic["era5"][1])]
    g = sns.jointplot(data=df_pres, x="Min pres (pangu, Pa)", y="Min pres (ERA5, Pa)",
                        kind='hex', gridsize=20, color='b', marginal_kws=dict(bins=20))
    g.fig.suptitle(f"{tc} {year} - Min pressure (pangu vs ERA5, 6h lead time)")
    #g.savefig(plot_path + f"pressure_pangu_era5_{tc}.png")
    
    #lim = [min(pres_dic["fc"][0], pres_dic["target"][0]), max(pres_dic["fc"][1], pres_dic["target"][1])]
    g = sns.jointplot(data=df_pres, x="Min pres (FCNv2, Pa)", y="Min pres (target, Pa)",
                        kind='hex', gridsize=20, color='b', marginal_kws=dict(bins=20))
    g.fig.suptitle(f"{tc} {year} - Min pressure (FCNv2 vs target, 6h lead time)")
    #g.savefig(plot_path + f"pressure_fcn_target_{tc}.png")
    
    #lim = [min(pres_dic["fc"][0], pres_dic["era5"][0]), max(pres_dic["fc"][1], pres_dic["era5"][1])]
    g = sns.jointplot(data=df_pres, x="Min pres (FCNv2, Pa)", y="Min pres (ERA5, Pa)",
                        kind='hex', gridsize=20, color='b', marginal_kws=dict(bins=20))
    g.fig.suptitle(f"{tc} {year} - Min pressure (FCNv2 vs ERA5, 6h lead time)")
    #g.savefig(plot_path + f"pressure_fcn_era5_{tc}.png")
    
    
