# KATRINA 2005 - 2005236N23285
import utils.utils as ut
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import glob
import cartopy.crs as ccrs

df = pd.read_csv("/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv", 
                 dtype="string", na_filter=False)


gc_path = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/graphcast/"
pangu_path = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/panguweather/"
fcn_path = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/fourcastnetv2/"
df = df[df["SID"]=="2005236N23285"]


key = lambda x: (ut.get_start_date_nc(x), ut.get_lead_time(x))
list_of_pred_gc = sorted(glob.glob(gc_path+f"graphcast_2005*ldt_6.nc"), key=key)
list_of_pred_pangu = sorted(glob.glob(pangu_path+f"pangu_2005*ldt_6.nc"), key=key)
list_of_pred_fc = sorted(glob.glob(fcn_path+f"fourcastnetv2_2005*ldt_6.nc"), key=key)

## GRAPHCAST
max_gc = 0
min_gc = 100

data_list_gc = []
times = []
for i, file in enumerate(list_of_pred_gc):
    ds_gc = xr.open_dataset(file).isel(time=0)
    t = ut.get_start_date_nc(file)+ds_gc.time.values
    wind_gc = (ds_gc.u10**2+ds_gc.v10**2)**(1/2)
    max_gc = max(wind_gc.values.max().max(), max_gc)
    min_gc = min(wind_gc.values.min().min(), min_gc)
    if i==0:
        lat, lon = df[df["ISO_TIME"].astype("datetime64[ns]")==t][["LAT", "LON"]].values[0]
    data_list_gc.append(wind_gc)
    times.append(t)


## PANGU
max_pg = 0
min_pg = 100

data_list_pangu = []    
for i, file in enumerate(list_of_pred_pangu):
    ds_pangu = xr.open_dataset(file).isel(time=0)
    #ds_pangu["lat"] = -ds_pangu["lat"]
    wind_pangu = (ds_pangu.u10**2+ds_pangu.v10**2)**(1/2)
    max_pg = max(wind_pangu.values.max().max(), max_pg)
    min_pg = min(wind_pangu.values.min().min(), min_pg)
    data_list_pangu.append(wind_pangu)
    

## FOURCASTNET
max_fc = 0
min_fc = 100

data_list_fc = []
for i, file in enumerate(list_of_pred_fc):
    ds_fc = xr.open_dataset(file).isel(time=1)
    wind_fc = (ds_fc["__xarray_dataarray_variable__"][0, 0]**2+ds_fc["__xarray_dataarray_variable__"][0, 1]**2)**(1/2)
    max_fc = max(wind_fc.values.max().max(), max_fc)
    min_fc = min(wind_fc.values.min().min(), min_fc)
    data_list_fc.append(wind_fc)


vmin, vmax = min(min_gc, min_pg, min_fc), max(max_gc, max_pg, max_fc)

fig, axs = plt.subplots(1, 3, figsize=(15, 7), subplot_kw={'projection': ccrs.Orthographic(lon, lat)})
ax1, ax2, ax3 = axs
def update(i):
    if i%10==0:
        print(i)
    ax1.clear()
    im = data_list_gc[i].plot.imshow(cmap='viridis', 
                      transform=ccrs.PlateCarree(),
                      ax=ax1,
                      add_colorbar=False,
                      vmin=vmin,
                      vmax=vmax)
    #ax.set_global()
    ax1.coastlines()
    ax1.set_title("GraphCast")
    
    ax2.clear()
    im = data_list_pangu[i].plot.imshow(cmap='viridis', #.reindex({"lat":data_list_pangu[i].lat.values[::-1]})
                      transform=ccrs.PlateCarree(),
                      ax=ax2,
                      add_colorbar=False,
                      vmin=vmin,
                      vmax=vmax)
    #ax.set_global()
    ax2.coastlines()
    ax2.set_title("PanguWeather")
    
    ax3.clear()
    im = data_list_fc[i].plot.imshow(cmap='viridis', #.reindex({"lat":data_list_pangu[i].lat.values[::-1]})
                      transform=ccrs.PlateCarree(),
                      ax=ax3,
                      add_colorbar=False,
                      vmin=vmin,
                      vmax=vmax)
    
    ax3.coastlines()
    ax3.set_title("FourcastNet")
    
    
    fig.subplots_adjust(bottom=0.005, top=0.95, left=0.05, right=0.8)
    if i==0:
        cbar_ax = fig.add_axes([0.85, 0.25, 0.05, 0.45])
        fig.colorbar(im, ax=ax1, orientation='vertical', fraction=.1, cax=cbar_ax)
    
    st=fig.suptitle(f'{np.datetime64(times[i], "h")} - Forecast lead time: 6h')
    st.set_y(0.98)

#fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': ccrs.Orthographic(lon, lat)})
#def update_pgw(i):    
#    ax2.clear()
#    im = data_list_pangu[i].plot.imshow(cmap='viridis', #.reindex({"lat":data_list_pangu[i].lat.values[::-1]})
#                      transform=ccrs.PlateCarree(),
#                      ax=ax2,
#                      add_colorbar=False,
#                      vmin=vmin,
#                      vmax=vmax)
#    #ax.set_global()
#    ax2.coastlines()
#    ax2.set_title("PanguWeather")
#    if i==0:
#        cbar_ax = fig2.add_axes([0.85, 0.15, 0.05, 0.7])
#        fig1.colorbar(im, ax=ax2, orientation='vertical', fraction=.1, cax=cbar_ax)
#    
#    st=fig2.suptitle(f'{np.datetime64(times[i], "h")} - Forecast lead time: 6h')
#    st.set_y(0.98)


#fig2.subplots_adjust(bottom=0.005, top=0.95, left=0.05, right=0.8)

full_size = len(data_list_gc)
ani1 = FuncAnimation(fig, update, full_size, blit=False, interval=1000, repeat=False)
#ani2 = FuncAnimation(fig2, update_pgw, full_size, blit=False, interval=1000, repeat=False)


#writervideo = animation.FFMpegWriter(fps=12)
ani1.save('/users/lpoulain/louis/plots/test_wind_katrina_gpc_pgw_fcn.mp4', codec="mpeg4")
#ani2.save('/users/lpoulain/louis/plots/test_wind_katrina_pangu.mp4', codec="mpeg4")

plt.show()