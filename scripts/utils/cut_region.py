import glob, os, subprocess
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import xarray as xr

from main_utils import get_start_date_nc, get_end_date_nc, get_lead_time, get_tc_id_nc


def haversine(latp, lonp, lat_list, lon_list, **kwargs):
    """──────────────────────────────────────────────────────────────────────────┐
      Haversine formula for calculating distance between target point (latp,
      lonp) and series of points (lat_list, lon_list). This function can handle
      2D lat/lon lists, but has been used with flattened data

      Based on:
      https://medium.com/@petehouston/calculate-distance-of-two-locations-on-earth-using-python-1501b1944d97


      Inputs:
          latp - latitude of target point

          lonp - longitude of target point

          lat_list - list of latitudess (lat_p-1, lat_p-2 ... lon_p-n)

          lon_list - list of longitudess (lon_p-1, lon_p-2 ... lon_p-n)

      Outputs:

    └──────────────────────────────────────────────────────────────────────────"""
    kwargs.get("epsilon", 1e-6)

    latp = np.radians(latp)
    lonp = np.radians(lonp)
    lat_list = np.radians(lat_list)
    lon_list = np.radians(lon_list)

    dlon = lonp - lon_list
    dlat = latp - lat_list
    a = np.power(np.sin(dlat / 2), 2) + np.cos(lat_list) * np.cos(latp) * np.power(
        np.sin(dlon / 2), 2
    )

    # Assert that sqrt(a) is within machine precision of 1
    # assert np.all(np.sqrt(a) <= 1 + epsilon), 'Invalid argument for arcsin'

    # Check if square root of a is a valid argument for arcsin within machine precision
    # If not, set to 1 or -1 depending on sign of a
    a = np.where(np.sqrt(a) <= 1, a, np.sign(a))

    return 2 * 6371 * np.arcsin(np.sqrt(a))



def get_rectmask(point, grid, **kwargs):
        # read in parameters if submitted, otherwise use defaults
        # https://stackoverflow.com/a/50140132 for an explanation of unravel_index
        grid = np.meshgrid(*grid)
        distance_calculator = kwargs.get("distance_calculator", haversine)
        circum_points = kwargs.get("circum_points", 4)

        distances = distance_calculator(
            point[0],
            point[1],
            grid[0],
            grid[1],
        ).T[np.newaxis, :, :]

        min_idx = np.unravel_index(distances.argmin(), distances.shape)

        output = np.zeros_like(distances)
        output[
            min_idx[0],
            min_idx[1] - circum_points : min_idx[1] + circum_points + 1,
            min_idx[2] - circum_points : min_idx[2] + circum_points + 1,
        ] = 1
        print(output)
        return output


def closest_longitude(lat, lon, lon_list, **kwargs):

    lat, lon, lon_list = float(lat), float(lon), np.array(lon_list).astype(float)
    distance_calculator = kwargs.get("distance_calculator", haversine)
    distances = distance_calculator(lat, lon, [lat], lon_list).T
    
    min_idx = np.unravel_index(distances.argmin(), distances.shape)

    return lon_list[min_idx].item()


def closest_latitude(lat, lon, lat_list, **kwargs):

    lat, lon, lat_list = float(lat), float(lon), np.array(lat_list).astype(float)
    distance_calculator = kwargs.get("distance_calculator", haversine)
    distances = distance_calculator(lat, lon, lat_list, [lon]).T
    
    min_idx = np.unravel_index(distances.argmin(), distances.shape)

    return lat_list[min_idx].item()
    

def cut_rectangle(ds: xr.Dataset, df_tracks: pd.DataFrame, tc_id, date_start, tropics=False) -> xr.Dataset:
    # ds: dataset to cut
    # df_tracks: dataframe containing the tracks
    # tc_id: id of the TC to cut

    tc_track = df_tracks[(df_tracks["SID"]==tc_id)]# & (df_tracks["ISO_TIME"]==iso_time)
    tc_track = tc_track[tc_track["ISO_TIME"].astype("datetime64[ns]")>=date_start]
    start_lat, start_lon = tc_track["LAT"].values[0], float(tc_track["LON"].values[0])
    if float(start_lon) < 0:
        start_lon = float(start_lon)+360
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    
    closest_lon = closest_longitude(start_lat, start_lon, ds[lon_name].values)
    if not tropics:
        closest_lat = closest_latitude(start_lat, closest_lon, ds[lat_name].values)
        if ds[lat_name].values[0]>ds[lat_name].values[-1]:
            lats = np.arange(closest_lat+30, closest_lat-30.25, -0.25)
        else:
            lats = np.arange(closest_lat-30, closest_lat+30.25, 0.25)
    else:
        if ds[lat_name].values[0]>ds[lat_name].values[-1]:
            lats = np.arange(30, -30.25, -0.25)
        else:
            lats = np.arange(-30, 30.25, 0.25)
        
    if lon_name=="lon":
        if closest_lon-30>0 and closest_lon+30<360:
            ds_new = ds.sel(lat=lats, lon=slice(closest_lon-30, closest_lon+30))
        elif closest_lon-30<0:
            ds_new = ds.sel(lat=lats, lon=((ds["lon"]>=0) & (ds["lon"]<=closest_lon+30)) | \
                                                    ((ds["lon"]>=360+closest_lon-30) & (ds["lon"]<=360)))
        else:
            ds_new = ds.sel(lat=lats, lon=((ds["lon"]>=closest_lon-30) & (ds["lon"]<=360)) | \
                                                    ((ds["lon"]>=0) & (ds["lon"]<=closest_lon+30-360)))
    else:
        if closest_lon-30>0 and closest_lon+30<360:
            ds_new = ds.sel(latitude=lats, longitude=slice(closest_lon-30, closest_lon+30))
        elif closest_lon-30<0:
            ds_new = ds.sel(latitude=lats, longitude=((ds["longitude"]>=0) & (ds["longitude"]<=closest_lon+30)) | \
                                                    ((ds["longitude"]>=360+closest_lon-30) & (ds["longitude"]<=360)))
        else:
            ds_new = ds.sel(latitude=lats, longitude=((ds["longitude"]>=closest_lon-30) & (ds["longitude"]<=360)) | \
                                                    ((ds["longitude"]>=0) & (ds["longitude"]<=closest_lon+30-360)))
    
    return ds_new


def cut_and_save_rect(ds_folder, models, df_tracks:pd.DataFrame, date_start, date_end, lead_time, tc_id, output_path, 
                      l=None, idx=None, tropics=False):

    assert set(models).issubset(["pangu", "graphcast", "fourcastnetv2"]), f"models must be in ['pangu', 'graphcast', 'fourcastnetv2']"
    
    folder_names = {"pangu":"panguweather", "graphcast":"graphcast", "fourcastnetv2":"fourcastnetv2"}
    
    for i, model in enumerate(models):
        save_name = output_path + f"{folder_names[model]}/{model}_{date_start}_to_{date_end}_ldt_{lead_time}_{tc_id}_small.nc"
        if not os.path.isfile(save_name):
            msg = f"{date_start} to {date_end} ({lead_time}h)" + f" - {idx+1}/{l}" if (l is not None and idx is not None) else \
                    f"{date_start} to {date_end} ({lead_time}h)"
            print(msg, flush=True)
            path = ds_folder + f"{folder_names[model]}/{model}_{date_start}_to_{date_end}_ldt_{lead_time}.nc"
            #ds = xr.load_dataset(ds_folder + f"{folder_names[model]}/{model}_{date_start}_to_{date_end}_ldt_{lead_time}_{tc_id}.nc", 
            #                    engine="netcdf4")
            ds = xr.load_dataset(path, engine="netcdf4")
            try:
                ds_new = cut_rectangle(ds, df_tracks, tc_id, date_start, tropics=tropics)
                # compress data
                encoding = {}
                for data_var in ds_new.data_vars:
                    encoding[data_var] = {
                    "original_shape": ds_new[data_var].shape,
                    "_FillValue": -32767,
                    "dtype": np.int16,
                    "add_offset": ds_new[data_var].mean().compute().values,
                    "scale_factor": ds_new[data_var].std().compute().values / 1000, # save up to mean +- 32 std
                    # "zlib": True,
                    # "complevel": 5,
                    }
                ds_new.to_netcdf(save_name, engine="netcdf4", mode="w", encoding=encoding, compute=True)
                
            except KeyError:
                print(f"KeyError for {date_start} to {date_end} ({lead_time}h)", flush=True)
                continue
            del ds
    return path

            
def cut_save_in_series(ds_folder, models, year, output_path, parallel=False, remove=False, remove_waiting=15,
                       df_tracks_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv"):
    
    assert set(models).issubset(["pangu", "graphcast", "fourcastnetv2"]), "models must be in ['pangu', 'graphcast', 'fourcastnetv2']"
    
    folder_names = {"pangu":"panguweather", "graphcast":"graphcast", "fourcastnetv2":"fourcastnetv2"}
    
    key = lambda x: (get_start_date_nc(x), get_lead_time(x))
    
    params = {model : [(get_start_date_nc(file), get_end_date_nc(file), get_lead_time(file), get_tc_id_nc(file), file) for file in \
        sorted(list(set([f for f in glob.glob(ds_folder+f"{folder_names[model]}/{model}_{year}*_ldt_*_*.nc") if \
            len(os.path.basename(f).split("_"))==7])\
                    -set(glob.glob(ds_folder+f"{folder_names[model]}/{model}_{year}_*small.nc"))), key=key)] for model in models}
    
    df_tracks = pd.read_csv(df_tracks_path, dtype="string", na_filter=False)
    for model in models:
        l = len(params[model])
        done = []
        if not parallel:
            for i in range(len(params[model])):
                date_start, date_end, ldt, tc_id, file = params[model][i]
                path = cut_and_save_rect(ds_folder, [model], df_tracks, date_start, date_end, ldt, tc_id, output_path, l=l, idx=i)
                done.append(file)
                if len(done) == 50:
                    if remove:
                        print(f"!!! Removing 50 old files !!! {remove_waiting}s to STOP if this is a mistake", flush=True)
                        import time
                        time.sleep(remove_waiting)
                        subprocess.run(["rm", "-r", *done])
                        done = []
                if i == len(params[model])-1:
                    if remove:
                        print(f"!!! Removing {len(done)} old files !!! {remove_waiting}s to STOP if this is a mistake", flush=True)
                        import time
                        time.sleep(remove_waiting)
                        subprocess.run(["rm", "-r", *done])
                        done = []
        else:
            nb_cpus = cpu_count()
            with Pool(nb_cpus//2) as p:
                print(f"Using {nb_cpus//2} cpus")
                p.starmap(cut_and_save_rect, [(ds_folder, [model], df_tracks, date_start, date_end, ldt,\
                    tc_id, output_path, l, i) for i, (date_start, date_end, ldt, tc_id, file) in enumerate(params[model])])
                
                
                
def renaming(folder_name, model="fourcastnetv2", year=2000, remove_old=False, remove_waiting:int=15, cut=False,
            ibtracs_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv"):
    print(year, model, flush=True)
    df_ibtracs = pd.read_csv(ibtracs_path, dtype="string", na_filter=False)
    model_folder = "panguweather" if model=="pangu" else model
    
    key = lambda x: (get_start_date_nc(x), get_lead_time(x))
    
    ldts = [l for l in range(6,174,6)]
    filelist = []
    for ldt in ldts:
        filelist.extend(glob.glob(folder_name + f"{model_folder}/{model}_{year}*_ldt_{ldt}.nc"))
    fileList = sorted(filelist, key=key)
    
    i = 1
    done = []
    for file in fileList:
        start_iso_time, end_iso_time = get_start_date_nc(file), get_end_date_nc(file)
        print(f"{i}/{len(fileList)}: {start_iso_time} to {end_iso_time}", flush=True)
        i += 1
        tc_ids = []
        df_track_tmp = df_ibtracs[df_ibtracs["ISO_TIME"].astype("datetime64[ns]")==start_iso_time]
        possible_ids = df_track_tmp["SID"].unique()
        for id in possible_ids:
            df_track_tmp2 = df_ibtracs[df_ibtracs["SID"]==id]
            if df_track_tmp2[df_track_tmp2["ISO_TIME"].astype("datetime64[ns]")==end_iso_time]["ISO_TIME"].values.shape[0] > 0:
                tc_ids.append(id)
        
        if False in [os.path.isfile(os.path.dirname(file) + f"/{model}_{start_iso_time}_to"+\
            f"_{end_iso_time}_ldt_{get_lead_time(file)}_{tc_id}.nc") for tc_id in tc_ids]:
            try:
                ds = xr.open_dataset(file, engine="netcdf4")
            except OSError: # some files are corrupted
                done.append(file)
                continue
            for tc_id in tc_ids:
                save_name = os.path.dirname(file) + f"/{model}_{start_iso_time}_to_{end_iso_time}_ldt_{get_lead_time(file)}_{tc_id}.nc"
                if not os.path.isfile(save_name):
                    ds.to_netcdf(save_name, engine="netcdf4")
        
        done.append(file)
        if len(done) >= 50:
            if remove_old:
                print(f"!!! Removing 50 old files !!! {remove_waiting}s to STOP if this is a mistake", flush=True)
                import time
                time.sleep(remove_waiting)
                subprocess.run(["rm", "-r", *done])
                done = []
            if cut:
                cut_save_in_series(ds_folder=folder_name, models=[model], year=year, output_path=folder_name, parallel=False, remove=remove_old,
                                   df_tracks_path=ibtracs_path)
        if file==fileList[-1]:
            if remove_old:
                print(f"!!! Removing {len(done)} old files !!! {remove_waiting}s to STOP if this is a mistake", flush=True)
                import time
                time.sleep(remove_waiting)
                subprocess.run(["rm", "-r", *done])
                done = []
            if cut:
                cut_save_in_series(ds_folder=folder_name, models=[model], year=year, output_path=folder_name, parallel=False, remove=remove_old,
                                   df_tracks_path=ibtracs_path)
    print("Done", flush=True)