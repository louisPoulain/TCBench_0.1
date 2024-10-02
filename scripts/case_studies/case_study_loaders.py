import numpy as np
import pandas as pd
import xarray as xr
from joblib import load
import xgboost as xgb
import glob, sys
import torch

from models.loading_utils import stats_list, stats_fcts
from utils.main_utils import get_start_date_nc, get_lead_time
from case_studies.case_study_utils import remove_duplicates
from models.cnn_loaders import CNN4PP_Dataset, linear
from models.cnn_blocks import CNN4PP


def get_era5(tc_id, 
             data_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/",
             df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv"):
    
    
    df = pd.read_csv(df_path, dtype="string", na_filter=False)
    df = df[df["SID"]==tc_id]
    
    wind_col = "USA_WIND"
    pres_cols = [col for col in df.columns if "_PRES" in col and "PRES_" not in col and col!="WMO_PRES"]
    
    idxs = [idx for idx in df.index if df.loc[idx, wind_col]!=" "]
        
    key = lambda x: np.count_nonzero(df[x].values.astype("string")!=" ")
    pres_col = sorted(pres_cols, key=key)[-1] # the one with the highest number of values reported
    idxs = [idx for idx in idxs if df.loc[idx, pres_col]!=" "] # remove rows with missing values
    
    if df.index[0] not in idxs:
        idxs = [df.index[0], *idxs]
    df_filtered = df.loc[idxs]

    valid_dates = df_filtered["ISO_TIME"].values
    year_month = list(set([date[:7] for date in valid_dates]))[0]
    
    era5 = xr.load_dataset(data_path + f"ERA5_{year_month[:4]}_{year_month[5:7]}_surface.grib")
    
    return era5, valid_dates, df_filtered, pres_col, df



def load_tc_forecast(model_name, tc_id, pp_type, pp_params, lead_time,
                     data_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/"):
    
    pp_type = pp_type.lower()
    assert pp_type in ["linear", "xgboost", "cnn"], f"Unknown pp_type: {pp_type}."
    model_name = "pangu" if model_name=="panguweather" else model_name
    data_folder = "panguweather" if model_name=="pangu" else model_name
    
    key = lambda x: (get_start_date_nc(x), get_lead_time(x))
    data_list = sorted(glob.glob(data_path+f"{data_folder}/{model_name}_*_{tc_id}_small.nc"), key=key)
    data_list = [p for p in data_list if get_lead_time(p)>=lead_time]
    data_list = remove_duplicates(data_list)
    
    if pp_type == "linear":
        data = []
        
        for path in data_list:
            ds = xr.load_dataset(path)
            ds = ds.isel(time=lead_time//6-1)
            tmp_time = ds.time.values if isinstance(ds.time.values, np.datetime64) else\
                            np.datetime64(ds.time.values + get_start_date_nc(path))
            max_wind = np.sqrt(ds.u10.values**2+ds.v10.values**2).max()
            min_pres = ds.msl.values.min()
            
            loc_idx = np.unravel_index(ds.msl.values.argmin(), ds.msl.values.shape)
            lat, lon = ds.lat.values[loc_idx[0]], ds.lon.values[loc_idx[1]]
            data.append((tmp_time, max_wind, min_pres, lat, lon, None))
        
        return data

    if pp_type == "xgboost":
        data = []
        
        stats = pp_params.get("stats", [])
        stats_wind = pp_params.get("stats_wind", ["max"])
        stats_pres = pp_params.get("stats_pres", ["min"])
        jsdiv = pp_params.get("jsdiv", False)
        basin = pp_params.get("basin", "NA")
        train_seasons = pp_params.get("train_seasons", ['2000'])
        
        stats_wind.extend(stats)
        stats_pres.extend(stats)
        
        stats_wind = sorted(list(set(stats_wind)), key=lambda x: stats_list.index(x))
        stats_pres = sorted(list(set(stats_pres)), key=lambda x: stats_list.index(x))
        
        for path in data_list:
            winds = []
            press = []
            ds = xr.load_dataset(path)
            ds = ds.isel(time=lead_time//6-1)
            tmp_time = ds.time.values if isinstance(ds.time.values, np.datetime64) else\
                            np.datetime64(ds.time.values + get_start_date_nc(path))
            wind = np.sqrt(ds.u10.values**2+ds.v10.values**2)
            pres = ds.msl.values
            loc_idx = np.unravel_index(pres.argmin(), pres.shape)
            lat, lon = ds.lat.values[loc_idx[0]], ds.lon.values[loc_idx[1]]
            for stat in stats_wind:
                winds.append(stats_fcts[stat](wind))
            for stat in stats_pres:
                press.append(stats_fcts[stat](pres))
            winds, press = np.array(winds).reshape(-1, 1), np.array(press).reshape(-1, 1)
            csts = np.load("/users/lpoulain/louis/plots/xgboost/Constants/" + f"norma_cst_{model_name}_{lead_time}h"\
                        + f"_{basin}_{'_'.join(sorted(train_seasons))}{'_jsdiv' if jsdiv else ''}.npy")
            truth_constants = np.load("/users/lpoulain/louis/plots/xgboost/Constants/" + f"norma_cst_truth_{model_name}_{lead_time}h"\
                        + f"_{basin}_{'_'.join(sorted(train_seasons))}{'_jsdiv' if jsdiv else ''}.npy")
            
            winds = (winds - csts[0]) / csts[1]
            press = (press - csts[2]) / csts[3]
            data.append((tmp_time, xgb.DMatrix(winds, feature_names=stats_wind), xgb.DMatrix(press, feature_names=stats_pres), 
                         lat, lon, truth_constants))
        
        return data
    
    if pp_type == "cnn":
        raise ValueError(f"Not implemented yet: {pp_type}.")
    
    

def load_one_tc_forecast(model_name, ds_forecast, pp_type, pp_params, lead_time):
    
    pp_type = pp_type.lower()
    assert pp_type in ["linear", "xgboost", "cnn"], f"Unknown pp_type: {pp_type}."
    model_name = "pangu" if model_name=="panguweather" else model_name
    
    fcn_var = '__xarray_dataarray_variable__'
        
    if pp_type == "linear":
        
        if model_name!="fourcastnetv2":
            max_wind = np.sqrt(ds_forecast.u10.values**2+ds_forecast.v10.values**2).max()
            min_pres = ds_forecast.msl.values.min()
        else:
            u10_idx, v10_idx, msl_idx = list(ds_forecast.channel).index('u10m'), list(ds_forecast.channel).index('v10m'), list(ds_forecast.channel).index('msl')
            max_wind = np.sqrt(ds_forecast[fcn_var].values[0, u10_idx]**2+ds_forecast[fcn_var].values[0, v10_idx]**2).max()
            min_pres = ds_forecast[fcn_var].values[0, msl_idx].min()
        
        data = (max_wind, min_pres, 0)
        return data

    if pp_type == "xgboost":
        
        stats = pp_params.get("stats", [])
        stats_wind = pp_params.get("stats_wind", [])
        stats_pres = pp_params.get("stats_pres", [])
        jsdiv = pp_params.get("jsdiv", False)
        basin = pp_params.get("basin", "NA")
        train_seasons = pp_params.get("train_seasons", ['2000'])
        
        stats_wind.extend(stats)
        stats_pres.extend(stats)
        
        stats_wind = sorted(list(set(stats_wind)), key=lambda x: stats_list.index(x))
        stats_pres = sorted(list(set(stats_pres)), key=lambda x: stats_list.index(x))
        winds = []
        press = []
        
        if model_name!="fourcastnetv2":
            wind = np.sqrt(ds_forecast.u10.values**2+ds_forecast.v10.values**2)
            pres = ds_forecast.msl.values
        else:
            u10_idx, v10_idx, msl_idx = list(ds_forecast.channel).index('u10m'), list(ds_forecast.channel).index('v10m'), list(ds_forecast.channel).index('msl')
            wind = np.sqrt(ds_forecast[fcn_var].values[0, u10_idx]**2+ds_forecast[fcn_var].values[0, v10_idx]**2)
            pres = ds_forecast[fcn_var].values[0, msl_idx]

        for stat in stats_wind:
            winds.append(stats_fcts[stat](wind))
        for stat in stats_pres:
            press.append(stats_fcts[stat](pres))
            
        winds, press = np.array(winds).reshape(1, -1), np.array(press).reshape(1, -1)
        csts = np.load("/users/lpoulain/louis/plots/xgboost/Constants/" + f"norma_cst_{model_name}_{lead_time}h"\
                    + f"_{basin}_{'_'.join(sorted(train_seasons))}{'_jsdiv' if jsdiv else ''}_min_max_mean_std_q1_median_q3_iqr_skew_kurtosis.npy")
        idx_wind = [stats_list.index(stat) for stat in stats_wind]
        idx_pres = [stats_list.index(stat) for stat in stats_pres]
        
        truth_constants = np.load("/users/lpoulain/louis/plots/xgboost/Constants/" + f"norma_cst_truth_{model_name}_{lead_time}h"\
                    + f"_{basin}_{'_'.join(sorted(train_seasons))}{'_jsdiv' if jsdiv else ''}.npy")
        
        csts_wind = csts[:2, idx_wind]
        csts_pres = csts[2:, idx_pres]
        # put csts[0], csts[1] etc acocrding to what written in xgboost_utils
        winds = (winds - csts_wind[0]) / csts_wind[1]
        press = (press - csts_pres[0]) / csts_pres[1]
        data = (xgb.DMatrix(winds, feature_names=stats_wind), xgb.DMatrix(press, feature_names=stats_pres), truth_constants)
        
        return data
    
    if pp_type == "cnn":
        
        train_seasons = pp_params.get("train_seasons", ['2000'])
        data_path = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/"
        ibtracs_path = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv"
        save_path = "/users/lpoulain/louis/plots/cnn/"
        
        dataset = CNN4PP_Dataset(data_path, model_name, ibtracs_path, train_seasons, pres=True, save_path=save_path, 
                                 train_seasons=train_seasons, noprint=True, create_input=False)
        fields_mean = dataset.mean
        fields_std = dataset.std
        wind_extent, pres_extent = dataset.wind_extent, dataset.pres_extent
        
        coords = pp_params.get("coords", [0., 0.])
        coords = np.array([coords[0], coords[1], float(lead_time)])
        coords_no_renorm = coords.copy()
        
        coords[0] = linear([-90, 90], [min(wind_extent[0], pres_extent[0]), max(wind_extent[1], pres_extent[1])], coords[0])
        coords[1] = linear([0, 359.75], [min(wind_extent[0], pres_extent[0]), max(wind_extent[1], pres_extent[1])], coords[1])
        coords[2] = linear([6., 168.], [min(wind_extent[0], pres_extent[0]), max(wind_extent[1], pres_extent[1])], coords[2])
        
        if model_name!="fourcastnetv2":
            wind = np.sqrt(ds_forecast.u10.values**2+ds_forecast.v10.values**2)
            pres = ds_forecast.msl.values
        else:
            u10_idx, v10_idx, msl_idx = list(ds_forecast.channel).index('u10m'), list(ds_forecast.channel).index('v10m'), list(ds_forecast.channel).index('msl')
            wind = np.sqrt(ds_forecast[fcn_var].values[0, u10_idx]**2+ds_forecast[fcn_var].values[0, v10_idx]**2)
            pres = ds_forecast[fcn_var].values[0, msl_idx]
        
        wind = (wind - fields_mean[0]) / fields_std[0]
        pres = (pres - fields_mean[1]) / fields_std[1]
        truth_constants = [dataset.target_mean, dataset.target_std]
        
        data = (torch.tensor(np.concatenate([wind.reshape(1, 1, *wind.shape), pres.reshape(1, 1, *pres.shape)], axis=1)).float(),
                torch.tensor(coords).view(-1, *coords.shape).float(), truth_constants, coords_no_renorm)
        return data

    


def load_pp_model(model_name, pp_type, pp_params, ldt):
    
    pp_type = pp_type.lower()
    assert pp_type in ["linear", "xgboost", "cnn"], f"Unknown pp_type: {pp_type}."
    model_name = "pangu" if model_name=="panguweather" else model_name
    train_seasons = pp_params.get("train_seasons", [])
    
    if pp_type == "linear":
        dim = pp_params.get("dim", 2)
        basin = pp_params.get("basin", "NA")
        train_seasons = train_seasons[0] if isinstance(train_seasons, list) else train_seasons
        
        if dim==1:
            models = [load("/users/lpoulain/louis/plots/linear_model/" +\
                    f"Models/{pp_type}_model_wind_{model_name}_{ldt}_{train_seasons}_b_{basin}.joblib"),
                      load("/users/lpoulain/louis/plots/linear_model/" +\
                    f"Models/{pp_type}_model_pres_{model_name}_{ldt}_{train_seasons}_b_{basin}.joblib")]
        else:
            models = [load("/users/lpoulain/louis/plots/linear_model/" +\
                    f"Models/{pp_type}_model2d_{model_name}_{ldt}_{train_seasons}_b_{basin}.joblib")]
        return models, pp_params
    
    if pp_type == "xgboost":
        jsdiv = pp_params.get("jsdiv", False)
        depth = pp_params.get("depth", 3)
        epochs = pp_params.get("epochs", 200)
        lr = pp_params.get("lr", 0.1)
        gamma = pp_params.get("gamma", 0.)
        sched = pp_params.get("sched", True)
        stats = pp_params.get("stats", [])
        stats_wind = pp_params.get("stats_wind", [])
        stats_pres = pp_params.get("stats_pres", [])
        basin = pp_params.get("basin", "NA")
        
        stats_wind = sorted(list(set(stats_wind + stats)), key=lambda x: stats_list.index(x))
        stats_pres = sorted(list(set(stats_pres + stats)), key=lambda x: stats_list.index(x))
        pp_params_tmp = pp_params.copy()
        
        
        model_path = "/users/lpoulain/louis/plots/xgboost/Models/"
        
        if len(stats_wind)!=0:
            model_wind_save_name = f"xgb_wind_{model_name}_{ldt}h{'_jsdiv' if jsdiv else ''}_basin_{basin}_"\
                        + f"{'_'.join(train_seasons)}_depth_{depth}_epoch_{epochs}_lr_{lr}_g_{gamma}"\
                        + (f"_sched" if sched else "")\
                        + f"_{'_'.join(stat for stat in stats_wind)}.json"
        else:
            model_wind_save_name = glob.glob(model_path + f"xgb_wind_{model_name}_{ldt}h{'_jsdiv' if jsdiv else ''}_basin_{basin}_"\
                        + f"{'_'.join(train_seasons)}_depth_{depth}_epoch_{epochs}_lr_{lr}_g_{gamma}"\
                        + (f"_sched" if sched else "")\
                        + "*.json")[0] # get the first model found that has all other characteristics
            model_wind_save_name = model_wind_save_name.split("/")[-1]
            
            if sched:
                s = model_wind_save_name.split("_sched_")[-1].split(".")[0]
                stats_wind = [s for s in s.split("_")]
                pp_params_tmp["stats_wind"] = stats_wind
            else:
                s = model_wind_save_name.split(f"g_{gamma}_")[-1].split(".")[0]
                stats_wind = [s for s in s.split("_")]
                pp_params_tmp["stats_wind"] = stats_wind
            
        if len(stats_pres)!=0:
            model_pres_save_name = f"xgb_pres_{model_name}_{ldt}h{'_jsdiv' if jsdiv else ''}_basin_{basin}_"\
                        + f"{'_'.join(train_seasons)}_depth_{depth}_epoch_{epochs}_lr_{lr}_g_{gamma}"\
                        + (f"_sched" if sched else "")\
                        + f"_{'_'.join(stat for stat in stats_pres)}.json"
        else:
            model_pres_save_name = glob.glob(model_path + f"xgb_pres_{model_name}_{ldt}h{'_jsdiv' if jsdiv else ''}_basin_{basin}_"\
                        + f"{'_'.join(train_seasons)}_depth_{depth}_epoch_{epochs}_lr_{lr}_g_{gamma}"\
                        + (f"_sched" if sched else "")\
                        + "*.json")[0] # get the first model found that has all other characteristics
            model_pres_save_name = model_pres_save_name.split("/")[-1]
            
            if sched:
                s = model_pres_save_name.split("_sched_")[-1].split(".")[0]
                stats_pres = [s for s in s.split("_")]
                pp_params_tmp["stats_pres"] = stats_pres
            else:
                s = model_pres_save_name.split(f"g_{gamma}_")[-1].split(".")[0]
                stats_pres = [s for s in s.split("_")]
                pp_params_tmp["stats_pres"] = stats_pres
        
        model_wind = xgb.Booster()
        model_wind.load_model(model_path + model_wind_save_name)
        model_pres = xgb.Booster()
        model_pres.load_model(model_path + model_pres_save_name)
        
        
        return (model_wind, model_pres), pp_params_tmp
    
    if pp_type == "cnn":
        
        crps = pp_params.get("crps", False)
        optim = pp_params.get("optim", "adam")
        sched = pp_params.get("sched", False)
        if sched:
            scheduler = '_cosine_annealing'
        else:
            scheduler = '_none'
        lr = pp_params.get("lr", 0.001)
        epochs = pp_params.get("epochs", 20)
        pp_params_tmp = pp_params.copy()
        
        model_path = "/users/lpoulain/louis/plots/cnn/Models/"
        if train_seasons==[]:
            model_path = glob.glob(model_path + f"{model_name}_{'crps_'if crps else ''}pres_True_epochs_{epochs}_lr_{lr}_optim_{optim}"\
                        + f"_sched{scheduler}_*.pt")[0]
            seasons_tmp = model_path.split("/")[-1].split(f"_sched{scheduler}_")[-1].split(".")[0].split("_")
            pp_params_tmp["train_seasons"] = seasons_tmp
        else:    
            model_path += f"{model_name}_{'crps_'if crps else ''}pres_True_epochs_{epochs}_lr_{lr}_optim_{optim}"\
                        + f"_sched{scheduler}_{'_'.join(sorted(train_seasons))}.pt"
                
        #model = CNN4PP(in_channels=2, out_channels=8, kernel_size=7, stride=1, padding=1, bias=True, deterministic=deterministic)
        model = torch.load(model_path, map_location="cpu")
        model.eval()
        
        return [model.float()], pp_params_tmp
    
    
