import numpy as np
import xarray as xr
import pandas as pd
import os, glob, pickle, sys
from scipy.stats import skew, kurtosis, iqr
from utils.main_utils import get_lead_time, get_start_date_nc

stats_list = ["min", "max", "mean", "std", "q1", "median", "q3", "iqr", "skew", "kurtosis"]
stats_fcts = {"mean":lambda x: np.mean(x, axis=(-2, -1)), "std":lambda x: np.std(x, axis=(-2, -1)), "min":lambda x: np.min(x, axis=(-2, -1)), 
              "max":lambda x: np.max(x, axis=(-2, -1)), "median":lambda x: np.median(x, axis=(-2, -1)), 
              "q1":lambda x: np.quantile(x, 0.25, axis=(-2,-1)), "q3":lambda x: np.quantile(x, 0.75, axis=(-2, -1)), 
              "iqr":lambda x: iqr(x, axis=(-2, -1)), "skew":lambda x: skew(x, axis=(-2, -1)), "kurtosis":lambda x: kurtosis(x, axis=(-2, -1)),
              }



def get_forecast_data(data_path, model, lead_time, tc_id):
    
    model = "pangu" if model=="panguweather" else model
    model_folder = "panguweather" if model=="pangu" else model
    
    data_path = os.path.join(data_path, model_folder)
    
    key = lambda x: (get_start_date_nc(x), get_lead_time(x))
    
    data_list = sorted(glob.glob(os.path.join(data_path, f"{model}*{tc_id}*small.nc")), key=key)
    data_list = [path for path in data_list if get_lead_time(path)>=lead_time]
    
    data = []
    isos = []
    for path in data_list:
        ds = xr.open_dataset(path)
        step = lead_time // 6
        if isinstance(ds.time.values[step-1], np.timedelta64):
            iso = np.datetime64(get_start_date_nc(path) + ds.time.values[step-1])
        else:
            iso = ds.time.values[step-1]
        if iso not in isos:
            isos.append(iso)
            ds_new = ds.isel(time=step-1)
            if model=="graphcast":
                ds_new["time"] = iso
            data.append(ds_new)
        ds.close()
        del ds
    #print("Acquired data from AI model")
    
    return data


def get_ibtracs_data(data_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv",
                     lead_time=24, tc_id=""):
    
    df = pd.read_csv(data_path, dtype="string", na_filter=False)
    df = df[df["SID"] == tc_id]
    
    step = lead_time // 6
    # Remove first date (beginning of storm hence, not predicted)
    df = df.iloc[1:]
    # Remove steps that can not be attained with the given lead time
    df = df.iloc[step-1:]
    
    wind_col = "USA_WIND"
    pres_col = [col for col in df.columns if "_PRES" in col and "PRES_" not in col and col!="WMO_PRES"]
    key = lambda x: np.count_nonzero(df[x].values.astype("string")!=" ")
    
    pres_col = sorted(pres_col, key=key)[-1] # the one with the highest number of values reported
    #print(pres_col)
    
    idxs = [idx for idx in df.index if df.loc[idx, wind_col]!=" " and df.loc[idx, pres_col]!=" "] # remove rows with missing values
    
    df = df.loc[idxs]
    
    data = df[["ISO_TIME", wind_col, pres_col]]
    #print("Acquired data from IBTrACS")
    return data


def get_data(data_path, model, lead_time, tc_id, 
             df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv"):
    assert lead_time % 6 == 0, "lead_time must be a multiple of 6"
    
    data_forecast = get_forecast_data(data_path, model, lead_time, tc_id)
    data_ibtracs = get_ibtracs_data(df_path, lead_time, tc_id)
    
    isos = data_ibtracs["ISO_TIME"].values.astype("datetime64[ns]")
    
    data_forecast = [data for data in data_forecast if data["time"].values in isos]
    
    # for graphcast we cannot run on dates like 01.mm.yyyyT00 so we remove the corresponding ibtracs data
    data_ibtracs = data_ibtracs[data_ibtracs["ISO_TIME"].astype("datetime64[ns]").isin([data["time"].values for data in data_forecast])] 

    return data_forecast, data_ibtracs


def data_loader(data_path, model_name, lead_time, season:int, basin:str,
                df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv",
                save_path = "/users/lpoulain/louis/plots/linear_model/"):
    
    forecast_wind_list, forecast_pres_list, truth_wind_list, truth_pres_list = [], [], [], []
    empty_tc = []
    df = pd.read_csv(df_path, dtype="string", na_filter=False)
    if basin.lower()!="all":
        tc_ids = df[(df["SEASON"].astype(int)==season) & (df["BASIN"]==basin)]["SID"].unique()
    else:
        tc_ids = df[(df["SEASON"].astype(int)==season)]["SID"].unique()
    
    print(f"{model_name} - Basin: {basin} - {len(tc_ids)} TCs")
    
    if False in [os.path.exists(save_path + f"Data/{season}/{model_name}_{lead_time}_{season}_{basin}_{x}.pkl") for x in\
                ["wnd_forecasts", "wnd_truth", "pres_forecasts", "pres_truth", "tc_ids"]]:
        for i, tc_id in enumerate(tc_ids):
            if len(tc_ids)>3:
                if i % (len(tc_ids)//3) == 0:
                    print(f"TC {i+1}/{len(tc_ids)}")
            forecast_data, truth_data = get_data(data_path, model_name, lead_time, tc_id, df_path)
            wind_col, pres_col = truth_data.columns[1:]
            if len(forecast_data)==0:
                empty_tc.append(tc_id)
                continue
            
            wind_truth, pres_truth = truth_data[wind_col].values.astype("float") * 0.514444, truth_data[pres_col].values.astype("float") * 100
            truth_wind_list.append(wind_truth)
            truth_pres_list.append(pres_truth)
            
            wind_forecast = np.array([np.sqrt(data.u10.values.astype("float")**2 + data.v10.values.astype("float")**2).max().max()\
                                    for data in forecast_data]).reshape(-1, 1)
            pres_forecast = np.array([data.msl.values.astype("float").min().min() for data in forecast_data]).reshape(-1, 1)
            forecast_wind_list.append(wind_forecast)
            forecast_pres_list.append(pres_forecast)
            
        tc_ids = [tc_id for tc_id in tc_ids if tc_id not in empty_tc]
        
        with open(save_path + f"Data/{season}/{model_name}_{lead_time}_{season}_{basin}_tc_ids.pkl", "wb") as f:
            pickle.dump(tc_ids, f)
        with open(save_path + f"Data/{season}/{model_name}_{lead_time}_{season}_{basin}_wnd_forecasts.pkl", "wb") as f:
            pickle.dump(forecast_wind_list, f)
        with open(save_path + f"Data/{season}/{model_name}_{lead_time}_{season}_{basin}_wnd_truth.pkl", "wb") as f:
            pickle.dump(truth_wind_list, f)
        with open(save_path + f"Data/{season}/{model_name}_{lead_time}_{season}_{basin}_pres_forecasts.pkl", "wb") as f:
            pickle.dump(forecast_pres_list, f)
        with open(save_path + f"Data/{season}/{model_name}_{lead_time}_{season}_{basin}_pres_truth.pkl", "wb") as f:
            pickle.dump(truth_pres_list, f)
            
    else:
        with open(save_path + f"Data/{season}/{model_name}_{lead_time}_{season}_{basin}_wnd_forecasts.pkl", "rb") as f:
            forecast_wind_list = pickle.load(f)
        with open(save_path + f"Data/{season}/{model_name}_{lead_time}_{season}_{basin}_wnd_truth.pkl", "rb") as f:
            truth_wind_list = pickle.load(f)
        with open(save_path + f"Data/{season}/{model_name}_{lead_time}_{season}_{basin}_pres_forecasts.pkl", "rb") as f:
            forecast_pres_list = pickle.load(f)
        with open(save_path + f"Data/{season}/{model_name}_{lead_time}_{season}_{basin}_pres_truth.pkl", "rb") as f:
            truth_pres_list = pickle.load(f)
        with open(save_path + f"Data/{season}/{model_name}_{lead_time}_{season}_{basin}_tc_ids.pkl", "rb") as f:
            tc_ids = pickle.load(f)
            
    return forecast_wind_list, forecast_pres_list, truth_wind_list, truth_pres_list, tc_ids



def statistics_loader(data_path, model_name, lead_time, season:int, basin,
                df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv",
                save_path = "/users/lpoulain/louis/plots/xgboost/"):
    
    # a more general version of data_loader

    wind_statistic_list, pres_statistic_list = {stat:[] for stat in stats_list}, {stat:[] for stat in stats_list}
    wind_truth_list, pres_truth_list = [], []
    
    empty_tc = []
    df = pd.read_csv(df_path, dtype="string", na_filter=False)
    if basin.lower()!="all":
        tc_ids = df[(df["SEASON"].astype(int)==int(season)) & (df["BASIN"]==basin)]["SID"].unique()
    else:
        tc_ids = df[(df["SEASON"].astype(int)==int(season))]["SID"].unique()
    
    print(f"{model_name} - {season}, {basin} ({len(tc_ids)} TCs)")
    
    save_names = [save_path + f"Data/{season}/statistics_{model_name}_{lead_time}_{season}_{basin}_{x}.pkl" for x in ["wnd", "pres"]] +\
                [save_path + f"Data/{season}/{model_name}_{lead_time}_{season}_{basin}_{x}.pkl" for x in ["wnd_truth", "pres_truth", "tc_ids"]]
    
    if False in [os.path.isfile(save_name) for save_name in save_names]:
        
        for i, tc_id in enumerate(tc_ids):
            if len(tc_ids)>3:
                if i % (len(tc_ids)//3) == 0:
                    print(f"TC {i+1}/{len(tc_ids)}")
            forecast_data, truth_data = get_data(data_path, model_name, lead_time, tc_id, df_path)
            wind_col, pres_col = truth_data.columns[1:]
            if len(forecast_data)==0:
                empty_tc.append(tc_id)
                continue
            
            wind_truth, pres_truth = truth_data[wind_col].values.astype("float") * 0.514444, truth_data[pres_col].values.astype("float") * 100
            try:
                wind_truth_list.append(wind_truth)
                pres_truth_list.append(pres_truth)
            except ValueError:
                print(f"Error with {tc_id}")
                print(pres_col)
                print(wind_truth)
                print(pres_truth)
                raise ValueError
            
            try:
                wind_forecast = np.array([np.sqrt(data.u10.values.astype("float")**2 + data.v10.values.astype("float")**2)\
                                    for data in forecast_data]).reshape(-1, 241, 241)
            except ValueError:
                print(f"Error with {tc_id}")
                print([(data.u10.values.astype("float")**2 + data.v10.values.astype("float")**2).shape for data in forecast_data])
                raise ValueError
            pres_forecast = np.array([data.msl.values.astype("float") for data in forecast_data]).reshape(-1, 241, 241)
            
            for stat in stats_list:
                wind_statistic_list[stat].append(stats_fcts[stat](wind_forecast))
                pres_statistic_list[stat].append(stats_fcts[stat](pres_forecast))
            
        tc_ids = [tc_id for tc_id in tc_ids if tc_id not in empty_tc]
        
        with open(save_path + f"Data/{season}/{model_name}_{lead_time}_{season}_{basin}_tc_ids.pkl", "wb") as f:
            pickle.dump(tc_ids, f)
        with open(save_path + f"Data/{season}/statistics_{model_name}_{lead_time}_{season}_{basin}_wnd.pkl", "wb") as f:
            pickle.dump(wind_statistic_list, f)
        with open(save_path + f"Data/{season}/{model_name}_{lead_time}_{season}_{basin}_wnd_truth.pkl", "wb") as f:
            pickle.dump(wind_truth_list, f)
        with open(save_path + f"Data/{season}/statistics_{model_name}_{lead_time}_{season}_{basin}_pres.pkl", "wb") as f:
            pickle.dump(pres_statistic_list, f)
        with open(save_path + f"Data/{season}/{model_name}_{lead_time}_{season}_{basin}_pres_truth.pkl", "wb") as f:
            pickle.dump(pres_truth_list, f)
            
    else:
        with open(save_path + f"Data/{season}/statistics_{model_name}_{lead_time}_{season}_{basin}_wnd.pkl", "rb") as f:
            wind_statistic_list = pickle.load(f)
        with open(save_path + f"Data/{season}/{model_name}_{lead_time}_{season}_{basin}_wnd_truth.pkl", "rb") as f:
            wind_truth_list = pickle.load(f)
        with open(save_path + f"Data/{season}/statistics_{model_name}_{lead_time}_{season}_{basin}_pres.pkl", "rb") as f:
            pres_statistic_list = pickle.load(f)
        with open(save_path + f"Data/{season}/{model_name}_{lead_time}_{season}_{basin}_pres_truth.pkl", "rb") as f:
            pres_truth_list = pickle.load(f)
        with open(save_path + f"Data/{season}/{model_name}_{lead_time}_{season}_{basin}_tc_ids.pkl", "rb") as f:
            tc_ids = pickle.load(f)
        
    return wind_statistic_list, pres_statistic_list, wind_truth_list, pres_truth_list, tc_ids
    