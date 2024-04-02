import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os, sys, glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import torch

from utils.main_utils import get_start_date_nc, get_lead_time
from utils.cut_region import haversine, cut_rectangle
from case_studies.case_study_utils import find_trajectory_point, remove_duplicates
from case_studies.case_study_loaders import get_era5, load_tc_forecast, load_pp_model, load_one_tc_forecast

colors_dic = {"pangu": 'blue', "graphcast": 'green', "fourcastnetv2":"orange", "era5": 'red', "truth": 'black'}


def trajectory(tc_id, model_names, max_lead_time=72, ldt_step=6, pp_type=None, pp_params=None,
                data_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/",
                df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv",
                plot_path="/users/lpoulain/louis/plots/case_studies/"):
    
    """
    Plot the trajectory of a tropical cyclone and compare it to IBTrACS (truth) and to the ERA5 reanalysis. The TC intensity is also compared
    
    Args:
        - tc_id (str): 
            ID of the tropical cyclone to plot. Currently implemented for Katrina (2005), Kai-Tak (2000) and Norbert (2008)
            In fact it should work for any id, except when you pass fourcastnetv2 as model name (hence the limitation to 3 tcs)
        - model_names (list of str): 
            List of the models to compare. Currently implemented for pangu, graphcast and fourcastnetv2
            fourcastnetv2 does not currently support the pp_type argument
        - max_lead_time (int or List): 
            Maximum lead time to plot. One figure is plot for each lead time from 6 to max_lead_time (every ldt_step, see after)
            If max_lead is a list, ldt_step is ignored and the lead times plotted are the ones in max_lead_time
        - ldt_step (int):
            Step of the lead time. The lead times plotted will be 6, 6+ldt_step, 6+2*ldt_step, ...
        - pp_type (str):
            None by default (no post-processing, only plots the outputted values by the ai models)
            Type of post-processing to use. Currently implemented for None, "linear", "xgboost" and "cnn"
        - pp_params (dict): 
            Dictionary of the parameters for the post-processing. 
            The keys are the names of the parameters and the values are the values of the parameters. Check cnn_main.py for the list of parameters.
        - data_path (str):
            Path to the data folder. Default is "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/".
            The specific folder for each model is inferred from the model_name (please follow the folder names in your data_path)
        - df_path (str):
            Path to the csv file containing the filtered TC tracks. 
            Default is "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv"
        - plot_path (str):
            Path to the folder where the plots will be saved. Default is "/users/lpoulain/louis/plots/case_studies/"
            One figure is save for each lead time and one figure is saved for the mean error
            
    Returns:
        - None
    """
    
    
    tc_sid_to_name = {"2005236N23285":"Katrina (2005)", "2000185N15117":"Kai-Tak (2000)", "2008278N13261": "Norbert (2008)"}
    assert tc_id in tc_sid_to_name.keys(), f"tc_id should be one of {list(tc_sid_to_name.keys())}"
    assert ldt_step%6==0, "ldt_step should be a multiple of 6"
    assert pp_type in [None, "linear", "xgboost", "cnn"], "pp_type should be one of [None, 'linear', 'xgboost', 'cnn']"

    era5, valid_dates, df_filt, pres_col, df = get_era5(tc_id, data_path+"ERA5/", df_path)
    idxs = [i for i in range(era5.valid_time.shape[0]) if era5.valid_time[i].values.astype("datetime64[ns]") in valid_dates.astype("datetime64[ns]")]
    era5 = era5.isel(step=idxs)
    
    if isinstance(max_lead_time, int):
        lead_times = np.arange(6, max_lead_time+6, ldt_step)
    else:
        lead_times = max_lead_time
    
    mean_err = {}
    for model in model_names:
        mean_err[model] = []
        if pp_type=='cnn':
            mean_err[model+"_"+pp_type] = []
    mean_err['era5'] = []
    
            
    for lead_time in lead_times:
        print(f"Lead time: {lead_time}h")
        grid = ([f"Traj_{x}" for x in model_names+["ERA5"]], 
                [f"Wind" for x in model_names+["ERA5"]],
                [f"Pres" for x in model_names+["ERA5"]],
                [f"km_error" for x in model_names+["ERA5"]]
                )
        
        
        df_tmp = df_filt.loc[df_filt.index[lead_time//6:]]
        times = df_tmp["ISO_TIME"].values.astype("datetime64[ns]")

        truth_lats, truth_lons = df_tmp["LAT"].values.astype(np.float32), df_tmp["LON"].values.astype(np.float32)
        truth_lons = [t+360 if t<0 else t for t in truth_lons]
        
        truth_wind, truth_pres = df_tmp["USA_WIND"].values.astype(np.float32)*0.514444, df_tmp[pres_col].values.astype(np.float32)*100
        
        lat_extent = [min(truth_lats), max(truth_lats)]
        lon_extent = [min(truth_lons), max(truth_lons)]
        lon_extent = [t-360 if t>180 else t for t in lon_extent]
        
        center = [(lat_extent[0]+lat_extent[1])/2, (lon_extent[0]+lon_extent[1])/2]
        radius = [(lat_extent[1]-lat_extent[0])/2, (lon_extent[1]-lon_extent[0])]
        lat_extent = [center[0]-max(radius)-4, center[0]+max(radius)+4]
        lon_extent = [center[1]-max(radius)-4, center[1]+max(radius)+4]
        center_lon = 0
        
        per_subplot_kw = {"Traj_"+model: dict(projection=ccrs.PlateCarree(center_lon)) for model in model_names+["ERA5"]}
        fig, axs = plt.subplot_mosaic(grid, figsize=((len(model_names)+1)*5, len(grid)*6), per_subplot_kw=per_subplot_kw,
                                      gridspec_kw={"hspace":0.1, "wspace":0.1, "top":0.93, "bottom":0.05, "left":0.05, "right":0.95},
                                      sharey=False, sharex=False)
        
        for model in model_names:
            data_folder = "panguweather" if model=="pangu" else model
            key = lambda x: (get_start_date_nc(x), get_lead_time(x))
            data_list = sorted(glob.glob(data_path+f"{data_folder}/{model}_*_{tc_id}_small.nc"), key=key)
            data_list = [p for p in data_list if get_lead_time(p)>=lead_time]
            data_list = remove_duplicates(data_list)
            
            
            lats, lons, winds, press, err_km = [], [], [], [], []
            winds_pp, press_pp = [], []
            if pp_type=="cnn":
                lats_pp, lons_pp, err_km_pp, std_wind_pp, std_pres_pp = [], [], [], [], []
            final_times = []
            
            if pp_type is not None:
                pp_models, pp_params_tmp = load_pp_model(model, pp_type, pp_params, ldt=lead_time)
            
            c_data = 0
            for path in data_list:

                data = xr.open_dataset(path).isel(time=lead_time//6-1 if not model=="fourcastnetv2" else lead_time//6)
                time0 = get_start_date_nc(path)
                tmp_time = data.time.values if isinstance(data.time.values, np.datetime64) else\
                            np.datetime64(data.time.values + get_start_date_nc(path))

                if tmp_time in times:
                    c_data += 1
                    max_wind, min_pres, pred_lat, pred_lon = find_trajectory_point(data, 
                                                                                   truth_lats[np.where(times==tmp_time)[0][0]],
                                                                                   truth_lons[np.where(times==tmp_time)[0][0]],
                                                                                   centroid_size=2*(lead_time//6),
                                                                                   fcnv2=model=="fourcastnetv2")
                    lats.append(pred_lat)
                    lons.append(pred_lon)
                    winds.append(max_wind)
                    press.append(min_pres)
                    err_km.append(haversine(pred_lat, pred_lon, [truth_lats[np.where(times==tmp_time)[0][0]]], [truth_lons[np.where(times==tmp_time)[0][0]]]))
                    final_times.append(np.datetime64(tmp_time, 'h'))
                    
                    
                    if pp_type is not None:
                        if pp_type!="cnn":
                            input_wind, input_pres, truth_csts = load_one_tc_forecast(model, data, pp_type, pp_params_tmp, lead_time)
                        if pp_type=="linear" and pp_params.get("dim", 2)==2:
                            wind, pres = pp_models[0].predict(np.array([input_wind, input_pres]).reshape(1, -1)).reshape(-1, 1)
                        else:
                            if pp_type == "linear":
                                wind = pp_models[0].predict(np.array([input_wind]).reshape(1, -1)).reshape(-1)
                                pres = pp_models[1].predict(np.array([input_pres]).reshape(1, -1)).reshape(-1)
                            
                            if pp_type == "xgboost":
                                wind = pp_models[0].predict(input_wind, iteration_range=(0, pp_models[0].best_iteration + 1))*truth_csts[1] + truth_csts[0]
                                pres = pp_models[1].predict(input_pres, iteration_range=(0, pp_models[1].best_iteration + 1))*truth_csts[3] + truth_csts[2]
                            
                            if pp_type == "cnn":
                                
                                pp_params_tmp["coords"] = df[df["ISO_TIME"].astype("datetime64[ns]")==time0][["LAT", "LON"]].values.astype(float)[0]
                                pp_params_tmp["coords"][1] = pp_params_tmp["coords"][1] + 360 if pp_params_tmp["coords"][1]<0 else pp_params_tmp["coords"][1]
                                if c_data==1:
                                    print(pp_params_tmp['coords'])
                                
                                with torch.no_grad():
                                    input_field, coords, truth_csts, coords_no_renorm = load_one_tc_forecast(model, data, pp_type, pp_params_tmp, lead_time)
                                    if pp_params.get("crps", False):
                                        out = pp_models[0](input_field, coords)
                                        std_wind, std_pres = out.view(out.shape[1:])[1], out.view(out.shape[1:])[3]
                                        std_wind, std_pres = std_wind.cpu().numpy()*truth_csts[1][0], std_pres.cpu().numpy()*truth_csts[1][1]
                                        std_wind, std_pres = np.sqrt(std_wind**2), np.sqrt(std_pres**2)
                                        out = out.view(out.shape[1:])[::2]

                                    else:
                                        out = pp_models[0](input_field, coords)
                                        out = out.view(out.shape[1:])
                                    out = out.cpu().numpy()*truth_csts[1] + truth_csts[0]
                                    wind, pres, delta_lat, delta_lon = out
                                    
                                    lat_pp = delta_lat + coords_no_renorm[0]
                                    lon_pp = delta_lon + coords_no_renorm[1]

                                    lats_pp.append(lat_pp)
                                    lons_pp.append(lon_pp)
                                    if pp_params.get("crps", False):
                                        std_wind_pp.append(std_wind)
                                        std_pres_pp.append(std_pres)
                                    err_km_pp.append(haversine(lat_pp, lon_pp, [truth_lats[np.where(times==tmp_time)[0][0]]], 
                                                               [truth_lons[np.where(times==tmp_time)[0][0]]]))
                                    
                        winds_pp.append(wind)
                        press_pp.append(pres)
                    
            lats, lons, winds, press, err_km = np.array(lats), np.array(lons), np.array(winds), np.array(press), np.array(err_km)
            winds_pp, press_pp = np.array(winds_pp), np.array(press_pp)
            if pp_type=="cnn":
                lats_pp, lons_pp, err_km_pp = np.array(lats_pp), np.array([l - 360 if l>180 else l for l in lons_pp]), np.array(err_km_pp)
                std_wind_pp, std_pres_pp = np.array(std_wind_pp), np.array(std_pres_pp)
            
            mean_err[model].append(err_km.mean())
            if pp_type=="cnn":
                mean_err[model+"_"+pp_type].append(err_km_pp.mean())
                
            lons_plot = np.array([t-360 if t>180 else t for t in lons])
            truth_lons_plot = np.array([t-360 if t>180 else t for t in truth_lons])
            

            axs["Traj_"+model].set_extent([lon_extent[0], lon_extent[1], lat_extent[0], lat_extent[1]], crs=ccrs.PlateCarree(center_lon))
            axs["Traj_"+model].plot(lons_plot, lats, linestyle='-', marker='x', c=colors_dic[model], label=model, transform=ccrs.PlateCarree(), markersize=5)
            axs["Traj_"+model].plot(truth_lons_plot, truth_lats, '-x', c=colors_dic['truth'], label="IBTrACS (Truth)", transform=ccrs.PlateCarree(), markersize=5)
            if pp_type=='cnn':
                axs["Traj_"+model].plot(lons_pp, lats_pp, linestyle='--', marker='x', c='orange',
                                        label=model+" + CNN", transform=ccrs.PlateCarree(), markersize=5)
            axs["Traj_"+model].add_feature(cfeature.COASTLINE.with_scale('50m'))
            axs["Traj_"+model].add_feature(cfeature.BORDERS.with_scale('50m'))
            axs["Traj_"+model].legend(fontsize=13)
            gridliner = axs["Traj_"+model].gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
            gridliner.top_labels = False
            gridliner.bottom_labels = True
            gridliner.left_labels = True if model==model_names[0] else False
            gridliner.right_labels = False
            gridliner.ylines = False
            gridliner.xlines = False
            axs["Traj_"+model].set_title(f"Model: {model}", fontsize=18)
            
            
            axs["Wind"].plot(winds, c=colors_dic[model], label=model)
            if pp_type is not None:
                axs["Wind"].plot(winds_pp, c=colors_dic[model], label=f"{model} + {pp_type+str(pp_params['dim'])+'d' if pp_type=='linear' else pp_type}",
                             linestyle='--')
                if pp_type=='cnn' and pp_params.get("crps", False):
                    axs["Wind"].fill_between(range(len(winds_pp)), winds_pp-std_wind_pp, winds_pp+std_wind_pp, color=colors_dic[model], alpha=0.2, linestyle='--')
            if model==model_names[0]:
                axs["Wind"].plot(truth_wind, c=colors_dic['truth'], label="IBTrACS (Truth)")
                axs["Wind"].set_ylabel("Wind speed (m/s)", fontsize=15)
            axs["Wind"].tick_params(axis='x', which='both', top=False, bottom=True, labelbottom=False)
            axs["Wind"].legend(fontsize=13)
                
            axs["Pres"].plot(press, c=colors_dic[model], label=model)
            if pp_type is not None:
                axs["Pres"].plot(press_pp, c=colors_dic[model], label=f"{model} + {pp_type+str(pp_params['dim'])+'d' if pp_type=='linear' else pp_type}",
                             linestyle='--')
                if pp_type=='cnn' and pp_params.get("crps", False):
                    axs["Pres"].fill_between(range(len(press_pp)), press_pp-std_pres_pp, press_pp+std_pres_pp, color=colors_dic[model], alpha=0.2, linestyle='--')
            if model==model_names[0]:
                axs["Pres"].plot(truth_pres, c=colors_dic['truth'], label="IBTrACS (Truth)")
                axs["Pres"].set_ylabel("Pressure (Pa)", fontsize=15)
            axs["Pres"].tick_params(axis='x', which='both', top=False, bottom=True, labelbottom=False)
            axs["Pres"].legend(fontsize=13)
            
            
            axs["km_error"].plot(err_km, c=colors_dic[model], label=model)
            axs["km_error"].legend(fontsize=13)
            if model==model_names[0]:
                axs["km_error"].set_ylabel("Error (km)", fontsize=15)
            if pp_type=='cnn':
                axs["km_error"].plot(err_km_pp, c=colors_dic[model], label=model+" + CNN", linestyle='--')
                
                
            
        lats, lons, winds, press, err_km = [], [], [], [], []
        final_times = []
        for i, tmp_time in enumerate(times):
            idx = np.where(era5.valid_time.values==tmp_time)[0][0]

            era5_tmp = era5.isel(step=idx)

            era5_tmp = cut_rectangle(era5_tmp, df_filt, tc_id, tmp_time)
            max_wind, min_pres, pred_lat, pred_lon = find_trajectory_point(era5_tmp,
                                                                           truth_lats[np.where(times==tmp_time)[0][0]],
                                                                           truth_lons[np.where(times==tmp_time)[0][0]],
                                                                           centroid_size=2*(lead_time//6),
                                                                           fcnv2=False)
            lats.append(pred_lat)
            lons.append(pred_lon)
            winds.append(max_wind)
            press.append(min_pres)
            err_km.append(haversine(pred_lat, pred_lon, [truth_lats[np.where(times==tmp_time)[0][0]]], [truth_lons[np.where(times==tmp_time)[0][0]]]))
            final_times.append(np.datetime64(tmp_time, 'h'))
            
        lats, lons, winds, press, err_km = np.array(lats), np.array(lons), np.array(winds), np.array(press), np.array(err_km)
        lons_plot = np.array([t-360 if t>180 else t for t in lons])
        truth_lons_plot = np.array([t-360 if t>180 else t for t in truth_lons])
        mean_err['era5'].append(err_km.mean())
        
        
        axs["Traj_ERA5"].set_extent([lon_extent[0], lon_extent[1], lat_extent[0], lat_extent[1]], crs=ccrs.PlateCarree(center_lon))
        axs["Traj_ERA5"].add_feature(cfeature.COASTLINE.with_scale('50m'))
        axs["Traj_ERA5"].add_feature(cfeature.BORDERS.with_scale('50m'))
        axs["Traj_ERA5"].plot(lons_plot, lats, '-x', c=colors_dic['era5'], label="ERA5", transform=ccrs.PlateCarree(), markersize=5)
        axs["Traj_ERA5"].plot(truth_lons_plot, truth_lats, '-x', c=colors_dic['truth'], label="IBTrACS (Truth)", transform=ccrs.PlateCarree(), markersize=5)
        axs["Traj_ERA5"].legend(fontsize=13)
        gridliner = axs["Traj_ERA5"].gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        gridliner.top_labels = False
        gridliner.bottom_labels = True
        gridliner.left_labels = False
        gridliner.right_labels = False
        gridliner.ylines = False
        gridliner.xlines = False
        axs["Traj_ERA5"].set_title(f"Model: ERA5", fontsize=18)
        axs["Traj_ERA5"].annotate(f"Trajectory comparison", xy=(0.5, 0.94), xycoords="figure fraction", ha="center", fontsize=20)
        
        
        axs["Wind"].plot(winds, c=colors_dic['era5'], label="ERA5")
        axs["Wind"].legend(fontsize=13)
        axs["Wind"].annotate(f"Evolution of wind speed", xy=(0.5, 0.71), xycoords="figure fraction", ha="center", fontsize=20)
        axs["Wind"].set_xticks(ticks=range(len(winds)) if len(winds) < 10 else range(len(winds))[::len(final_times)//10])
        
        
        axs["Pres"].plot(press, c=colors_dic['era5'], label="ERA5")
        axs["Pres"].legend(fontsize=13)
        axs["Pres"].annotate(f"Evolution of pressure", xy=(0.5, 0.485), xycoords="figure fraction", ha="center", fontsize=20)
        axs["Pres"].set_xticks(ticks=range(len(winds)) if len(winds) < 10 else range(len(winds))[::len(final_times)//10])
        
        
        axs["km_error"].plot(err_km, c=colors_dic['era5'], label="ERA5")
        axs["km_error"].set_xticks(ticks=range(len(winds)) if len(winds) < 10 else range(len(winds))[::len(final_times)//10], 
                                          labels=[str(x) for x in final_times] if len(winds) < 10 else [str(x) for x in final_times][::len(final_times)//10], 
                                          rotation=45, 
                                          horizontalalignment='right', fontsize='small')
        axs["km_error"].legend(fontsize=13)
        axs["km_error"].set_xlabel("Time", fontsize=15)
        axs["km_error"].annotate(f"Evolution of location error", xy=(0.5, 0.26), xycoords="figure fraction", ha="center", fontsize=20)
        
        
        pp_msg = f"Post-processing: {pp_type}" if pp_type is not None else "No post-processing"
        if pp_type=='linear' and pp_params.get("dim", 2)==2:
            pp_msg += f"{pp_params['dim']}d"
        pp_save = ""
        if pp_type is not None:
            pp_save += f"_{pp_type}_"
            if pp_type == 'linear':
                pp_save += f"{pp_params['dim']}d_"
            
            if pp_type!='cnn':    
                pp_save += f"b_{pp_params['basin']}_"
            pp_save += f"train_{'_'.join(sorted(pp_params['train_seasons']))}_"
            
            if pp_type == 'xgboost':
                pp_save += f"jsdiv_" if pp_params.get("jsdiv", False) else ""
                pp_save += f"lr_{pp_params['lr']}_"
                pp_save += f"d_{pp_params['depth']}_"
                pp_save += f"epochs_{pp_params['epochs']}_"
                pp_save += f"g_{pp_params['gamma']}_"
                pp_save += f"stats_{'_'.join(pp_params['stats'])}_"
                pp_save += f"wind_{'_'.join([s for s in pp_params['stats_wind'] if s not in pp_params['stats']])}_"
                pp_save += f"pres_{'_'.join([s for s in pp_params['stats_pres'] if s not in pp_params['stats']])}"
            
            if pp_type == 'cnn':
                pp_save += f"lr_{pp_params['lr']}_"
                pp_save += f"epochs_{pp_params['epochs']}_"
                pp_save += f"optim_{pp_params['optim']}"
                pp_save += f"_cosine_annealing" if pp_params.get("sched", False) else ""
                pp_save += f"_{'crps' if pp_params['crps'] else 'MSE'}_"
                pp_save += f"pres_{pp_params['pres']}"
            
            
        st = fig.suptitle(f"Trajectory comparison for {tc_sid_to_name[tc_id]} - Lead time: {lead_time}h\n{pp_msg}", fontsize=20)
        st.set_y(0.98)
        fig.savefig(plot_path + f"trajectory_{tc_id}_{lead_time}{pp_save}.png", dpi=500, bbox_inches="tight")
        plt.close(fig)
        
        
    fig2, axs2 = plt.subplot_mosaic([["mean_err"]], figsize=(10, 6))
    
    for key in mean_err.keys():
        axs2['mean_err'].plot(lead_times, mean_err[key], label=key if key in model_names+['era5'] else " + ".join(key.split('_')),
                                linestyle='--' if key not in model_names+['era5'] else '-', c=colors_dic[key.split('_')[0]])
    axs2['mean_err'].legend(fontsize=13)
    axs2['mean_err'].set_xlabel("Lead time (h)")
    axs2['mean_err'].set_ylabel("Mean error (km)")
    axs2['mean_err'].set_title(f"Mean error for {tc_sid_to_name[tc_id]}")
    
    fig.suptitle(f"Mean error for {tc_sid_to_name[tc_id]} at each lead time", fontsize=20)
    fig2.savefig(plot_path + f"mean_err_{tc_id}_{'_'.join(str(l) for l in lead_times)}{pp_save}.png", dpi=64, bbox_inches="tight")
        
        
        
        
        
def trajectory_with_pp(tc_id, model_names, max_lead_time=72, pp_type="linear", pp_params=None,
                     data_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/",
                     df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv",
                     plot_path="/users/lpoulain/louis/plots/case_studies/"):
    
    tc_sid_to_name = {"2005236N23285":"Katrina (2005)", "2000185N15117":"Kai-Tak (2000)", "2008278N13261": "Norbert (2008)"}
    assert tc_id in tc_sid_to_name.keys(), f"tc_id should be one of {list(tc_sid_to_name.keys())}"

    era5, valid_dates, df, pres_col = get_era5(tc_id, data_path+"ERA5/", df_path)
    idxs = [i for i in range(era5.valid_time.shape[0]) if era5.valid_time[i].values.astype("datetime64[ns]") in valid_dates.astype("datetime64[ns]")]
    era5 = era5.isel(step=idxs)
    
    lead_times = np.arange(6, max_lead_time+6, 6)
    lead_times = [48]
    
    time0 = df["ISO_TIME"].values[0]
    for lead_time in lead_times:
        print(f"Lead time: {lead_time}h")
        grid = ([f"Traj_{x}" for x in model_names+["ERA5"]], 
                [f"Wind_{x}" for x in model_names+["ERA5"]],
                [f"Pres_{x}" for x in model_names+["ERA5"]],
                [f"km_error_{x}" for x in model_names+["ERA5"]]
                )
        pp_params["ldt"] = lead_time
        
        df_tmp = df.loc[df.index[lead_time//6:]]
        times = df_tmp["ISO_TIME"].values.astype("datetime64[ns]")
        truth_lats, truth_lons = df_tmp["LAT"].values.astype(np.float32), df_tmp["LON"].values.astype(np.float32)
        truth_lons = [t+360 if t<0 else t for t in truth_lons]
        
        truth_wind, truth_pres = df_tmp["USA_WIND"].values.astype(np.float32)*0.514444, df_tmp[pres_col].values.astype(np.float32)*100
        
        lat_extent = [min(truth_lats), max(truth_lats)]
        lon_extent = [min(truth_lons), max(truth_lons)]
        lon_extent = [t-360 if t>180 else t for t in lon_extent]
        
        center = [(lat_extent[0]+lat_extent[1])/2, (lon_extent[0]+lon_extent[1])/2]
        radius = [(lat_extent[1]-lat_extent[0])/2, (lon_extent[1]-lon_extent[0])]
        lat_extent = [center[0]-max(radius)-4, center[0]+max(radius)+4]
        lon_extent = [center[1]-max(radius)-4, center[1]+max(radius)+4]
        center_lon = 0
        
        per_subplot_kw = {"Traj_"+model: dict(projection=ccrs.PlateCarree(center_lon)) for model in model_names+["ERA5"]}
        fig, axs = plt.subplot_mosaic(grid, figsize=((len(model_names)+1)*5, len(grid)*6), per_subplot_kw=per_subplot_kw,
                                      gridspec_kw={"hspace":0.1, "wspace":0.1, "top":0.93, "bottom":0.05, "left":0.05, "right":0.95},
                                      sharey=False, sharex=False)
        pres_extent = [truth_pres.min(), truth_pres.max()]
        wind_extent = [truth_wind.min(), truth_wind.max()]
        err_km_extent = [1e6, 0]
        
        for model in model_names:
            data = load_tc_forecast(model, tc_id, pp_type, pp_params, lead_time)
            pp_models = load_pp_model(model, pp_type, pp_params)
            
            lats, lons, winds, press, err_km = [], [], [], [], []
            final_times = []
            
            for i in range(len(data)):
                tmp_time = data[i][0]

                if tmp_time in times:
                    input_wind, input_pres, pred_lat, pred_lon, truth_csts = data[i][1:]
                    
                    if pp_type=="linear" and pp_params.get("dim", 2)==2:
                        max_wind, min_pres = pp_models[0].predict(np.array([input_pres]).reshape(-1, 1))
                    else:
                        if pp_type == "linear":
                            max_wind = pp_models[0].predict(np.array([input_wind, input_pres]).reshape(1, -1))
                            min_pres = pp_models[1].predict(np.array([input_wind, input_pres]).reshape(1, -1))
                        if pp_type == "xgboost":
                            max_wind = pp_models[0].predict(input_wind, iteration_range=(0, pp_models[0].best_iteration + 1))*truth_csts[1] + truth_csts[0]
                            min_pres = pp_models[1].predict(input_pres, iteration_range=(0, pp_models[1].best_iteration + 1))*truth_csts[3] + truth_csts[2]
                        if pp_type == "cnn":
                            #max_wind, min_pres, pred_lat, pred_lon = pp_models[0](torch.tensor([input_wind, input_pres]).reshape(1, 1, 2))
                            raise NotImplementedError("CNN post-processing plot not implemented yet")
                        
                    lats.append(pred_lat)
                    lons.append(pred_lon)
                    winds.append(max_wind)
                    press.append(min_pres)
                    err_km.append(haversine(pred_lat, pred_lon, [truth_lats[np.where(times==tmp_time)[0][0]]], [truth_lons[np.where(times==tmp_time)[0][0]]]))
                    final_times.append(np.datetime64(tmp_time, 'h'))
            
            lats, lons, winds, press, err_km = np.array(lats), np.array(lons), np.array(winds), np.array(press), np.array(err_km)
            lons_plot = np.array([t-360 if t>180 else t for t in lons])
            truth_lons_plot = np.array([t-360 if t>180 else t for t in truth_lons])
            

            axs["Traj_"+model].set_extent([lon_extent[0], lon_extent[1], lat_extent[0], lat_extent[1]], crs=ccrs.PlateCarree(center_lon))
            axs["Traj_"+model].plot(lons_plot, lats, linestyle='-', marker='x', c = 'blue', label=model, transform=ccrs.PlateCarree(), markersize=5)
            axs["Traj_"+model].plot(truth_lons_plot, truth_lats, '-x', c = 'red', label="IBTrACS (Truth)", transform=ccrs.PlateCarree(), markersize=5)
            axs["Traj_"+model].add_feature(cfeature.COASTLINE.with_scale('50m'))
            axs["Traj_"+model].add_feature(cfeature.BORDERS.with_scale('50m'))
            axs["Traj_"+model].legend()
            gridliner = axs["Traj_"+model].gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
            gridliner.top_labels = False
            gridliner.bottom_labels = True
            gridliner.left_labels = True if model==model_names[0] else False
            gridliner.right_labels = False
            gridliner.ylines = False
            gridliner.xlines = False
            axs["Traj_"+model].set_title(f"Model: {model}", fontsize=18)
            
            
            axs["Wind"].plot(winds, c = 'blue', label=model)
            axs["Wind"].plot(truth_wind, c = 'red', label="IBTrACS (Truth)")
            axs["Wind"].legend()
            if model==model_names[0]:
                axs["Wind"].set_ylabel("Wind speed (m/s)")
            if not model==model_names[0]:
                axs["Wind"].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            axs["Wind"].tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
            if winds.min()<wind_extent[0]:
                wind_extent[0] = winds.min()
            if winds.max()>wind_extent[1]:
                wind_extent[1] = winds.max()
                
            axs["Pres_"+model].plot(press, c = 'blue', label=model)
            axs["Pres_"+model].plot(truth_pres, c = 'red', label="IBTrACS (Truth)")
            axs["Pres_"+model].legend()
            if model==model_names[0]:
                axs["Pres_"+model].set_ylabel("Pressure (Pa)")
            if not model==model_names[0]:
                axs["Pres_"+model].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            axs["Pres_"+model].tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
            if press.min()<pres_extent[0]:
                pres_extent[0] = press.min()
            if press.max()>pres_extent[1]:
                pres_extent[1] = press.max()
            
            
            axs["km_error_"+model].plot(err_km, c = 'blue', label=model)
            axs["km_error_"+model].set_xticks(ticks=range(len(winds))[::len(final_times)//7], 
                                          labels=[str(x) for x in final_times][::len(final_times)//7], 
                                          rotation=45, 
                                          horizontalalignment='right', fontsize='small')
            axs["km_error_"+model].legend()
            axs["km_error_"+model].set_xlabel("Time")
            if model==model_names[0]:
                axs["km_error_"+model].set_ylabel("Error (km)")
            if not model==model_names[0]:
                axs["km_error_"+model].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            if err_km.min()<err_km_extent[0]:
                err_km_extent[0] = err_km.min()
            if err_km.max()>err_km_extent[1]:
                err_km_extent[1] = err_km.max()
            
        lats, lons, winds, press, err_km = [], [], [], [], []
        final_times = []
        for i, tmp_time in enumerate(times):
            idx = np.where(era5.valid_time.values==tmp_time)[0][0]

            era5_tmp = era5.isel(step=idx)

            era5_tmp = cut_rectangle(era5_tmp, df, tc_id, tmp_time)
            max_wind, min_pres, pred_lat, pred_lon = find_trajectory_point(era5_tmp,
                                                                           truth_lats[np.where(times==tmp_time)[0][0]],
                                                                           truth_lons[np.where(times==tmp_time)[0][0]],
                                                                           centroid_size=5*(lead_time//6))
            lats.append(pred_lat)
            lons.append(pred_lon)
            winds.append(max_wind)
            press.append(min_pres)
            err_km.append(haversine(pred_lat, pred_lon, [truth_lats[np.where(times==tmp_time)[0][0]]], [truth_lons[np.where(times==tmp_time)[0][0]]]))
            final_times.append(np.datetime64(tmp_time, 'h'))
            
        lats, lons, winds, press, err_km = np.array(lats), np.array(lons), np.array(winds), np.array(press), np.array(err_km)
        lons_plot = np.array([t-360 if t>180 else t for t in lons])
        truth_lons_plot = np.array([t-360 if t>180 else t for t in truth_lons])
        
        
        axs["Traj_ERA5"].set_extent([lon_extent[0], lon_extent[1], lat_extent[0], lat_extent[1]], crs=ccrs.PlateCarree(center_lon))
        axs["Traj_ERA5"].add_feature(cfeature.COASTLINE.with_scale('50m'))
        axs["Traj_ERA5"].add_feature(cfeature.BORDERS.with_scale('50m'))
        axs["Traj_ERA5"].plot(lons_plot, lats, '-x', c = 'blue', label="ERA5", transform=ccrs.PlateCarree(), markersize=5)
        axs["Traj_ERA5"].plot(truth_lons_plot, truth_lats, '-x', c = 'red', label="IBTrACS (Truth)", transform=ccrs.PlateCarree(), markersize=5)
        axs["Traj_ERA5"].legend()
        gridliner = axs["Traj_ERA5"].gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        gridliner.top_labels = False
        gridliner.bottom_labels = True
        gridliner.left_labels = False
        gridliner.right_labels = False
        gridliner.ylines = False
        gridliner.xlines = False
        axs["Traj_ERA5"].set_title(f"Model: ERA5", fontsize=18)
        axs["Traj_ERA5"].annotate(f"Trajectory comparison", xy=(0.5, 0.94), xycoords="figure fraction", ha="center", fontsize=20)
        
        
        axs["Wind_ERA5"].plot(winds, c = 'blue', label="ERA5")
        axs["Wind_ERA5"].plot(truth_wind, c = 'red', label="IBTrACS (Truth)")
        axs["Wind_ERA5"].legend()
        axs["Wind_ERA5"].annotate(f"Evolution of wind speed", xy=(0.5, 0.705), xycoords="figure fraction", ha="center", fontsize=20)
        axs["Wind_ERA5"].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        axs["Wind_ERA5"].tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
        if winds.min()<wind_extent[0]:
            wind_extent[0] = winds.min()
        if winds.max()>wind_extent[1]:
            wind_extent[1] = winds.max()
        
        
        axs["Pres_ERA5"].plot(press, c = 'blue', label="ERA5")
        axs["Pres_ERA5"].plot(truth_pres, c = 'red', label="IBTrACS (Truth)")
        axs["Pres_ERA5"].legend()
        axs["Pres_ERA5"].annotate(f"Evolution of pressure", xy=(0.5, 0.48), xycoords="figure fraction", ha="center", fontsize=20)
        axs["Pres_ERA5"].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        axs["Pres_ERA5"].tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
        if press.min()<pres_extent[0]:
            pres_extent[0] = press.min()
        if press.max()>pres_extent[1]:
            pres_extent[1] = press.max()
        
        
        axs["km_error_ERA5"].plot(err_km, c = 'blue', label="ERA5")
        axs["km_error_ERA5"].set_xticks(ticks=range(len(winds))[::len(final_times)//7], 
                                          labels=[str(x) for x in final_times][::len(final_times)//7], 
                                          rotation=45, 
                                          horizontalalignment='right', fontsize='small')
        axs["km_error_ERA5"].legend()
        axs["km_error_ERA5"].set_xlabel("Time")
        axs["km_error_ERA5"].annotate(f"Evolution of location error", xy=(0.5, 0.255), xycoords="figure fraction", ha="center", fontsize=20)
        if err_km.min()<err_km_extent[0]:
            err_km_extent[0] = err_km.min()
        if err_km.max()>err_km_extent[1]:
            err_km_extent[1] = err_km.max()
        
        
        for m in model_names+["ERA5"]:
            axs["Wind_"+m].set_ylim(wind_extent[0]-1, wind_extent[1]+1)
            axs["Pres_"+m].set_ylim(pres_extent[1]+1000, pres_extent[0]-1000)
            axs["km_error_"+m].set_ylim(err_km_extent[0]-10, err_km_extent[1]+10)
            
        st = fig.suptitle(f"Trajectory comparison for {tc_sid_to_name[tc_id]} - Lead time: {lead_time}h\nPost-processing using {pp_type}")
        st.set_y(0.98)
        fig.savefig(plot_path + f"trajectory_{pp_type}_{tc_id}_{lead_time}.png", dpi=500, bbox_inches="tight")
        
       
if __name__ == "__main__":
    # 2000185N15117 - Kai-Tak (2000)
    # 2005236N23285 - Katrina (2005)
    tc_id = "2005236N23285"
    model_names = ["pangu", "graphcast"]
    max_lead_time = 72
    trajectory(tc_id, model_names, max_lead_time)