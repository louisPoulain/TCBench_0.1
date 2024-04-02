import pickle
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.linear_model import LinearRegression
from loading_utils import get_data, data_loader, statistics_loader
from utils.main_utils import global_wind_bins, global_pres_bins
from joblib import dump
from matplotlib.colors import LogNorm


def normalize_data(data, mean, std):
    return (data - mean) / std


def linear_model_test(data_path, model_name, lead_time, tc_id, 
                 df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv",
                 save_path = "/users/lpoulain/louis/plots/linear_model/"):
    
    data_forecast, data_ibtracs = get_data(data_path, model_name, lead_time, tc_id, df_path)
    if not len(data_forecast)==0:
        wind_col, pres_col = data_ibtracs.columns[1:]
        
        wind = data_ibtracs[wind_col].values.astype("float") * 0.514444 # knots to m/s
        pres = data_ibtracs[pres_col].values.astype("float") * 100 # mb to Pa
        
        wind_forecast = np.array([np.sqrt(data.u10.values.astype("float")**2 + data.v10.values.astype("float")**2).max().max()\
                                    for data in data_forecast]).reshape(-1, 1)

        pres_forecast = np.array([data.msl.values.astype("float").min().min() for data in data_forecast]).reshape(-1, 1)
        
        model_wind, model_pres = LinearRegression(), LinearRegression()
        estimator_wind = model_wind.fit(wind_forecast, wind)
        estimator_pres = model_pres.fit(pres_forecast, pres)
        
        print(f"Model: {model_name} at ldt {lead_time} - TC {tc_id}\n",
            f"R2 determination (wind): {estimator_wind.score(wind_forecast, wind)}\n",
            f"Coeffs (wind): {estimator_wind.coef_}, Intercept (wind): {estimator_wind.intercept_}\n",
            f"R2 determination (pres): {estimator_pres.score(pres_forecast, pres)}\n",
            f"Coeffs (pres): {estimator_pres.coef_}, Intercept (pres): {estimator_pres.intercept_}"
            )
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].scatter(wind_forecast, wind, label="Wind")
        axs[0].plot(wind_forecast, estimator_wind.predict(wind_forecast), color="red", label="Linear model")
        axs[0].plot(wind, wind, color="black", label="Perfect model")
        axs[0].set_xlabel("Wind forecast (m/s)")
        axs[0].set_ylabel("Wind observed (m/s)")
        axs[0].set_title(f"Model: {model_name} at ldt {lead_time} - TC {tc_id}")
        axs[0].legend()
        
        axs[1].scatter(pres_forecast, pres, label="Pressure")
        axs[1].plot(pres_forecast, estimator_pres.predict(pres_forecast), color="red", label="Linear model")
        axs[1].plot(pres, pres, color="black", label="Perfect model")
        axs[1].set_xlabel("Pressure forecast (Pa)")
        axs[1].set_ylabel("Pressure observed (Pa)")
        axs[1].set_title(f"Model: {model_name} at ldt {lead_time} - TC {tc_id}")
        axs[1].legend()
        
        fig.savefig(save_path + f"linear_model_{model_name}_{lead_time}_{tc_id}.png")
    else:
        print(f"No data available for TC {tc_id} at lead time {lead_time} (model: {model_name})")
        

def linear_model2d(data_path, model_name, lead_time, season:int, basin='all',
                   df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv",
                   save_path = "/users/lpoulain/louis/plots/linear_model/"):
    
    forecast_wind_list, forecast_pres_list, truth_wind_list, truth_pres_list, tc_ids = statistics_loader(data_path=data_path, model_name=model_name, 
                                                                                                         lead_time=lead_time, season=season, basin=basin, 
                                                                                                         df_path=df_path, save_path=save_path)
    
    #data_loader(data_path=data_path, model_name=model_name, lead_time=lead_time, season=season, basin=basin, df_path=df_path, save_path=save_path)
    
    forecast_wind_list = forecast_wind_list['max']
    forecast_pres_list = forecast_pres_list['min']
    
    forecast_wind = np.concatenate(forecast_wind_list).reshape(-1, 1)
    forecast_pres = np.concatenate(forecast_pres_list).reshape(-1, 1)
    truth_wind = np.concatenate(truth_wind_list).reshape(-1, 1)
    truth_pres = np.concatenate(truth_pres_list).reshape(-1, 1)
    
    
    forecast = np.concatenate([forecast_wind, forecast_pres], axis=1)
    truth = np.concatenate([truth_wind, truth_pres], axis=1)
    
    # Fit and predict
    model = LinearRegression()
    estimator = model.fit(forecast, truth)
    
    save_name = f"linear_model2d_{model_name}_{lead_time}_{season}_b_{basin}.joblib"
    dump(estimator, save_path + "Models/" + save_name)
    
    print(f"Model: {model_name} at ldt {lead_time}\n",
        f"Overall R2 determination: {estimator.score(forecast, truth)}")
    
    results = {}
    for i, tc_id in enumerate(tc_ids):
        input_data = np.concatenate([forecast_wind_list[i].reshape(-1,1), forecast_pres_list[i].reshape(-1,1)], axis=1)
        truth_data = np.concatenate([truth_wind_list[i].reshape(-1, 1), truth_pres_list[i].reshape(-1, 1)], axis=1)
        
        results[tc_id] = estimator.score(input_data, truth_data)
    
    results = {w:results[w] for w in sorted(results, key=results.get, reverse=True)}
    
    with open(save_path + "Scores/" + f"linear_model2d_{model_name}_{lead_time}_{season}_b_{basin}_scores.pkl", "wb") as f:
        pickle.dump(results, f)
        
    fig, axs = plt.subplot_mosaic([['a)', 'b)'],
                                   ['c)', 'd)'],
                                   ], figsize=(10,8), gridspec_kw={'hspace': 0.4, 'wspace': 0.3})
    
    ax0, ax1, ax2, ax3 = axs.values()
    
    # def bins for scatter
    wind_bins = global_wind_bins
    pres_bins = global_pres_bins
    
    # create histograms
    
    hist_wind, e_x_wind, e_y_wind = np.histogram2d(estimator.predict(forecast)[:, 0], truth[:, 0],
                                                   bins=[wind_bins, wind_bins], density=True)
    x_wind, y_wind = np.meshgrid(e_x_wind, e_y_wind)
    #hist_wind = np.ma.masked_array(hist_wind, hist_wind == 0)
    
    hist_pres, e_x_pres, e_y_pres = np.histogram2d(estimator.predict(forecast)[:, 1], truth[:, 1],
                                                   bins=[pres_bins, pres_bins], density=True)
    x_pres, y_pres = np.meshgrid(e_x_pres, e_y_pres)
    
    wnd = ax0.pcolormesh(x_wind, y_wind, hist_wind.T, cmap='plasma', norm=LogNorm())
    cbar = plt.colorbar(wnd, label='Density')
    wind_extent = [min(truth[:, 0].min(), estimator.predict(forecast)[:, 0].min()),
                   max(truth[:, 0].max(), estimator.predict(forecast)[:, 0].max())]
    ax0.plot(wind_extent, wind_extent, color="black", label="Identity")
    ax0.set_xlim(wind_extent)
    ax0.set_ylim(wind_extent)
    ax0.set_xlabel("Post-processed wind (m/s)")
    ax0.set_ylabel("Observed wind (m/s)")
    ax0.set_title(f"Model: {model_name} at ldt {lead_time} - Wind density histogram")
    
    
    prs = ax1.pcolormesh(x_pres, y_pres, hist_pres.T, cmap='plasma', norm=LogNorm())
    cbar = plt.colorbar(prs, label='Density')
    pres_extent = [min(truth[:, 1].min(), estimator.predict(forecast)[:, 1].min()),
                   max(truth[:, 1].max(), estimator.predict(forecast)[:, 1].max())]
    ax1.plot(pres_extent, pres_extent, color="black", label="Identity")
    ax1.set_xlim(pres_extent)
    ax1.set_ylim(pres_extent)
    ticks = [t for t in ax1.get_xticks() if t>=ax1.get_xlim()[0] and t<=ax1.get_xlim()[1]]
    ax1.set_xticks(ticks, ticks, ha='right', rotation=15)
    ax1.set_xlabel("Post-processed pressure (Pa) ")
    ax1.set_ylabel("Observed pressure (Pa)")
    ax1.set_title(f"Model: {model_name} at ldt {lead_time} - Pressure density histogram")
    
    
    ax2.hist(truth_wind, bins=wind_bins, label="Observed wind distribution", alpha=0.25, color='blue')
    ax2.hist(estimator.predict(forecast)[:, 0], bins=wind_bins, label=f"Post-processed wind distribution", alpha=0.7, color='red')
    ax2.hist(forecast[:, 0], bins=wind_bins, label="Original wind distribution", alpha=0.5, color='green')
    ax2.set_title("Wind distributions")
    ax2.set_xlabel("Wind (m/s)")
    ax2.set_ylabel("Number of occurences")
    ax2.legend()
    
    ax3.hist(truth_pres, bins=pres_bins, label="Observed pressure distribution", alpha=0.25, color='blue')
    ax3.hist(estimator.predict(forecast)[:, 1], bins=pres_bins, label=f"Post-processed pressure distribution", alpha=0.7, color='red')
    ax3.hist(forecast[:, 1], bins=pres_bins, label="Original pressure distribution", alpha=0.5, color='green')
    ax3.set_title("Pressure distributions")
    ticks = [t for t in ax3.get_xticks() if t>=ax3.get_xlim()[0] and t<=ax3.get_xlim()[1]]
    ax3.set_xticks(ticks, ticks, ha='right', rotation=45)
    ax3.set_xlabel("Pressure (Pa)")
    ax3.set_ylabel("Number of occurences")
    ax3.legend()
    
    fig.suptitle(f"2D linear model | {model_name} - Season {season} - Basin {basin}")
    
    fig.savefig(save_path + f"Figs/linear_model2d_{model_name}_{lead_time}_{season}_b_{basin}.png", dpi=500)
    
        
    
def linear_model(data_path, model_name, lead_time, season:int, basin='all',
                 df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv",
                 save_path = "/users/lpoulain/louis/plots/linear_model/"):
    
    forecast_wind_list, forecast_pres_list, truth_wind_list, truth_pres_list, tc_ids = statistics_loader(data_path=data_path, model_name=model_name, 
                                                                                                         lead_time=lead_time, season=season, basin=basin, 
                                                                                                         df_path=df_path, save_path=save_path)
    
    #data_loader(data_path=data_path, model_name=model_name, lead_time=lead_time, season=season, basin=basin, df_path=df_path, save_path=save_path)
    
    forecast_wind_list = forecast_wind_list['max']
    forecast_pres_list = forecast_pres_list['min']
        
    forecast_wind = np.concatenate(forecast_wind_list).reshape(-1,1)
    forecast_pres = np.concatenate(forecast_pres_list).reshape(-1,1)
    truth_wind = np.concatenate(truth_wind_list).reshape(-1,1)
    truth_pres = np.concatenate(truth_pres_list).reshape(-1,1)
    
    # Fit and predict
    model_wind, model_pres = LinearRegression(), LinearRegression()
    estimator_wind = model_wind.fit(forecast_wind, truth_wind)
    estimator_pres = model_pres.fit(forecast_pres, truth_pres)
    
    save_name_wind = f"linear_model_wind_{model_name}_{lead_time}_{season}_b_{basin}.joblib"
    save_name_pres = f"linear_model_pres_{model_name}_{lead_time}_{season}_b_{basin}.joblib"
    dump(estimator_wind, save_path + "Models/" + save_name_wind)
    dump(estimator_pres, save_path + "Models/" + save_name_pres)
    
    results_wind = {}
    results_pres = {}
    for i, tc_id in enumerate(tc_ids):
        results_wind[tc_id] = estimator_wind.score(forecast_wind_list[i].reshape(-1,1), truth_wind_list[i].reshape(-1,1))
        results_pres[tc_id] = estimator_pres.score(forecast_pres_list[i].reshape(-1,1), truth_pres_list[i].reshape(-1,1))
    
    results_wind = {w:results_wind[w] for w in sorted(results_wind, key=results_wind.get, reverse=True)}
    results_pres = {p:results_pres[p] for p in sorted(results_pres, key=results_pres.get, reverse=True)}
    
    fig, axs = plt.subplot_mosaic([['a)', 'b)'],
                                   ['c)', 'd)'],
                                   ], figsize=(10,8))
    
    ax0, ax1, ax2, ax3 = axs.values()
    
    # def bins for scatter
    wind_bins = global_wind_bins
    pres_bins = global_pres_bins
    
    # create histograms
    
    hist_wind, e_x_wind, e_y_wind = np.histogram2d(estimator_wind.predict(forecast_wind).reshape(-1), truth_wind.reshape(-1),
                                                   bins=[wind_bins, wind_bins], density=True)
    x_wind, y_wind = np.meshgrid(e_x_wind, e_y_wind)
    #hist_wind = np.ma.masked_array(hist_wind, hist_wind == 0)
    
    hist_pres, e_x_pres, e_y_pres = np.histogram2d(estimator_pres.predict(forecast_pres).reshape(-1), truth_pres.reshape(-1),
                                                   bins=[pres_bins, pres_bins], density=True)
    x_pres, y_pres = np.meshgrid(e_x_pres, e_y_pres)
    
    wnd = ax0.pcolormesh(x_wind, y_wind, hist_wind.T, cmap='plasma', norm=LogNorm())
    cbar = plt.colorbar(wnd, label='Density')
    wind_extent = [min(truth_wind.min(), estimator_wind.predict(forecast_wind).min()),
                   max(truth_wind.max(), estimator_wind.predict(forecast_wind).max())]
    ax0.plot(wind_extent, wind_extent, color="black", label="Identity")
    ax0.set_xlim(wind_extent)
    ax0.set_ylim(wind_extent)
    ax0.set_xlabel("Post-processed wind (m/s)")
    ax0.set_ylabel("Observed wind (m/s)")
    ax0.set_title(f"Model: {model_name} at ldt {lead_time} - Wind density histogram")
    
    
    prs = ax1.pcolormesh(x_pres, y_pres, hist_pres.T, cmap='plasma', norm=LogNorm())
    cbar = plt.colorbar(prs, label='Density')
    pres_extent = [min(truth_pres.min(), estimator_pres.predict(forecast_pres).min()),
                   max(truth_pres.max(), estimator_pres.predict(forecast_pres).max())]
    ax1.plot(pres_extent, pres_extent, color="black", label="Identity")
    ax1.set_xlim(pres_extent)
    ax1.set_ylim(pres_extent)
    ax1.set_xlabel("Post-processed pressure (Pa) ")
    ax1.set_ylabel("Observed pressure (Pa)")
    ax1.set_title(f"Model: {model_name} at ldt {lead_time} - Pressure density histogram")
    
    
    ax2.hist(truth_wind, bins=wind_bins, label="Observed wind distribution", alpha=0.25, color='blue')
    ax2.hist(estimator_wind.predict(forecast_wind), bins=wind_bins, label=f"Post-processed wind distribution", alpha=0.7, color='red')
    ax2.hist(forecast_wind, bins=wind_bins, label="Original wind distribution", alpha=0.5, color='green')
    ax2.set_title("Wind distributions")
    ax2.set_xlabel("Wind (m/s)")
    ax2.set_ylabel("Number of occurences")
    ax2.legend()
    
    
    ax3.hist(truth_pres, bins=pres_bins, label="Observed pressure distribution", alpha=0.25, color='blue')
    ax3.hist(estimator_pres.predict(forecast_pres), bins=pres_bins, label=f"Post-processed pressure distribution", alpha=0.7, color='red')
    ax3.hist(forecast_pres, bins=pres_bins, label="Original pressure distribution", alpha=0.5, color='green')
    ax3.set_title("Pressure distributions")
    ax3.set_xlabel("Pressure (Pa)")
    ticks = [t for t in ax3.get_xticks() if t>=ax3.get_xlim()[0] and t<=ax3.get_xlim()[1]]
    ax3.set_xticks(ticks, ticks, ha='right', rotation=45)
    ax3.set_ylabel("Number of occurences")
    ax3.legend()
    
    
    fig.suptitle(f"Linear model | {model_name} - Season {season} - Basin {basin}")
    fig.tight_layout()
    
    fig.savefig(save_path + f"Figs/linear_model_{model_name}_{lead_time}_{season}_b_{basin}.png", dpi=500)
    
    res_dict = {"R2 global (wind)": estimator_wind.score(forecast_wind, truth_wind),
                "R2 global (pressure)": estimator_pres.score(forecast_pres, truth_pres),
                "Wind coeffs": estimator_wind.coef_,
                "Pressure coeffs": estimator_pres.coef_,
                "Wind scores": results_wind,
                "Pressure scores": results_pres}
    with open(save_path + "Scores/" + f"linear_model_{model_name}_{lead_time}_{season}_b_{basin}_scores.pkl", "wb") as f:
        pickle.dump(res_dict, f)



def isolate_linearity_per_tc(data_path, model_name, lead_time, season:int, normalize=True,
                 df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv",
                 save_path = "/users/lpoulain/louis/plots/linear_model/"):
    
    forecast_wind_list, forecast_pres_list, truth_wind_list, truth_pres_list, tc_ids = data_loader(data_path, model_name, lead_time,
                                                                                                    season, df_path, save_path)
    norma = "_norma" if normalize else ""
    if normalize:
        wind_mean_pred, wind_std_pred = np.mean(np.concatenate(forecast_wind_list)), np.std(np.concatenate(forecast_wind_list))
        pres_mean_pred, pres_std_pred = np.mean(np.concatenate(forecast_pres_list)), np.std(np.concatenate(forecast_pres_list))
        wind_mean_truth, wind_std_truth = np.mean(np.concatenate(truth_wind_list)), np.std(np.concatenate(truth_wind_list))
        pres_mean_truth, pres_std_truth = np.mean(np.concatenate(truth_pres_list)), np.std(np.concatenate(truth_pres_list))
        
        forecast_wind_list = [normalize_data(forecast_wind, wind_mean_pred, wind_std_pred) for forecast_wind in forecast_wind_list]
        forecast_pres_list = [normalize_data(forecast_pres, pres_mean_pred, pres_std_pred) for forecast_pres in forecast_pres_list]
        truth_wind_list = [normalize_data(truth_wind, wind_mean_truth, wind_std_truth) for truth_wind in truth_wind_list]
        truth_pres_list = [normalize_data(truth_pres, pres_mean_truth, pres_std_truth) for truth_pres in truth_pres_list]
        
    
    results_wind = {}
    results_pres = {}        
    for i, tc_id in enumerate(tc_ids):
        model_wind, model_pres = LinearRegression(), LinearRegression()
        input_wind = forecast_wind_list[i]
        input_pres = forecast_pres_list[i]
        
        target_wind = truth_wind_list[i]
        target_pres = truth_pres_list[i]
        
        estimator_wind = model_wind.fit(input_wind, target_wind)
        estimator_pres = model_pres.fit(input_pres, target_pres)
        
        results_wind[tc_id] = estimator_wind.score(input_wind, target_wind)
        results_pres[tc_id] = estimator_pres.score(input_pres, target_pres)
        
    results_wind = {w:results_wind[w] for w in sorted(results_wind, key=results_wind.get, reverse=True)}
    results_pres = {p:results_pres[p] for p in sorted(results_pres, key=results_pres.get, reverse=True)}
    
    res_dict = {"Wind scores": results_wind,
                "Pressure scores": results_pres}
    with open(save_path + "Scores/" + f"linear_model_{model_name}_{lead_time}_{season}_scores_individual{norma}.pkl", "wb") as f:
        pickle.dump(res_dict, f)
        

if __name__ == "__main__":
    
    fct_dict = {"linear_model2d":linear_model2d,
                "linear_model":linear_model,
                "isolate_linearity_per_tc":isolate_linearity_per_tc}
    
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/")
    parser.add_argument("--df_path", type=str, 
                        default="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv")
    parser.add_argument("--save_path", type=str, default="/users/lpoulain/louis/plots/linear_model/")
    
    parser.add_argument("--lead_time", type=int, default=6)
    parser.add_argument("--model", type=str, default="graphcast")
    parser.add_argument("--season", type=int, default=2000)
    parser.add_argument("--basin", type=str, default="all", choices=["all", "NA", "WP", "EP", "NI", "SI", "SP", "SA"])
    parser.add_argument("--tc_id", type=str, default=None)
    
    parser.add_argument('-f', '--func', dest='func', choices=fct_dict.keys(), required=True,
                            help="Choose one of the specified function to be run.")
    
    args = parser.parse_args()
    print("Input args:\n", args)
    
    
    data_path, df_path, save_path = args.data_path, args.df_path, args.save_path
    lead_time, model, season, tc_id = args.lead_time, args.model, args.season, args.tc_id
    basin = args.basin
    fct = fct_dict[args.func]
    
    if tc_id is not None:
        fct(data_path=data_path, model_name=model, lead_time=lead_time, tc_id=tc_id, df_path=df_path, save_path=save_path)
    else:
        fct(data_path=data_path, model_name=model, lead_time=lead_time, season=season, basin=basin, 
            df_path=df_path, save_path=save_path)
    
        