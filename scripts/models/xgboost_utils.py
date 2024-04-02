import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from loading_utils import statistics_loader, stats_list
from matplotlib.colors import LogNorm

from utils.main_utils import multiline_label, global_wind_bins, global_pres_bins
import sys, random, os



def create_dataset_new(data_path, model_name, lead_time, seasons, force_in_train, basin, split_ratio:list,
            stats=[], stats_wind=["max"], stats_pres=["min"], jsdiv=False,
            df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv",
            save_path="/users/lpoulain/louis/plots/xgboost/"
            ):
    
    if stats==[''] or stats==[' ']:
        stats = []
    assert round(sum(split_ratio),5)==1.0, f"split_ratio must sum to 1, your sum is {'+'.join(str(ratio) for ratio in split_ratio)} ="\
                                                + f"{round(sum(split_ratio),5)}"
    assert set(stats).union(stats_wind).union(stats_pres).issubset(set(stats_list)), f"stats/stats_wind/stats_pres must be a subset of {stats_list}.\n"\
                    + f"Your stats are {stats}, stats_wind are {stats_wind} and stats_pres are {stats_pres}"
    
    if type(seasons)==str or type(seasons)==int:
        seasons = [seasons]
    seasons = sorted(seasons)
    
    # sort stats to always have the same order
    stats_wind.extend(stats)
    stats_pres.extend(stats)
    stats_wind = sorted(list(set(stats_wind)), key=lambda x: stats_list.index(x))
    stats_pres = sorted(list(set(stats_pres)), key=lambda x: stats_list.index(x))
    
    test_seasons = seasons[-int(np.ceil(split_ratio[2]*len(seasons))):]
    assert set(test_seasons).intersection(force_in_train)==set(), f"Test seasons and force_in_train must be disjoint,"\
                                                        f" but they intersect at {set(test_seasons).intersection(force_in_train)}"
    nb_test_seasons = len(test_seasons)
    nb_val_seasons = int(np.ceil(split_ratio[1]*len(seasons)))
    
    train_seasons = [s for s in force_in_train] + random.sample(seasons, len(seasons)-nb_val_seasons-nb_test_seasons-len(force_in_train))
    val_seasons = [s for s in seasons if s not in train_seasons and s not in test_seasons]
    
    # datasets for xgb
    
    stats_wind_train, stats_wind_val, stats_wind_test,\
    stats_pres_train, stats_pres_val, stats_pres_test,\
    truth_wind_train, truth_wind_val, truth_wind_test,\
    truth_pres_train, truth_pres_val, truth_pres_test, nb_tc = split_data_new(model_name, lead_time, basin, train_seasons, 
                                                                            val_seasons, test_seasons, stats_wind, stats_pres, 
                                                                            data_path, df_path, save_path)
    
    print(f"Train seasons: {train_seasons} ({nb_tc['train']} TCs)"\
        + f"\nVal seasons: {val_seasons} ({nb_tc['val']} TCs)"\
        + f"\nTest seasons: {test_seasons} ({nb_tc['test']} TCs)")
    
    # transform into np.ndarrays
    
    stats_wind_train = {key:np.concatenate(val).ravel().reshape(-1, 1) for key, val in stats_wind_train.items()}
    stats_wind_val = {key:np.concatenate(val).ravel().reshape(-1, 1) for key, val in stats_wind_val.items()}
    stats_wind_test = {key:np.concatenate(val).ravel().reshape(-1, 1) for key, val in stats_wind_test.items()}
    
    stats_pres_train = {key:np.concatenate(val).ravel().reshape(-1, 1) for key, val in stats_pres_train.items()}
    stats_pres_val = {key:np.concatenate(val).ravel().reshape(-1, 1) for key, val in stats_pres_val.items()}
    stats_pres_test = {key:np.concatenate(val).ravel().reshape(-1, 1) for key, val in stats_pres_test.items()}
    
    truth_wind_train = np.concatenate(truth_wind_train)
    truth_wind_val = np.concatenate(truth_wind_val)
    truth_wind_test = np.concatenate(truth_wind_test)
    
    stats_wind_train_np = np.concatenate([stats_wind_train[stat] for stat in stats_wind], axis=1)
    stats_wind_val_np = np.concatenate([stats_wind_val[stat] for stat in stats_wind], axis=1)
    stats_wind_test_np = np.concatenate([stats_wind_test[stat] for stat in stats_wind], axis=1)
    
    truth_pres_train = np.concatenate(truth_pres_train)
    truth_pres_val = np.concatenate(truth_pres_val)
    truth_pres_test = np.concatenate(truth_pres_test)
    
    stats_pres_train_np = np.concatenate([stats_pres_train[stat] for stat in stats_pres], axis=1)
    stats_pres_val_np = np.concatenate([stats_pres_val[stat] for stat in stats_pres], axis=1)
    stats_pres_test_np = np.concatenate([stats_pres_test[stat] for stat in stats_pres], axis=1)
    
    # normalize 
    
    if not jsdiv:
        truth_wind_train_mean, truth_wind_train_std = np.mean(truth_wind_train), np.std(truth_wind_train)
        truth_pres_train_mean, truth_pres_train_std = np.mean(truth_pres_train), np.std(truth_pres_train)
    else:
        truth_wind_train_mean, truth_wind_train_std = truth_wind_train.min()-(1e-3), truth_wind_train.max()-truth_wind_train.min()
        truth_pres_train_mean, truth_pres_train_std = truth_pres_train.min()-(1e-3), truth_pres_train.max()-truth_pres_train.min()
        
    
    truth_wind_train = (truth_wind_train - truth_wind_train_mean)/truth_wind_train_std
    truth_wind_val = (truth_wind_val - truth_wind_train_mean)/truth_wind_train_std
    truth_wind_test = (truth_wind_test - truth_wind_train_mean)/truth_wind_train_std
    
    truth_pres_train = (truth_pres_train - truth_pres_train_mean)/truth_pres_train_std
    truth_pres_val = (truth_pres_val - truth_pres_train_mean)/truth_pres_train_std
    truth_pres_test = (truth_pres_test - truth_pres_train_mean)/truth_pres_train_std
    
    stats_wind_train_mean, stats_wind_train_std = np.mean(stats_wind_train_np, axis=0), np.std(stats_wind_train_np, axis=0)
    stats_pres_train_mean, stats_pres_train_std = np.mean(stats_pres_train_np, axis=0), np.std(stats_pres_train_np, axis=0)
    
    if not os.path.isfile(save_path + "Constants/" + f"norma_cst_{model_name}_{lead_time}h"\
                        + f"_{basin}_{'_'.join(sorted(train_seasons))}{'_jsdiv' if jsdiv else ''}_wind_{'_'.join(stats_wind)}_pres_{'_'.join(stats_pres)}.npy"):
        np.save(save_path + "Constants/" + f"norma_cst_{model_name}_{lead_time}h"\
                        + f"_{basin}_{'_'.join(sorted(train_seasons))}{'_jsdiv' if jsdiv else ''}_wind_{'_'.join(stats_wind)}_pres_{'_'.join(stats_pres)}.npy", 
                np.array([stats_wind_train_mean, stats_wind_train_std, stats_pres_train_mean, stats_pres_train_std]))
        
    if not os.path.isfile(save_path + "Constants/" + f"norma_cst_truth_{model_name}_{lead_time}h"\
                        + f"_{basin}_{'_'.join(sorted(train_seasons))}{'_jsdiv' if jsdiv else ''}.npy"):
        np.save(save_path + "Constants/" + f"norma_cst_truth_{model_name}_{lead_time}h"\
                        + f"_{basin}_{'_'.join(sorted(train_seasons))}{'_jsdiv' if jsdiv else ''}.npy", 
                np.array([truth_wind_train_mean, truth_wind_train_std, truth_pres_train_mean, truth_pres_train_std]))
    
    stats_wind_train_np = (stats_wind_train_np - stats_wind_train_mean)/stats_wind_train_std
    stats_wind_val_np = (stats_wind_val_np - stats_wind_train_mean)/stats_wind_train_std
    stats_wind_test_np = (stats_wind_test_np - stats_wind_train_mean)/stats_wind_train_std
    
    stats_pres_train_np = (stats_pres_train_np - stats_pres_train_mean)/stats_pres_train_std
    stats_pres_val_np = (stats_pres_val_np - stats_pres_train_mean)/stats_pres_train_std
    stats_pres_test_np = (stats_pres_test_np - stats_pres_train_mean)/stats_pres_train_std
    
    # transform into xgb.DMatrix
            
    dtrain_wind = xgb.DMatrix(stats_wind_train_np, label=truth_wind_train, feature_names=stats_wind)
    dval_wind = xgb.DMatrix(stats_wind_val_np, label=truth_wind_val, feature_names=stats_wind)
    dtest_wind = xgb.DMatrix(stats_wind_test_np, label=truth_wind_test, feature_names=stats_wind)
    
    dtrain_pres = xgb.DMatrix(stats_pres_train_np, label=truth_pres_train, feature_names=stats_pres)
    dval_pres = xgb.DMatrix(stats_pres_val_np, label=truth_pres_val, feature_names=stats_pres)
    dtest_pres = xgb.DMatrix(stats_pres_test_np, label=truth_pres_test, feature_names=stats_pres)
    
    norma_cst = {"wind": {"Truth": (truth_wind_train_mean, truth_wind_train_std), "Input": (stats_wind_train_mean, stats_wind_train_std)},
                 "pres": {"Truth": (truth_pres_train_mean, truth_pres_train_std), "Input": (stats_pres_train_mean, stats_pres_train_std)}}

    return dtrain_wind, dval_wind, dtest_wind, dtrain_pres, dval_pres, dtest_pres, stats, stats_wind, stats_pres, nb_tc,\
            truth_wind_test, truth_pres_test, train_seasons, val_seasons, test_seasons, norma_cst



def split_data_new(model_name, lead_time, basin, train_seasons, val_seasons, test_seasons, stats_wind, stats_pres, data_path, df_path, save_path):
    
    nb_tc = {"train":0, "val":0, "test":0}
    stats_wind_list_train, stats_wind_list_val, stats_wind_list_test = {key:[] for key in stats_wind},\
                                                                       {key:[] for key in stats_wind},\
                                                                       {key:[] for key in stats_wind}
    stats_pres_list_train, stats_pres_list_val, stats_pres_list_test = {key:[] for key in stats_pres},\
                                                                       {key:[] for key in stats_pres},\
                                                                       {key:[] for key in stats_pres}
    
    truth_wind_list_train, truth_wind_list_val, truth_wind_list_test = [], [], []
    truth_pres_list_train, truth_pres_list_val, truth_pres_list_test = [], [], []
    
    for season in train_seasons:
        stats_wind_list, stats_pres_list, truth_wind_list, truth_pres_list, tc_tmp = statistics_loader(data_path, model_name, lead_time,
                                                                                                    season, basin, df_path, save_path)
        stats_wind_list_train = {key:val+stats_wind_list[key] for key, val in stats_wind_list_train.items()}
        stats_pres_list_train = {key:val+stats_pres_list[key] for key, val in stats_pres_list_train.items()}
        truth_wind_list_train.extend(truth_wind_list)
        truth_pres_list_train.extend(truth_pres_list)
        nb_tc["train"] += len(tc_tmp)
    
    for season in val_seasons:
        stats_wind_list, stats_pres_list, truth_wind_list, truth_pres_list, tc_tmp = statistics_loader(data_path, model_name, lead_time, 
                                                                                                    season, basin)
        stats_wind_list_val = {key:val+stats_wind_list[key] for key, val in stats_wind_list_val.items()}
        stats_pres_list_val = {key:val+stats_pres_list[key] for key, val in stats_pres_list_val.items()}
        truth_wind_list_val.extend(truth_wind_list)
        truth_pres_list_val.extend(truth_pres_list)
        nb_tc["val"] += len(tc_tmp)
        
    for season in test_seasons:
        stats_wind_list, stats_pres_list, truth_wind_list, truth_pres_list, tc_tmp = statistics_loader(data_path, model_name, lead_time, 
                                                                                                    season, basin)
        stats_wind_list_test = {key:val+stats_wind_list[key] for key, val in stats_wind_list_test.items()}
        stats_pres_list_test = {key:val+stats_pres_list[key] for key, val in stats_pres_list_test.items()}
        truth_wind_list_test.extend(truth_wind_list)
        truth_pres_list_test.extend(truth_pres_list)
        nb_tc["test"] += len(tc_tmp)
        
    return stats_wind_list_train, stats_wind_list_val, stats_wind_list_test, stats_pres_list_train, stats_pres_list_val, stats_pres_list_test,\
            truth_wind_list_train, truth_wind_list_val, truth_wind_list_test, truth_pres_list_train, truth_pres_list_val, truth_pres_list_test, nb_tc

            

def create_booster(booster_type, dtrain_wind, dval_wind, dtrain_pres, dval_pres, max_depth, learning_rate, gamma, jsdiv=False):
    
    param = {}
    param['booster'] = booster_type
    param['verbosity'] = 0
    if not jsdiv:
        param['objective'] = 'reg:squarederror'
        param['eval_metric'] = 'rmse'
    else:
        param['disable_default_eval_metric'] = 1
    
    if booster_type=="gbtree":    
        param['max_depth'] = max_depth
        param['eta'] = learning_rate
        param['gamma'] = gamma
        param['subsample'] = 0.8
        
    if booster_type=="gblinear":
        param['lambda'] = 0.1
        param['alpha'] = 0.1
        param["feature_selector"] = "shuffle"
        
    param_wind = param.copy()
    param_pres = param.copy()
    if jsdiv:
        param_pres['eta'] = learning_rate*2
    param_pres['eta'] = learning_rate
    evallist_wind = [(dtrain_wind, 'train'), (dval_wind, 'val')]
    evallist_pres = [(dtrain_pres, 'train'), (dval_pres, 'val')]
    
    return param_wind, param_pres, evallist_wind, evallist_pres


def cos_annealing(epoch, max_epochs, lr_ini):
    return 0.05 + 0.5 * (lr_ini-0.05) * (1 + np.cos((np.abs(epoch-max_epochs//2))/max_epochs*np.pi))


def custome_JSDiv_Obj(predt: np.ndarray, dtrain: xgb.DMatrix):
    truth = dtrain.get_label()
    
    # just to ensure positivity
    predt[predt+truth<0] = 1e-6 - (predt+truth)[predt+truth<0] + predt[predt+truth<0]
    grad = 1/2*np.log(2*predt/(truth+predt))
    hess = 1/2*(truth/(predt*truth + predt**2))
    return grad, hess


def custom_JSDiv_Loss(predt: np.ndarray, dtrain: xgb.DMatrix):
    truth = dtrain.get_label()
    
    # just to ensure positivity
    predt[predt+truth<0] = 1e-6 - (predt+truth)[predt+truth<0] + predt[predt+truth<0]
    loss = 1/2*np.sum(truth*np.log(2*truth/(predt+truth)) + predt*np.log(2*predt/(truth+predt))) / len(truth)
    return 'jsdiv_loss', loss


def train_save_xgb(params, dtrain, num_round, evals, early_stopping_rounds=10, callbacks=[], obj=None, custom_metric=None,
                   save_path="/users/lpoulain/louis/plots/xgboost/",
                   save_name=""):
    
    eval_res = {}
    bst = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_round, evals=evals, evals_result=eval_res, 
                    early_stopping_rounds=early_stopping_rounds, callbacks=callbacks, obj=obj, custom_metric=custom_metric)
    
    bst.save_model(save_path + "Models/" + save_name)
    
    return bst, eval_res


def plot_xgb(bst_wind, bst_pres, eval_res_wind, eval_res_pres, model_name, xgb_model, lead_time=0, train_seasons=0, val_seasons=0,
             test_seasons=0, basin='', nb_tc=0, max_depth=0, n_rounds=0, learning_rate=0., gamma=0., sched=False, stats=[], 
             stats_wind=[], stats_pres=[], jsdiv=False, test_loss_wind=0., test_loss_pres=0., truth_wind_test=None, 
             truth_pres_test=None, y_pred_wind=None, y_pred_pres=None,
             save_path="/users/lpoulain/louis/plots/xgboost/",
            ):
    
    if len(stats)>0:
        for s in stats:
            stats_wind.remove(s)
            stats_pres.remove(s)
    
    loss = "RMSE" if not jsdiv else "Mean Jensen-Shannon Div"
    loss_name = "JSDiv_loss" if jsdiv else "RMSE"
    
    grid = [['a)', 'b)'],
            ['c)', 'd)'],
            ['g)', 'h)'],
            ['e)', 'e)'],
            ['f)', 'f)']] if max_depth<4 else\
           [['a)', 'b)'],
            ['c)', 'd)'],
            ['e)', 'f)']]
    
    fig, axs = plt.subplot_mosaic(grid, figsize=(15, 20))
    if max_depth<4:
        ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axs.values()
    else:
        ax1, ax2, ax3, ax4, ax5, ax6 = axs.values()
    
    xgb.plot_importance(bst_wind, importance_type="cover", show_values=False if max_depth<4 else True, ax=ax1, 
                        title="Importance plot (average cover) - Wind", values_format="{v:.2f}")
    xgb.plot_importance(bst_pres, importance_type="cover", show_values=False if max_depth<4 else True, ax=ax2, 
                        title="Importance plot (average cover) - Pressure", values_format="{v:.2f}")
    
    ax3.plot(eval_res_wind['train'][loss_name.lower()], label="train loss")
    ax3.plot(eval_res_wind['val'][loss_name.lower()], label="val loss", alpha=0.5)
    ax3.set_title("Losses - Wind")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel(f"{loss}")
    ax3.legend()
    
    ax4.plot(eval_res_pres['train'][loss_name.lower()], label="train loss")
    ax4.plot(eval_res_pres['val'][loss_name.lower()], label="val loss", alpha=0.5)
    ax4.set_title("Losses - Pressure")
    ax4.set_xlabel("Epochs")
    ax4.set_ylabel(f"{loss}")
    ax4.legend()
    
    # def bins for scatter
    wind_bins = global_wind_bins
    pres_bins = global_pres_bins
    
    # create histograms
    
    hist_wind, e_x_wind, e_y_wind = np.histogram2d(y_pred_wind, truth_wind_test,
                                                   bins=[wind_bins, wind_bins], density=True)
    x_wind, y_wind = np.meshgrid(e_x_wind, e_y_wind)
    
    hist_pres, e_x_pres, e_y_pres = np.histogram2d(y_pred_pres, truth_pres_test,
                                                   bins=[pres_bins, pres_bins], density=True)
    x_pres, y_pres = np.meshgrid(e_x_pres, e_y_pres)
    
    wnd = ax5.pcolormesh(x_wind, y_wind, hist_wind.T, cmap='plasma', norm=LogNorm())
    cbar = plt.colorbar(wnd, label='Density')
    wind_extent = [min(y_pred_wind.min(), truth_wind_test.min()),
                   max(y_pred_wind.max(), truth_wind_test.max())]
    ax5.plot(wind_extent, wind_extent, color="black", label="Identity")
    ax5.set_xlim(wind_extent)
    ax5.set_ylim(wind_extent)
    ax5.set_xlabel("Post-processed wind (m/s)")
    ax5.set_ylabel("Observed wind (m/s)")
    ax5.set_title(f"Model: {model_name} at ldt {lead_time} - Wind density histogram")
    
    
    wnd = ax6.pcolormesh(x_pres, y_pres, hist_pres.T, cmap='plasma', norm=LogNorm())
    cbar = plt.colorbar(wnd, label='Density')
    pres_extent = [min(y_pred_pres.min(), truth_pres_test.min()),
                   max(y_pred_pres.max(), truth_pres_test.max())]
    ax6.plot(pres_extent, pres_extent, color="black", label="Identity")
    ax6.set_xlim(pres_extent)
    ax6.set_ylim(pres_extent)
    ax6.set_xlabel("Post-processed pressure (Pa)")
    ax6.set_ylabel("Observed pressure (Pa)")
    ax6.set_title(f"Model: {model_name} at ldt {lead_time} - Pressure density histogram")
    
    if max_depth<4:
        xgb.plot_tree(bst_wind, ax=ax7, num_trees=0)
        ax7.set_title("Tree plot - Wind")
        
        xgb.plot_tree(bst_pres, ax=ax8, num_trees=0)
        ax8.set_title("Tree plot - Pressure")
    
    
    train_seasons, val_seasons = sorted(train_seasons), sorted(val_seasons)
    st = fig.suptitle(multiline_label(f"{model_name} | {lead_time}h | Train: {', '.join(train_seasons)} ({nb_tc['train']} TCs)"\
            + f" Val: {', '.join(val_seasons)} ({nb_tc['val']} TCs) Test: {', '.join(test_seasons)} ({nb_tc['test']} TCs)"\
            + f" - Basin: {basin}", cutting=3) + f"\nTest RMSE: {test_loss_wind:.4f} (wind), {test_loss_pres:.4f} (pres)")
    st.set_y(0.98)
    fig.subplots_adjust(bottom=0.005, top=0.90, left=0.05, right=0.95)
    fig.tight_layout()
    
    
    fig.savefig(save_path + "Figs/" + f"{xgb_model}_{model_name}_{lead_time}h_{'_'.join(s for s in sorted(train_seasons))}_{basin}_depth_"+\
            f"{max_depth}_epoch_{n_rounds}_lr_{learning_rate}_g_{gamma}"+\
            (f"_{loss_name}" if jsdiv else "")+\
            ("_sched" if sched else "")+\
            (f"_{'_'.join(stat for stat in stats)}" if len(stats)>0 else "")+\
            (f"_w_{'_'.join(stat for stat in stats_wind)}" if len(stats_wind)>0 else "") +\
            (f"_p_{'_'.join(stat for stat in stats_pres)}" if len(stats_pres)>0 else "") +".png", dpi=500)
    
    
    
    
"""def create_dataset(data_path, model_name, lead_time, seasons, basin, split_ratio:list,
            stats=[], stats_wind=["max"], stats_pres=["min"],
            df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv",
            save_path="/users/lpoulain/louis/plots/xgboost/"
            ):
    
    if stats==[''] or stats==[' ']:
        stats = []
    assert round(sum(split_ratio),5)==1.0, f"split_ratio must sum to 1, your sum is {'+'.join(str(ratio) for ratio in split_ratio)} ="\
                                                + f"{round(sum(split_ratio),5)}"
    assert set(stats).union(stats_wind).union(stats_pres).issubset(set(stats_list)), f"stats/stats_wind/stats_pres must be a subset of {stats_list}.\n"\
                    + f"Your stats are {stats}, stats_wind are {stats_wind} and stats_pres are {stats_pres}"
    
    if type(seasons)==str or type(seasons)==int:
        seasons = [seasons]
        
    # sort stats to always have the same order
    stats_wind.extend(stats)
    stats_pres.extend(stats)
    stats_wind = sorted(list(set(stats_wind)), key=lambda x: stats_list.index(x))
    stats_pres = sorted(list(set(stats_pres)), key=lambda x: stats_list.index(x))
    
    stats_wind_list, stats_pres_list, truth_wind_list, truth_pres_list, tc_ids = statistics_loader(data_path, model_name, lead_time, 
                                                                                                    seasons[0], basin, df_path, save_path)
    
    if len(seasons)>1:
        for season in seasons[1:]:
            tmp_wind_stats, tmp_pres_stats, tmp_wnd_truth, tmp_pres_truth, tmp_ids = statistics_loader(data_path, model_name, lead_time,
                                                                                                    season, basin, df_path, save_path)
            stats_wind_list = {key:val+tmp_wind_stats[key] for key, val in stats_wind_list.items()}
            stats_pres_list = {key:val+tmp_pres_stats[key] for key, val in stats_pres_list.items()}
            truth_wind_list.extend(tmp_wnd_truth)
            truth_pres_list.extend(tmp_pres_truth)
            tc_ids.extend(tmp_ids)
            
    # retain only selected statistics
    
    stats_wind_list = {key:val for key, val in stats_wind_list.items() if key in stats_wind}
    stats_pres_list = {key:val for key, val in stats_pres_list.items() if key in stats_pres}
    
    # shuffle data
    
    rng = np.random.default_rng(73)
    nb_tc = len(tc_ids)
    idx_train, idx_val = int(split_ratio[0]*nb_tc), int((sum(split_ratio[:2]))*nb_tc)
    shuffle_idx = rng.permutation(nb_tc)

    tc_ids = [tc_ids[idx] for idx in shuffle_idx]
    tc_ids_train, tc_ids_val, tc_ids_test = tc_ids[:idx_train], tc_ids[idx_train:idx_val], tc_ids[idx_val:]
    
    stats_wind_list, stats_pres_list = {key:[val[idx] for idx in shuffle_idx] for key, val in stats_wind_list.items()},\
                                       {key:[val[idx] for idx in shuffle_idx] for key, val in stats_pres_list.items()}
    truth_wind_list, truth_pres_list = [truth_wind_list[idx] for idx in shuffle_idx], [truth_pres_list[idx] for idx in shuffle_idx]
    
    # datasets for xgb
    
    stats_wind_train_np, stats_wind_val_np, stats_wind_test_np,\
    stats_pres_train_np, stats_pres_val_np, stats_pres_test_np,\
    truth_wind_train, truth_wind_val, truth_wind_test,\
    truth_pres_train, truth_pres_val, truth_pres_test = split_data(idx_train, idx_val, stats_wind_list, stats_pres_list, 
                                                                   truth_wind_list, truth_pres_list, stats_wind, stats_pres)
            
    dtrain_wind = xgb.DMatrix(stats_wind_train_np, label=truth_wind_train, feature_names=stats_wind)
    dval_wind = xgb.DMatrix(stats_wind_val_np, label=truth_wind_val, feature_names=stats_wind)
    dtest_wind = xgb.DMatrix(stats_wind_test_np, label=truth_wind_test, feature_names=stats_wind)
    
    dtrain_pres = xgb.DMatrix(stats_pres_train_np, label=truth_pres_train, feature_names=stats_pres)
    dval_pres = xgb.DMatrix(stats_pres_val_np, label=truth_pres_val, feature_names=stats_pres)
    dtest_pres = xgb.DMatrix(stats_pres_test_np, label=truth_pres_test, feature_names=stats_pres)

    return dtrain_wind, dval_wind, dtest_wind, dtrain_pres, dval_pres, dtest_pres, stats, stats_wind, stats_pres, nb_tc,\
            truth_wind_test, truth_pres_test
            


def split_data(idx_train, idx_val, stats_wind_list, stats_pres_list, truth_wind_list, truth_pres_list, stats_wind, stats_pres):
    
    stats_wind_train = {key:val[:idx_train] for key, val in stats_wind_list.items()}
    stats_pres_train = {key:val[:idx_train] for key, val in stats_pres_list.items()}

    truth_wind_train = np.concatenate(truth_wind_list[:idx_train])
    truth_pres_train = np.concatenate(truth_pres_list[:idx_train])
    
    
    stats_wind_val = {key:val[idx_train:idx_val] for key, val in stats_wind_list.items()}
    stats_pres_val = {key:val[idx_train:idx_val] for key, val in stats_pres_list.items()}
    truth_wind_val = np.concatenate(truth_wind_list[idx_train:idx_val])
    truth_pres_val = np.concatenate(truth_pres_list[idx_train:idx_val])
    
    stats_wind_test = {key:val[idx_val:] for key, val in stats_wind_list.items()}
    stats_pres_test = {key:val[idx_val:] for key, val in stats_pres_list.items()}
    truth_wind_test = np.concatenate(truth_wind_list[idx_val:])
    truth_pres_test = np.concatenate(truth_pres_list[idx_val:])
    
    # normalize targets wrt train set
    
    truth_wind_train_mean, truth_wind_train_std = 0., 1. #np.mean(truth_wind_train), np.std(truth_wind_train)
    truth_pres_train_mean, truth_pres_train_std = 0., 1. #np.mean(truth_pres_train), np.std(truth_pres_train)
    
    truth_wind_train = (truth_wind_train - truth_wind_train_mean)/truth_wind_train_std
    truth_wind_val = (truth_wind_val - truth_wind_train_mean)/truth_wind_train_std
    truth_wind_test = (truth_wind_test - truth_wind_train_mean)/truth_wind_train_std
    
    truth_pres_train = (truth_pres_train - truth_pres_train_mean)/truth_pres_train_std
    truth_pres_val = (truth_pres_val - truth_pres_train_mean)/truth_pres_train_std
    truth_pres_test = (truth_pres_test - truth_pres_train_mean)/truth_pres_train_std
    
    # construct datasets in np.array format
    # input data for xgb should be batch x features (ie [[min1, max1, mean1, std1], [min2, max2, mean2, std2], ...])
    # first, we concatenate for each stat, and then each feature (i.e. each stat)
    stats_wind_train = {key:np.concatenate(val).ravel().reshape(-1, 1) for key, val in stats_wind_train.items()}
    stats_wind_val = {key:np.concatenate(val).ravel().reshape(-1, 1) for key, val in stats_wind_val.items()}
    stats_wind_test = {key:np.concatenate(val).ravel().reshape(-1, 1) for key, val in stats_wind_test.items()}
    
    stats_pres_train = {key:np.concatenate(val).ravel().reshape(-1, 1) for key, val in stats_pres_train.items()}
    stats_pres_val = {key:np.concatenate(val).ravel().reshape(-1, 1) for key, val in stats_pres_val.items()}
    stats_pres_test = {key:np.concatenate(val).ravel().reshape(-1, 1) for key, val in stats_pres_test.items()}
    
    stats_wind_train_np = np.concatenate([stats_wind_train[stat] for stat in stats_wind], axis=1)
    stats_wind_val_np = np.concatenate([stats_wind_val[stat] for stat in stats_wind], axis=1)
    stats_wind_test_np = np.concatenate([stats_wind_test[stat] for stat in stats_wind], axis=1)
    
    stats_pres_train_np = np.concatenate([stats_pres_train[stat] for stat in stats_pres], axis=1)
    stats_pres_val_np = np.concatenate([stats_pres_val[stat] for stat in stats_pres], axis=1)
    stats_pres_test_np = np.concatenate([stats_pres_test[stat] for stat in stats_pres], axis=1)
    
    # normalize the sets wrt train set
    stats_wind_train_mean, stats_wind_train_std = 0., 1. #np.mean(stats_wind_train_np, axis=0), np.std(stats_wind_train_np, axis=0)
    stats_pres_train_mean, stats_pres_train_std = 0., 1. #np.mean(stats_pres_train_np, axis=0), np.std(stats_pres_train_np, axis=0)
    
    stats_wind_train_np = (stats_wind_train_np - stats_wind_train_mean)/stats_wind_train_std
    stats_wind_val_np = (stats_wind_val_np - stats_wind_train_mean)/stats_wind_train_std
    stats_wind_test_np = (stats_wind_test_np - stats_wind_train_mean)/stats_wind_train_std    
    
    stats_pres_train_np = (stats_pres_train_np - stats_pres_train_mean)/stats_pres_train_std
    stats_pres_val_np = (stats_pres_val_np - stats_pres_train_mean)/stats_pres_train_std
    stats_pres_test_np = (stats_pres_test_np - stats_pres_train_mean)/stats_pres_train_std
    
    return stats_wind_train_np, stats_wind_val_np, stats_wind_test_np, stats_pres_train_np, stats_pres_val_np, stats_pres_test_np,\
            truth_wind_train, truth_wind_val, truth_wind_test, truth_pres_train, truth_pres_val, truth_pres_test"""