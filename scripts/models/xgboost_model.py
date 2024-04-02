import numpy as np
import xgboost as xgb
import sys
from argparse import ArgumentParser
from utils.main_utils import str2list
from xgboost_utils import create_dataset_new, create_booster, train_save_xgb, plot_xgb, cos_annealing, custom_JSDiv_Loss, custome_JSDiv_Obj




def gbtree(data_path, model_name, lead_time, seasons, force_in_train, basin, split_ratio:list,
            max_depth=6, n_rounds=30, learning_rate=0.3, gamma=0., stats=[],
            stats_wind=["max"], stats_pres=["min"], jsdiv=False, sched=False,
            df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv",
            save_path="/users/lpoulain/louis/plots/xgboost/"
            ):
    
    obj = custome_JSDiv_Obj if jsdiv else None
    custom_metric = custom_JSDiv_Loss if jsdiv else None
    loss_name = '_jsdiv' if jsdiv else ''
    
    model_name = "pangu" if model_name=="panguweather" else model_name
    dtrain_wind, dval_wind, dtest_wind,\
    dtrain_pres, dval_pres, dtest_pres,\
    stats, stats_wind, stats_pres, nb_tc,\
    truth_wind_test, truth_pres_test,\
    train_seasons, val_seasons, test_seasons, norma_cst = create_dataset_new(data_path=data_path, model_name=model_name, lead_time=lead_time, 
                                                      seasons=seasons, force_in_train=force_in_train, basin=basin, split_ratio=split_ratio,
                                                      stats=stats, stats_wind=stats_wind, stats_pres=stats_pres, jsdiv=jsdiv,
                                                      df_path=df_path, save_path=save_path)
    
    print(f"Train seasons: {train_seasons}\nVal seasons: {val_seasons}\nTest seasons: {test_seasons}")
    
    params_wind, params_pres, evallist_wind, evallist_pres = create_booster("gbtree", dtrain_wind=dtrain_wind, dval_wind=dval_wind, 
                                                          dtrain_pres=dtrain_pres, dval_pres=dval_pres, max_depth=max_depth, 
                                                          learning_rate=learning_rate, gamma=gamma, jsdiv=jsdiv)
    
    # train and save models
    
    wind_save = f"xgb_wind_{model_name}_{lead_time}h{loss_name}_basin_{basin}_{'_'.join(s for s in sorted(train_seasons))}_depth_{max_depth}"\
                    + f"_epoch_{n_rounds}_lr_{learning_rate}_g_{gamma}"\
                    + (f"_sched" if sched else "")\
                    + f"_{'_'.join(stat for stat in stats_wind)}.json"
    pres_save = f"xgb_pres_{model_name}_{lead_time}h{loss_name}_basin_{basin}_{'_'.join(s for s in sorted(train_seasons))}_depth_{max_depth}"\
                    + f"_epoch_{n_rounds}_lr_{learning_rate}_g_{gamma}"\
                    + (f"_sched" if sched else "")\
                    + f"_{'_'.join(stat for stat in stats_pres)}.json"
    
    scheduler_wind = lambda x: cos_annealing(x, n_rounds, learning_rate)
    scheduler_pres = lambda x: cos_annealing(x, n_rounds, learning_rate)
    callback_wind = [xgb.callback.LearningRateScheduler(scheduler_wind)] if sched else None
    callback_pres = [xgb.callback.LearningRateScheduler(scheduler_pres)] if sched else None
                        
    bst_wind, eval_res_wind = train_save_xgb(params=params_wind, dtrain=dtrain_wind, num_round=n_rounds, evals=evallist_wind, early_stopping_rounds=n_rounds,
                                            save_path=save_path, save_name=wind_save, callbacks=callback_wind, obj=obj, custom_metric=custom_metric)
    bst_pres, eval_res_pres = train_save_xgb(params=params_pres, dtrain=dtrain_pres, num_round=n_rounds, evals=evallist_pres, early_stopping_rounds=n_rounds,
                                            save_path=save_path, save_name=pres_save, callbacks=callback_pres, obj=obj, custom_metric=custom_metric)
    
    # predict and plot
    
    ypred_wind = bst_wind.predict(dtest_wind, iteration_range=(0, bst_wind.best_iteration + 1))
    ypred_pres = bst_pres.predict(dtest_pres, iteration_range=(0, bst_pres.best_iteration + 1))
    
    test_loss_wind = np.sqrt(np.mean((ypred_wind - truth_wind_test)**2))
    test_loss_pres = np.sqrt(np.mean((ypred_pres - truth_pres_test)**2))
    
        
    ypred_wind = ypred_wind * norma_cst["wind"]["Truth"][1] + norma_cst["wind"]["Truth"][0]
    truth_wind_test = truth_wind_test * norma_cst["wind"]["Truth"][1] + norma_cst["wind"]["Truth"][0]
    
    ypred_pres = ypred_pres * norma_cst["pres"]["Truth"][1] + norma_cst["pres"]["Truth"][0]
    truth_pres_test = truth_pres_test * norma_cst["pres"]["Truth"][1] + norma_cst["pres"]["Truth"][0]
    
    plot_xgb(bst_wind=bst_wind, bst_pres=bst_pres, eval_res_wind=eval_res_wind, eval_res_pres=eval_res_pres, 
             model_name=model_name, xgb_model="gbtree", lead_time=lead_time, train_seasons=train_seasons, val_seasons=val_seasons,
             test_seasons=test_seasons, basin=basin, nb_tc=nb_tc, max_depth=max_depth, n_rounds=n_rounds, learning_rate=learning_rate, 
             gamma=gamma, sched=sched, stats=stats, stats_wind=stats_wind, stats_pres=stats_pres, jsdiv=jsdiv, test_loss_wind=test_loss_wind, 
             test_loss_pres=test_loss_pres, truth_wind_test=truth_wind_test, truth_pres_test=truth_pres_test, y_pred_wind=ypred_wind, y_pred_pres=ypred_pres,
             save_path="/users/lpoulain/louis/plots/xgboost/"
             )
    



def gblinear(data_path, model_name, lead_time, seasons, basin, split_ratio:list,
            max_depth=6, n_rounds=30, learning_rate=0.3, stats=[],
            stats_wind=["max"], stats_pres=["min"], jsdiv=False,
            df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv",
            save_path="/users/lpoulain/louis/plots/xgboost/"
            ):
    
    obj = custome_JSDiv_Obj if jsdiv else None
    custom_metric = custom_JSDiv_Loss if jsdiv else None
    
    dtrain_wind, dval_wind, dtest_wind,\
    dtrain_pres, dval_pres, dtest_pres,\
    stats_wind, stats_pres, nb_tc,\
    truth_wind_test, truth_pres_test = create_dataset_new(data_path=data_path, model_name=model_name, lead_time=lead_time, 
                                                      seasons=seasons, basin=basin, split_ratio=split_ratio,
                                                      stats=stats, stats_wind=stats_wind, stats_pres=stats_pres,
                                                      jsdiv=jsdiv, df_path=df_path)
    
    params, evallist_wind, evallist_pres = create_booster("gblinear", dtrain_wind, dval_wind, dtrain_pres, dval_pres, max_depth, learning_rate)
    
    # train and save models
    
    wind_save = f"xgblinear_wind_{model_name}_{lead_time}h_{'_'.join(s for s in seasons)}_depth_{max_depth}_epoch_{n_rounds}_lr_{learning_rate}" +\
                        f"_{'_'.join(stat for stat in stats_wind)}.json"
    pres_save = f"xgblinear_pres_{model_name}_{lead_time}h_{'_'.join(s for s in seasons)}_depth_{max_depth}_epoch_{n_rounds}_lr_{learning_rate}" +\
                        f"_{'_'.join(stat for stat in stats_pres)}.json"
                        
                        
    bst_wind, eval_res_wind = train_save_xgb(params, dtrain_wind, n_rounds, evallist_wind, early_stopping_rounds=10,
                                            save_path=save_path, save_name=wind_save, obj=obj, custom_metric=custom_metric)
    bst_pres, eval_res_pres = train_save_xgb(params, dtrain_pres, n_rounds, evallist_pres, early_stopping_rounds=10,
                                            save_path=save_path, save_name=pres_save, obj=obj, custom_metric=custom_metric)
    
    # predict and plot
    
    ypred_wind = bst_wind.predict(dtest_wind, iteration_range=(0, bst_wind.best_iteration + 1))
    ypred_pres = bst_pres.predict(dtest_pres, iteration_range=(0, bst_pres.best_iteration + 1))
    
    test_loss_wind = np.sqrt(np.mean((ypred_wind - truth_wind_test)**2))
    test_loss_pres = np.sqrt(np.mean((ypred_pres - truth_pres_test)**2))

    
    plot_xgb(bst_wind, bst_pres, eval_res_wind, eval_res_pres, model_name, "gblinear", lead_time, seasons, basin, nb_tc,
                max_depth, n_rounds, learning_rate, stats, stats_wind, stats_pres, jsdiv, test_loss_wind, test_loss_pres,
                save_path="/users/lpoulain/louis/plots/xgboost/",
                )
    


def dart():
    pass

    
    
if __name__ == "__main__":
    
    fct_dict = {"gbtree":gbtree,
                "dart":dart,
                "gblinear":gblinear,
                }
    
    
    parser = ArgumentParser()
    # paths
    parser.add_argument("--data_path", type=str, default="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/")
    parser.add_argument("--df_path", type=str, 
                        default="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv")
    parser.add_argument("--save_path", type=str, default="/users/lpoulain/louis/plots/xgboost/")
    
    # data params
    parser.add_argument("--split_ratio", type=str2list, default=[0.7,0.2,0.1])
    
    # AI models params
    parser.add_argument("--lead_time", type=int, default=6)
    parser.add_argument("--model", type=str, default="graphcast", choices=["graphcast", "pangu", "panguweather"])
    parser.add_argument("--seasons", type=str2list, default=["2000"])
    parser.add_argument("--force_in_train", type=str2list, help="Force the seasons passed in the list to be in train set", default=[])
    parser.add_argument("--basin", type=str, default="NA", choices=["NA", "WP", "EP", "NI", "SI", "SP", "all"]) # Not enough TCs in SA
    
    # type of xgb model
    parser.add_argument('-f', '--func', dest='func', choices=fct_dict.keys(), required=True,
                            help="Choose one of the specified function to be run.")
    
    # stats as input of xgb model
    parser.add_argument("--stats", type=str2list, default=[])
    parser.add_argument("--stats_wind", type=str2list, default=["max"])
    parser.add_argument("--stats_pres", type=str2list, default=["min"])
    
    # xgb model params
    parser.add_argument("--model_args", type=str2list, default=["lr","0.05","depth","3","epochs","20"],
                        help="List of arguments for the model, in the form [arg1,val1,arg2,val2, ...]")
    parser.add_argument("--jsdiv", action="store_true", help="Use Jensen-Shannon divergence as loss function")
    parser.add_argument("--sched", action="store_true", help="Use a scheduler for the learning rate [cos_annealing]")
    
    args = parser.parse_args()
    print("Input args:\n", args)
    
    # paths
    data_path, df_path, save_path = args.data_path, args.df_path, args.save_path
    
    # AI models params
    lead_time, model, seasons, basin = args.lead_time, args.model, args.seasons, args.basin
    force_in_train = args.force_in_train
    if force_in_train == [''] or force_in_train == [' ']:
        force_in_train = []
    
    # xgb method and inputs
    fct = fct_dict[args.func]
    stats, stats_wind, stats_pres = args.stats, args.stats_wind, args.stats_pres
    
    # specific xgb model args
    model_args = {"depth":3, "epochs":20, "lr":0.01, "gamma":0.}
    models_args_dtypes = {"depth":int, "epochs":int, "lr":float, "gamma":float}
    jsdiv = args.jsdiv
    sched = args.sched
    
    for i in range(0, len(args.model_args), 2):
        model_args[args.model_args[i]] = models_args_dtypes[args.model_args[i]](args.model_args[i+1])
    
    print("Model args: ", model_args)
    
    # split ratio
    split_ratio = [float(ratio) for ratio in args.split_ratio]
    if stats_wind == [''] or stats_wind == [' ']:
        stats_wind = []
    if stats_pres == [''] or stats_pres == [' ']:
        stats_pres = []
    if stats == [''] or stats == [' ']:
        stats = []
    
    fct(data_path=data_path, model_name=model, lead_time=lead_time, seasons=seasons, force_in_train=force_in_train, basin=basin, split_ratio=split_ratio,
            max_depth=model_args["depth"], n_rounds=model_args["epochs"], learning_rate=model_args["lr"], gamma=model_args["gamma"], 
            stats=stats, stats_wind=stats_wind, stats_pres=stats_pres, jsdiv=jsdiv, sched=sched,
            df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv",
            save_path="/users/lpoulain/louis/plots/xgboost/")