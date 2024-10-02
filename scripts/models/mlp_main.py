from mlp_utils import *
from utils.main_utils import str2list, str2intlist
from argparse import ArgumentParser

optims_dict = {"adam": torch.optim.Adam}
scheds_dict = {"step_lr": lambda x: torch.optim.lr_scheduler.StepLR(x, step_size=10, gamma=0.1), 
               "cosine_annealing": lambda x,y: torch.optim.lr_scheduler.CosineAnnealingLR(x, T_max=y, eta_min=0.)}
criterions_dict = {"mlp": nn.MSELoss(), "mlp_normal": NormalLossMLP, "mlp_shash": ShashLossMLP}
mlp_dict = {"mlp": MLP, "mlp_normal": MLP_normal, "mlp_shash": MLP_shash}


def main(data_path, model_name, lead_time, seasons, split_ratio:list,
            epochs=30, learning_rate=0.3, hidden_units=[5,5,5], mlp_type="mlp", 
            optim="adam", sched=None, stats=[], stats_wind=["max"], stats_pres=["min"],
            df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv",
            save_path="/users/lpoulain/louis/plots/mlp/"
            ):
    
    assert optim in optims_dict.keys(), f"optim must be in {optims_dict.keys()}, not {optim}"
    assert sched in scheds_dict.keys(), f"sched must be in {scheds_dict.keys()}, not {sched}"
    assert mlp_type in mlp_dict.keys(), f"mlp_type must be in {mlp_dict.keys()}, not {mlp_type}"
    # create datasets
    
    dtrain_wind, dval_wind, dtest_wind,\
    dtrain_pres, dval_pres, dtest_pres,\
    truth_wind_train, truth_wind_val, truth_wind_test,\
    truth_pres_train, truth_pres_val, truth_pres_test,\
    stats, stats_wind, stats_pres, nb_tcs = create_dataset_mlp(data_path, model_name, lead_time, seasons, split_ratio,
                                            stats=stats, stats_wind=stats_wind, stats_pres=stats_pres,
                                            df_path=df_path, save_path=save_path)
    
    # create dataloaders
    
    train_wind_loader, val_wind_loader, test_wind_loader = mlp_dataloader(dtrain_wind, dval_wind, dtest_wind, 
                                                                        truth_wind_train, truth_wind_val, truth_wind_test,
                                                                        batch_size=32)
    train_pres_loader, val_pres_loader, test_pres_loader = mlp_dataloader(dtrain_pres, dval_pres, dtest_pres,
                                                                        truth_pres_train, truth_pres_val, truth_pres_test,
                                                                        batch_size=32)
    
    # create models
    wind_inputs = len(stats_wind)
    pres_inputs = len(stats_pres)

    mlp_wind = mlp_dict[mlp_type](input_dim=wind_inputs, hidden_units=hidden_units, dropout_rate=0.2).apply(initialize_weights).float()
    mlp_pres = mlp_dict[mlp_type](input_dim=pres_inputs, hidden_units=hidden_units, dropout_rate=0.2).apply(initialize_weights).float()
    
    # initialize optim amd sched
    
    optimizer_wind = optims_dict[optim](mlp_wind.parameters(), lr=learning_rate)
    optimizer_pres = optims_dict[optim](mlp_pres.parameters(), lr=learning_rate)
    
    if sched is not None:
        if sched=="cosine_annealing":
            scheduler_wind = scheds_dict[sched](optimizer_wind, epochs//2)
            scheduler_pres = scheds_dict[sched](optimizer_pres, epochs//2)
        else:
            scheduler_wind = scheds_dict[sched](optimizer_wind)
            scheduler_pres = scheds_dict[sched](optimizer_pres)
    else:
        scheduler_wind = None
        scheduler_pres = None
        
    # train models and save
    train_losses, val_losses = {}, {}
    criterion = criterions_dict[mlp_type]
    
    train_losses["wind"], val_losses["wind"] = train_mlp(mlp_wind, train_wind_loader, val_wind_loader, criterion=criterion,
                                                        optimizer=optimizer_wind, num_epochs=epochs, scheduler=scheduler_wind, var="wind")
    train_losses["pres"], val_losses["pres"] = train_mlp(mlp_pres, train_pres_loader, val_pres_loader, criterion=criterion,
                                                        optimizer=optimizer_pres, num_epochs=epochs, scheduler=scheduler_pres, var="pres")
    
    torch.save(mlp_wind.state_dict(), f"{save_path}mlp_wind_{model_name}_{lead_time}h_{'_'.join(seasons)}_layers_"+\
                f"{'.'.join(str(h) for h in hidden_units)}_lr_{learning_rate}_epochs_{epochs}_sched_{sched}"+\
                f"_{'_'.join(stat for stat in stats_wind)}.pt")
    torch.save(mlp_pres.state_dict(), f"{save_path}mlp_pres_{model_name}_{lead_time}h_{'_'.join(seasons)}_layers_"+\
                f"{'.'.join(str(h) for h in hidden_units)}_lr_{learning_rate}_epochs_{epochs}_sched_{sched}"+\
                f"_{'_'.join(stat for stat in stats_pres)}.pt")
    
    
    # plot
    mlp_params = {"lr": learning_rate, "epochs": epochs, "scheduler": sched, "layers": hidden_units, "mlp_type": mlp_type}
    
    plot_mlp(mlp_wind, mlp_pres, train_losses, val_losses, test_wind_loader, test_pres_loader, model_name, seasons, num_tcs=nb_tcs,
             stats = stats, stats_wind = stats_wind, stats_pres = stats_pres, mlp_params=mlp_params, lead_time=lead_time,
             save_path="/users/lpoulain/louis/plots/mlp/")
    
    
if __name__ == "__main__":
    
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
    parser.add_argument("--model", type=str, default="graphcast")
    parser.add_argument("--seasons", type=str2list, default=["2000"])
    
    # stats as input of mlp model
    parser.add_argument("--stats", type=str2list, default=[])
    parser.add_argument("--stats_wind", type=str2list, default=["max"])
    parser.add_argument("--stats_pres", type=str2list, default=["min"])
    
    # mlp model params
    parser.add_argument("--model_args", type=str2list, default=["lr",0.5,"epochs",50],
                        help="List of arguments for the model, in the form [arg1,val1,arg2,val2, ...]")
    parser.add_argument("--type", type=str, default="mlp", choices=["mlp", "mlp_normal", "mlp_shash"])
    parser.add_argument("--hidden_units", type=str2intlist, default=[5,5,5])
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--sched", type=str, default="cosine_annealing")
    
    args = parser.parse_args()
    print("Input args:\n", args)
    
    # paths
    data_path, df_path, save_path = args.data_path, args.df_path, args.save_path
    
    # AI models params
    lead_time, model, seasons = args.lead_time, args.model, args.seasons
    
    # mlp method and inputs
    stats, stats_wind, stats_pres = args.stats, args.stats_wind, args.stats_pres
    optim, sched = args.optim, args.sched
    mlp_type = args.type
    
    # specific mlp model args
    model_args = {"lr":0.5,"hidden":[5,5,5],"epochs":50}
    models_args_dtypes = {"hidden":str2list, "epochs":int, "lr":float}
    
    for i in range(0, len(args.model_args), 2):
        model_args[args.model_args[i]] = models_args_dtypes[args.model_args[i]](args.model_args[i+1])
    model_args["hidden"] = args.hidden_units
    
    print("Model args: ", model_args)
    
    # split ratio
    split_ratio = [float(ratio) for ratio in args.split_ratio]
    
    main(data_path=data_path, model_name=model, lead_time=lead_time, seasons=seasons, split_ratio=split_ratio,
            epochs=model_args["epochs"], learning_rate=model_args["lr"], hidden_units=model_args["hidden"], mlp_type=mlp_type,
            optim=optim, sched=sched, stats=stats, stats_wind=stats_wind, stats_pres=stats_pres, df_path=df_path, save_path=save_path,
            )