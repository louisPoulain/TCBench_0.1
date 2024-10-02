from argparse import ArgumentParser
import numpy as np
import torch, os, random
import matplotlib.pyplot as plt
import torch.nn as nn

from cnn_loaders import CNN4PP_Dataset
from cnn_blocks import CNN4PP, CNN_train
from cnn_utils import plot_deterministic, plot_probabilistic, CRPSLoss
from utils.main_utils import str2list, str2intlist, str2bool
from mlp_main import optims_dict, scheds_dict
from torch.utils.data import DataLoader
from time import time


def main_cnn(data_path, model_name, seasons, split_ratio:list,
            epochs=30, learning_rate=0.001,
            optim="adam", sched=None, pres=True, crps=False,
            df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv",
            save_path="/users/lpoulain/louis/plots/cnn/"
            ):

    assert optim in optims_dict.keys(), f"optim must be in {optims_dict.keys()}, not {optim}"
    assert sched in scheds_dict.keys(), f"sched must be in {scheds_dict.keys()}, not {sched}"
    assert round(sum(split_ratio),5)==1.0, f"split_ratio must sum to 1, your sum is {'+'.join(str(ratio) for ratio in split_ratio)} ="\
                                                + f"{round(sum(split_ratio),5)}"
    assert len(seasons)>=3, f"You need at least 3 seasons to split into train, val and test. Your input is {seasons} (len={len(seasons)})"
     
    s_copy = seasons.copy()   
    # as long as len(seasons)>3 and split is less than 33% for val and test is should be ok
    test_seasons = s_copy[::-1][:int(np.ceil(split_ratio[-1]*len(s_copy)))][::-1]
    s_copy = [s for s in s_copy if s not in test_seasons]
    
    # Shuffle the validation + train to avoid having the same seasons in val and train ateach training
    random.shuffle(s_copy)
    val_seasons = s_copy[::-1][:int(np.ceil(split_ratio[-2]*len(s_copy)))][::-1]
    s_copy = [s for s in s_copy if s not in val_seasons]
    train_seasons = s_copy
    
    print(f"Train seasons: {train_seasons}\nVal seasons: {val_seasons}\nTest seasons: {test_seasons}")
    
    t = time()
    train_set = CNN4PP_Dataset(data_path, model_name, df_path, train_seasons, pres=pres, train_seasons=train_seasons)
    t_train = time()-t
    t = time()
    val_set = CNN4PP_Dataset(data_path, model_name, df_path, val_seasons, pres=pres, train_seasons=train_seasons)
    t_val = time()-t
    t = time()
    test_set = CNN4PP_Dataset(data_path, model_name, df_path, test_seasons, pres=pres, train_seasons=train_seasons)
    t_test = time()-t
    print(f"Dataset built in {t_train:.2f} (train) + {t_val:.2f} (val) + {t_test:.2f} (test) = {t_train+t_val+t_test:.2f} s")
    
    # for early convergence with 10% of train data
    subset_train_loader = DataLoader(torch.utils.data.Subset(train_set, random.sample(range(0, len(train_set)), len(train_set)//10)), 
                                     batch_size=512, shuffle=True, num_workers=4)
    
    # prepare dataloaders
    print("Len of train/val/test sets: ", len(train_set), len(val_set), len(test_set))
    
    train_loader = DataLoader(train_set, batch_size=1024, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=1024, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False, num_workers=4)
    print("Len of train/val/test loaders: ", len(train_loader), len(val_loader), len(test_loader))
    
    # Initialize model
    
    deterministic = crps is False
    print("Deterministic?", deterministic)
    model = CNN4PP(in_channels=2 if pres else 1, out_channels=8, kernel_size=7, stride=1, padding=1, bias=True, deterministic=deterministic)
    
    optimizer = optims_dict[optim](model.parameters(), lr=learning_rate)
    scheduler = scheds_dict[sched](optimizer, epochs) if sched is not None else None
    
    train_losses, val_losses = CNN_train(model=model, subset_train_loader=subset_train_loader, train_loader=train_loader,
                                         val_loader=val_loader, optimizer=optimizer, num_epochs=epochs, scheduler=scheduler, 
                                         deterministic=deterministic)
    print("Train losses: ", train_losses)
    print("Val losses: ", val_losses)
    
    np.save(f"{save_path}/Figs/Losses/{model_name}_pres_{pres}_epochs_{epochs}_lr_{learning_rate}_optim_{optim}_sched_{sched}"\
                    + f"_{'_'.join(sorted(train_seasons))}_train_losses{'_crps' if crps else ''}.npy", train_losses)
    np.save(f"{save_path}/Figs/Losses/{model_name}_pres_{pres}_epochs_{epochs}_lr_{learning_rate}_optim_{optim}_sched_{sched}"\
                    + f"_{'_'.join(sorted(train_seasons))}_val_losses{'_crps' if crps else ''}.npy", val_losses)
    model_name_save = model_name + "_crps" if crps else model_name
    
    torch.save(model, f"{save_path}/Models/{model_name_save}_pres_{pres}_epochs_{epochs}_lr_{learning_rate}_optim_{optim}_sched_{sched}"\
                    + f"_{'_'.join(sorted(train_seasons))}.pt")
    
    if not crps:
        crit = nn.MSELoss(reduction='none')
        plot_deterministic(model_name=model_name, train_seasons=train_seasons, val_seasons=val_seasons, test_seasons=test_seasons,
                           model=model, criterion=crit , train_losses=train_losses, val_losses=val_losses, test_set=test_set, test_loader=test_loader, 
                           lr=learning_rate, optim=optim, sched=sched, epochs=epochs, 
                           device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), save_path=save_path)
    else:
        crit = lambda x, y: CRPSLoss(x, y, reduction='none')
        plot_probabilistic(model_name=model_name, train_seasons=train_seasons, val_seasons=val_seasons, test_seasons=test_seasons,
                           model=model, criterion=crit , train_losses=train_losses, val_losses=val_losses, test_set=test_set, test_loader=test_loader,
                           lr=learning_rate, optim=optim, sched=sched, epochs=epochs,
                           device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), save_path=save_path)
    
    
def continue_training(data_path, model_name, seasons, split_ratio:list,
            epochs=30, learning_rate=0.001,
            optim="adam", sched=None, pres=True, crps=False,
            df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv",
            save_path="/users/lpoulain/louis/plots/cnn/"
            ):
    
    assert optim in optims_dict.keys(), f"optim must be in {optims_dict.keys()}, not {optim}"
    assert sched in scheds_dict.keys(), f"sched must be in {scheds_dict.keys()}, not {sched}"
    
    model = torch.load(f"{save_path}/Models/{model_name}{'_crps' if crps else ''}_pres_{pres}_epochs_{epochs}_lr_{learning_rate}_optim_{optim}_sched_{sched}"\
                                    + f"_{'_'.join(sorted(seasons))}.pt")
    
    optimizer = optims_dict[optim](model.parameters(), lr=learning_rate/2)
    scheduler = scheds_dict[sched](optimizer, epochs*2) if sched is not None else None
    
    train_seasons = seasons
    val_seasons = [s for s in ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007'] if s not in train_seasons]
    test_seasons = ['2008']
    
    deterministic = crps is False
    print(f"Train seasons: {train_seasons}\nVal seasons: {val_seasons}\nTest seasons: {test_seasons}")
    
    t = time()
    train_set = CNN4PP_Dataset(data_path, model_name, df_path, train_seasons, pres=pres, train_seasons=train_seasons)
    t_train = time()-t
    t = time()
    val_set = CNN4PP_Dataset(data_path, model_name, df_path, val_seasons, pres=pres, train_seasons=train_seasons)
    t_val = time()-t
    t = time()
    test_set = CNN4PP_Dataset(data_path, model_name, df_path, test_seasons, pres=pres, train_seasons=train_seasons)
    t_test = time()-t
    print(f"Dataset built in {t_train:.2f} (train) + {t_val:.2f} (val) + {t_test:.2f} (test) = {t_train+t_val+t_test:.2f} s")
    
    print("Len of train/val/test sets: ", len(train_set), len(val_set), len(test_set))
    
    train_loader = DataLoader(train_set, batch_size=1024, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=1024, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False, num_workers=4)
    print("Len of train/val/test loaders: ", len(train_loader), len(val_loader), len(test_loader))
    
    
    train_losses, val_losses = CNN_train(model=model, subset_train_loader=None, train_loader=train_loader,
                                         val_loader=val_loader, optimizer=optimizer, num_epochs=epochs*2, scheduler=scheduler, 
                                         deterministic=deterministic)
    print("Train losses: ", train_losses)
    print("Val losses: ", val_losses)
    
    old_train_losses = np.load(f"{save_path}/Figs/Losses/{model_name}_pres_{pres}_epochs_{epochs}_lr_{learning_rate}_optim_{optim}_sched_{sched}"\
                    + f"_{'_'.join(sorted(seasons))}_train_losses{'_crps' if crps else ''}.npy")
    old_val_losses = np.load(f"{save_path}/Figs/Losses/{model_name}_pres_{pres}_epochs_{epochs}_lr_{learning_rate}_optim_{optim}_sched_{sched}"\
                    + f"_{'_'.join(sorted(seasons))}_val_losses{'_crps' if crps else ''}.npy")
    
    train_losses = np.concatenate((old_train_losses, train_losses))
    val_losses = np.concatenate((old_val_losses, val_losses))
    
    np.save(f"{save_path}/Figs/Losses/{model_name}_pres_{pres}_epochs_{epochs*3}_lr_{learning_rate}_optim_{optim}_sched_{sched}"\
                    + f"_{'_'.join(sorted(train_seasons))}_train_losses{'_crps' if crps else ''}.npy", train_losses)
    np.save(f"{save_path}/Figs/Losses/{model_name}_pres_{pres}_epochs_{epochs*3}_lr_{learning_rate}_optim_{optim}_sched_{sched}"\
                    + f"_{'_'.join(sorted(train_seasons))}_val_losses{'_crps' if crps else ''}.npy", val_losses)
    model_name_save = model_name + "_crps" if crps else model_name
    
    torch.save(model, f"{save_path}/Models/{model_name_save}_pres_{pres}_epochs_{epochs*3}_lr_{learning_rate}_optim_{optim}_sched_{sched}"\
                    + f"_{'_'.join(sorted(train_seasons))}.pt")
    
    if not crps:
        crit = nn.MSELoss(reduction='none')
        plot_deterministic(model_name=model_name, train_seasons=train_seasons, val_seasons=val_seasons, test_seasons=test_seasons,
                           model=model, criterion=crit , train_losses=train_losses, val_losses=val_losses, test_set=test_set, test_loader=test_loader, 
                           lr=learning_rate, optim=optim, sched=sched, epochs=epochs*3, retrain=True,
                           device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), save_path=save_path)
    else:
        crit = lambda x, y: CRPSLoss(x, y, reduction='none')
        plot_probabilistic(model_name=model_name, train_seasons=train_seasons, val_seasons=val_seasons, test_seasons=test_seasons,
                           model=model, criterion=crit , train_losses=train_losses, val_losses=val_losses, test_set=test_set, test_loader=test_loader,
                           lr=learning_rate, optim=optim, sched=sched, epochs=epochs*3, retrain=True,
                           device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), save_path=save_path)
    
    

if __name__ == "__main__":
    
    parser = ArgumentParser()
    # paths
    parser.add_argument("--data_path", type=str, default="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/")
    parser.add_argument("--df_path", type=str, 
                        default="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv")
    parser.add_argument("--save_path", type=str, default="/users/lpoulain/louis/plots/cnn/")
    
    # data params
    parser.add_argument("--split_ratio", type=str2list, default=[0.7,0.2,0.1])
    
    # AI models params
    parser.add_argument("--model", type=str, default="graphcast")
    parser.add_argument("--seasons", type=str2list, default=["2000","2001","2002"], help="List of seasons to train/val/test on. If passed"\
                        + f" with --retrain, it is assumed to be the train_seasons only")
    parser.add_argument("--pres", help="whether to use pressure data", type=bool, default=True)
    
    # cnn model params
    parser.add_argument("--model_args", type=str2list, default=["lr","0.01","epochs","20"],
                        help="List of arguments for the model, in the form [arg1,val1,arg2,val2, ...]")
    #parser.add_argument("--type", type=str, default="mlp", choices=["mlp", "mlp_normal", "mlp_shash"])
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--sched", type=str, default="cosine_annealing")
    parser.add_argument("--crps", type=str2bool, default=False)
    parser.add_argument("--retrain", type=str2bool, default=False, help="continue training from a previous model")
    
    args = parser.parse_args()
    print("Input args:\n", args)
    
    # paths
    data_path, df_path, save_path = args.data_path, args.df_path, args.save_path
    
    # AI models params
    model, seasons, pres = args.model, args.seasons, args.pres
    
    # cnn method and inputs
    optim, sched = args.optim, args.sched
    crps = args.crps
    #mlp_type = args.type
    
    # specific mlp model args
    model_args = {"lr":0.01,"epochs":20}
    models_args_dtypes = {"epochs":int, "lr":float}
    
    for i in range(0, len(args.model_args), 2):
        model_args[args.model_args[i]] = models_args_dtypes[args.model_args[i]](args.model_args[i+1])
    
    print("Model args: ", model_args)
    
    # split ratio
    split_ratio = [float(ratio) for ratio in args.split_ratio]
    
    if not args.retrain:
        main_cnn(data_path=data_path, model_name=model, seasons=seasons, split_ratio=split_ratio,
             epochs=model_args["epochs"], learning_rate=model_args["lr"],
             optim=optim, sched=sched, pres=pres, crps=crps,
             df_path=df_path, save_path=save_path)
    else:
        continue_training(data_path=data_path, model_name=model, seasons=seasons, split_ratio=split_ratio,
             epochs=model_args["epochs"], learning_rate=model_args["lr"],
             optim=optim, sched=sched, pres=pres, crps=crps,
             df_path=df_path, save_path=save_path)