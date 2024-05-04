from cnn_loaders import CNN4PP_Dataset, linear
from cnn_blocks import CNN4PP
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from cnn_utils import CRPSLoss, CRPSNumpy
from helpers_comparison_cnn import get_pit_points, plot_pit_dict, calculate_spread_skill

import torch, time, random, sys
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt



def final_test_deterministic(model_name, pres, epochs, learning_rate, optim, sched, device, 
                train_seasons, val_seasons, test_seasons=2008, small=False,
                save_path='/users/lpoulain/louis/plots/cnn/',
                data_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/",
                df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv"):
    
    model = torch.load(f"{save_path}/Models/{model_name}_pres_{pres}_epochs_{epochs}_lr_{learning_rate}_optim_{optim}"\
                     + f"_sched_{sched}_{'_'.join(train_seasons)}.pt", map_location=device)
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    
    train_set = CNN4PP_Dataset(data_path, model_name, df_path, seasons=train_seasons, pres=pres, train_seasons=train_seasons, create_input=False)
    val_set = CNN4PP_Dataset(data_path, model_name, df_path, seasons=val_seasons, pres=pres, train_seasons=train_seasons, create_input=False)
    test_set = CNN4PP_Dataset(data_path, model_name, df_path, seasons=test_seasons, pres=pres, train_seasons=train_seasons, create_input=False)
    
    if not small:
        train_loader = DataLoader(train_set, batch_size=1024, shuffle=False, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=1024, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=1024, shuffle=False, num_workers=4)
    else:
        train_loader = DataLoader(torch.utils.data.Subset(train_set, random.sample(range(0, len(train_set)), len(train_set)//80)), 
                                batch_size=1024, shuffle=False, num_workers=1)
        val_loader = DataLoader(torch.utils.data.Subset(val_set, random.sample(range(0, len(val_set)), len(val_set)//80)),
                                batch_size=1024, shuffle=False, num_workers=1)
        test_loader = DataLoader(torch.utils.data.Subset(test_set, random.sample(range(0, len(test_set)), len(test_set)//80)),
                                batch_size=1024, shuffle=False, num_workers=1)
    
    
    train_targets = {str(float(i)): [] for i in range(6, 174, 6)}
    val_targets = {str(float(i)): [] for i in range(6, 174, 6)}
    test_targets = {str(float(i)): [] for i in range(6, 174, 6)}
    
    train_preds_cnn = {str(float(i)): [] for i in range(6, 174, 6)}
    val_preds_cnn = {str(float(i)): [] for i in range(6, 174, 6)}
    test_preds_cnn = {str(float(i)): [] for i in range(6, 174, 6)}
    
    train_preds_model = {str(float(i)): [] for i in range(6, 174, 6)}
    val_preds_model = {str(float(i)): [] for i in range(6, 174, 6)}
    test_preds_model = {str(float(i)): [] for i in range(6, 174, 6)}
    
    WIND_extent = test_set.wind_extent
    PRES_extent = test_set.pres_extent
    
    indiv_losses_train = np.zeros(4)
    indiv_losses_val = np.zeros(4)
    indiv_losses_test = np.zeros(4)

    t = time.time()
    with torch.no_grad():
        for batch_idx, (fields, coords, targets) in enumerate(train_loader):
            fields, coords, targets = fields.float().to(device), coords.float().to(device), targets.float().to(device)
            
            lats, lons, ldts = coords[:, 0].cpu().numpy(), coords[:, 1].cpu().numpy(), coords[:, 2].cpu().numpy()
            lats = np.round(linear([min(WIND_extent[0], PRES_extent[0]), max(WIND_extent[1], PRES_extent[1])], [-90, 90], lats), 2)
            lons = np.round(linear([min(WIND_extent[0], PRES_extent[0]), max(WIND_extent[1], PRES_extent[1])], [0, 359.75], lons), 2)
            ldts = np.round(linear([min(WIND_extent[0], PRES_extent[0]), max(WIND_extent[1], PRES_extent[1])], [6., 168.], ldts), 1)
            fields_no_renorm = fields.cpu().numpy()*train_set.std + train_set.mean
            
            pred = model(fields, coords)
            l = criterion(pred, targets).cpu()
            indiv_losses_train += np.array(l.mean(axis=0))
            
            targets, preds = targets.cpu().numpy(), pred.cpu().numpy()
            
            for i, ldt in enumerate(ldts):
                ldt = str(ldt)
                lat, lon = lats[i], lons[i]
                target = targets[i]
                pred_cnn = preds[i]
                pred_model = np.array([fields_no_renorm[i][0].max(), fields_no_renorm[i][1].min()])
                
                target_renorm = (target*train_set.target_std+train_set.target_mean + np.array([0, 0, lat, lon])).T
                pred_renorm = (pred_cnn*train_set.target_std+train_set.target_mean + np.array([0, 0, lat, lon])).T
                
                train_targets[str(float(ldt))].append(target_renorm)
                train_preds_cnn[str(float(ldt))].append(pred_renorm)
                train_preds_model[str(float(ldt))].append(pred_model)
                
            if batch_idx%(max(len(train_loader)//5, 1))==0:
                print(f"Batch {batch_idx}/{len(train_loader)} ({time.time()-t:.2f} s)")
                
        print(f"Train done in {time.time()-t:.2f} s")
        t = time.time()
        
        for batch_idx, (fields, coords, targets) in enumerate(val_loader):
            fields, coords, targets = fields.float().to(device), coords.float().to(device), targets.float().to(device)
            
            lats, lons, ldts = coords[:, 0].cpu().numpy(), coords[:, 1].cpu().numpy(), coords[:, 2].cpu().numpy()
            lats = np.round(linear([min(WIND_extent[0], PRES_extent[0]), max(WIND_extent[1], PRES_extent[1])], [-90, 90], lats), 2)
            lons = np.round(linear([min(WIND_extent[0], PRES_extent[0]), max(WIND_extent[1], PRES_extent[1])], [0, 359.75], lons), 2)
            ldts = np.round(linear([min(WIND_extent[0], PRES_extent[0]), max(WIND_extent[1], PRES_extent[1])], [6., 168.], ldts), 1)
            fields_no_renorm = fields.cpu().numpy()*val_set.std + val_set.mean
            
            pred = model(fields, coords)
            l = criterion(pred, targets).cpu()
            indiv_losses_val += np.array(l.mean(axis=0))
            
            targets, preds = targets.cpu().numpy(), pred.cpu().numpy()
            
            for i, ldt in enumerate(ldts):
                ldt = str(ldt)
                lat, lon = lats[i], lons[i]
                target = targets[i]
                pred_cnn = preds[i]
                pred_model = np.array([fields_no_renorm[i][0].max(), fields_no_renorm[i][1].min()])
                
                target_renorm = (target*val_set.target_std + val_set.target_mean + np.array([0, 0, lat, lon])).T
                pred_renorm = (pred_cnn*val_set.target_std + val_set.target_mean + np.array([0, 0, lat, lon])).T
                
                val_targets[str(float(ldt))].append(target_renorm)
                val_preds_cnn[str(float(ldt))].append(pred_renorm)
                val_preds_model[str(float(ldt))].append(pred_model)
                
            if batch_idx%(max(len(val_loader)//5, 1))==0:
                print(f"Batch {batch_idx}/{len(val_loader)}")
        
        print(f"Val done in {time.time()-t:.2f} s")
        t = time.time()
        
        for batch_idx, (fields, coords, targets) in enumerate(test_loader):
            fields, coords, targets = fields.float().to(device), coords.float().to(device), targets.float().to(device)
            
            lats, lons, ldts = coords[:, 0].cpu().numpy(), coords[:, 1].cpu().numpy(), coords[:, 2].cpu().numpy()
            lats = np.round(linear([min(WIND_extent[0], PRES_extent[0]), max(WIND_extent[1], PRES_extent[1])], [-90, 90], lats), 2)
            lons = np.round(linear([min(WIND_extent[0], PRES_extent[0]), max(WIND_extent[1], PRES_extent[1])], [0, 359.75], lons), 2)
            ldts = np.round(linear([min(WIND_extent[0], PRES_extent[0]), max(WIND_extent[1], PRES_extent[1])], [6., 168.], ldts), 1)
            fields_no_renorm = fields.cpu().numpy()*test_set.std + test_set.mean
            
            pred = model(fields, coords)
            l = criterion(pred, targets).cpu()
            indiv_losses_test += np.array(l.mean(axis=0))
            
            targets, preds = targets.cpu().numpy(), pred.cpu().numpy()
            
            for i, ldt in enumerate(ldts):
                ldt = str(ldt)
                lat, lon = lats[i], lons[i]
                target = targets[i]
                pred_cnn = preds[i]
                pred_model = np.array([fields_no_renorm[i][0].max(), fields_no_renorm[i][1].min()])
                
                target_renorm = (target*test_set.target_std + test_set.target_mean + np.array([0, 0, lat, lon])).T
                pred_renorm = (pred_cnn*test_set.target_std + test_set.target_mean + np.array([0, 0, lat, lon])).T
                
                test_targets[str(float(ldt))].append(target_renorm)
                test_preds_cnn[str(float(ldt))].append(pred_renorm)
                test_preds_model[str(float(ldt))].append(pred_model)
    
    print(f"Test done in {time.time()-t:.2f} s")        
    
    indiv_losses_train /= len(train_loader)
    indiv_losses_val /= len(val_loader)
    indiv_losses_test /= len(test_loader)
    
    
    train_targets = {key: np.array(val) for key, val in train_targets.items()}
    train_preds_cnn = {key: np.array(val) for key, val in train_preds_cnn.items()}
    train_preds_model = {key: np.array(val) for key, val in train_preds_model.items()}
    
    val_targets = {key: np.array(val) for key, val in val_targets.items()}
    val_preds_cnn = {key: np.array(val) for key, val in val_preds_cnn.items()}
    val_preds_model = {key: np.array(val) for key, val in val_preds_model.items()}
    
    test_targets = {key: np.array(val) for key, val in test_targets.items()}
    test_preds_cnn = {key: np.array(val) for key, val in test_preds_cnn.items()}
    test_preds_model = {key: np.array(val) for key, val in test_preds_model.items()}
    
    
    train_mse_cnn = {key: np.sqrt(np.mean((train_targets[key] - train_preds_cnn[key])**2, axis=0)) for key in train_targets.keys()}
    train_mae_cnn = {key: np.mean(np.abs(train_targets[key] - train_preds_cnn[key]), axis=0) for key in train_targets.keys()}
    train_r2_cnn = {key: 1 - np.sum((train_targets[key] - train_preds_cnn[key])**2, axis=0) / np.sum((train_targets[key] - np.mean(train_targets[key], axis=0))**2, axis=0)\
                for key in train_targets.keys()}
    train_mse_model = {key: np.sqrt(np.mean((train_targets[key][:, :2] - train_preds_model[key])**2, axis=0)) for key in train_targets.keys()}
    train_mae_model = {key: np.mean(np.abs(train_targets[key][:, :2] - train_preds_model[key]), axis=0) for key in train_targets.keys()}
    train_r2_model = {key: 1 - np.sum((train_targets[key][:, :2] - train_preds_model[key])**2, axis=0) / np.sum((train_targets[key][:, :2] - np.mean(train_targets[key][:, :2], axis=0))**2, axis=0)\
                for key in train_targets.keys()}
    
    val_mse_cnn = {key: np.sqrt(np.mean((val_targets[key] - val_preds_cnn[key])**2, axis=0)) for key in val_targets.keys()}
    val_mae_cnn = {key: np.mean(np.abs(val_targets[key] - val_preds_cnn[key]), axis=0) for key in val_targets.keys()}
    val_r2_cnn = {key: 1 - np.sum((val_targets[key] - val_preds_cnn[key])**2, axis=0) / np.sum((val_targets[key] - np.mean(val_targets[key], axis=0))**2, axis=0)\
                for key in val_targets.keys()}
    val_mse_model = {key: np.sqrt(np.mean((val_targets[key][:, :2] - val_preds_model[key])**2, axis=0)) for key in val_targets.keys()}
    val_mae_model = {key: np.mean(np.abs(val_targets[key][:, :2] - val_preds_model[key]), axis=0) for key in val_targets.keys()}
    val_r2_model = {key: 1 - np.sum((val_targets[key][:, :2] - val_preds_model[key])**2, axis=0) / np.sum((val_targets[key][:, :2] - np.mean(val_targets[key][:, :2], axis=0))**2, axis=0)\
                for key in val_targets.keys()}
    
    test_mse_cnn = {key: np.sqrt(np.mean((test_targets[key] - test_preds_cnn[key])**2, axis=0)) for key in test_targets.keys()}
    test_mae_cnn = {key: np.mean(np.abs(test_targets[key] - test_preds_cnn[key]), axis=0) for key in test_targets.keys()}
    test_r2_cnn = {key: 1 - np.sum((test_targets[key] - test_preds_cnn[key])**2, axis=0) / np.sum((test_targets[key] - np.mean(test_targets[key], axis=0))**2, axis=0)\
                for key in test_targets.keys()}
    test_mse_model = {key: np.sqrt(np.mean((test_targets[key][:, :2] - test_preds_model[key])**2, axis=0)) for key in test_targets.keys()}
    test_mae_model = {key: np.mean(np.abs(test_targets[key][:, :2] - test_preds_model[key]), axis=0) for key in test_targets.keys()}
    test_r2_model = {key: 1 - np.sum((test_targets[key][:, :2] - test_preds_model[key])**2, axis=0) / np.sum((test_targets[key][:, :2] - np.mean(test_targets[key][:, :2], axis=0))**2, axis=0)\
                for key in test_targets.keys()}
    
    for ldt in train_targets.keys():
        with open(f"{save_path}/Final_comp/{model_name}_MSE_pres_{pres}_ldt_{int(float(ldt))}_epochs_{epochs}_lr_{learning_rate}_optim_{optim}"\
                        + f"_sched_{sched}_{'_'.join(train_seasons)}.txt", 'w') as f:
            f.write(f"Train losses: {np.round(indiv_losses_train, 3)}\n")
            f.write(f"Val losses: {np.round(indiv_losses_val, 3)}\n")
            f.write(f"Test losses: {np.round(indiv_losses_test, 3)}\n")
            f.write(f"\n")
            f.write(f"Train MSE CNN: {np.round(train_mse_cnn[ldt], 3)}\n")
            f.write(f"Train MAE CNN: {np.round(train_mae_cnn[ldt], 3)}\n")
            f.write(f"Train R2 CNN: {np.round(train_r2_cnn[ldt], 3)}\n")
            f.write(f"Train MSE Model: {np.round(train_mse_model[ldt], 3)}\n")
            f.write(f"Train MAE Model: {np.round(train_mae_model[ldt], 3)}\n")
            f.write(f"Train R2 Model: {np.round(train_r2_model[ldt], 3)}\n")
            f.write(f"\n")
            f.write(f"Val MSE CNN: {np.round(val_mse_cnn[ldt], 3)}\n")
            f.write(f"Val MAE CNN: {np.round(val_mae_cnn[ldt], 3)}\n")
            f.write(f"Val R2 CNN: {np.round(val_r2_cnn[ldt], 3)}\n")
            f.write(f"Val MSE Model: {np.round(val_mse_model[ldt], 3)}\n")
            f.write(f"Val MAE Model: {np.round(val_mae_model[ldt], 3)}\n")
            f.write(f"Val R2 Model: {np.round(val_r2_model[ldt], 3)}\n")
            f.write(f"\n")
            f.write(f"Test MSE CNN: {np.round(test_mse_cnn[ldt], 3)}\n")
            f.write(f"Test MAE CNN: {np.round(test_mae_cnn[ldt], 3)}\n")
            f.write(f"Test R2 CNN: {np.round(test_r2_cnn[ldt], 3)}\n")
            f.write(f"Test MSE Model: {np.round(test_mse_model[ldt], 3)}\n")
            f.write(f"Test MAE Model: {np.round(test_mae_model[ldt], 3)}\n")
            f.write(f"Test R2 Model: {np.round(test_r2_model[ldt], 3)}\n")
    
    
    
def final_test_probabilistic(model_name, pres, epochs, learning_rate, optim, sched, device, 
                train_seasons, val_seasons, test_seasons=2008, small=False,
                save_path='/users/lpoulain/louis/plots/cnn/',
                data_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/",
                df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv"):
    
    model = torch.load(f"{save_path}/Models/{model_name}_crps_pres_{pres}_epochs_{epochs}_lr_{learning_rate}_optim_{optim}"\
                     + f"_sched_{sched}_{'_'.join(train_seasons)}.pt", map_location=device)
    model.eval()
    
    criterion = lambda x, y: CRPSLoss(x, y, reduction='none')
    
    train_set = CNN4PP_Dataset(data_path, model_name, df_path, seasons=train_seasons, pres=pres, train_seasons=train_seasons, create_input=False)
    val_set = CNN4PP_Dataset(data_path, model_name, df_path, seasons=val_seasons, pres=pres, train_seasons=train_seasons, create_input=False)
    test_set = CNN4PP_Dataset(data_path, model_name, df_path, seasons=test_seasons, pres=pres, train_seasons=train_seasons, create_input=False)
    
    if not small:
        train_loader = DataLoader(train_set, batch_size=1024, shuffle=False, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=1024, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=1024, shuffle=False, num_workers=4)
    else:
        train_loader = DataLoader(torch.utils.data.Subset(train_set, random.sample(range(0, len(train_set)), len(train_set)//200)), 
                                batch_size=1024, shuffle=False, num_workers=1)
        val_loader = DataLoader(torch.utils.data.Subset(val_set, random.sample(range(0, len(val_set)), len(val_set)//100)),
                                batch_size=1024, shuffle=False, num_workers=1)
        test_loader = DataLoader(torch.utils.data.Subset(test_set, random.sample(range(0, len(test_set)), len(test_set)//80)),
                                batch_size=1024, shuffle=False, num_workers=1)
        
    train_targets = {str(float(i)): [] for i in range(6, 174, 6)}
    val_targets = {str(float(i)): [] for i in range(6, 174, 6)}
    test_targets = {str(float(i)): [] for i in range(6, 174, 6)}
    
    train_preds_mean = {str(float(i)): [] for i in range(6, 174, 6)}
    train_preds_std = {str(float(i)): [] for i in range(6, 174, 6)}
    val_preds_mean = {str(float(i)): [] for i in range(6, 174, 6)}
    val_preds_std = {str(float(i)): [] for i in range(6, 174, 6)}
    test_preds_mean = {str(float(i)): [] for i in range(6, 174, 6)}
    test_preds_std = {str(float(i)): [] for i in range(6, 174, 6)}
    
    WIND_extent = train_set.wind_extent
    PRES_extent = train_set.pres_extent
    
    indiv_losses_train = np.zeros(4)
    indiv_losses_val = np.zeros(4)
    indiv_losses_test = np.zeros(4)
    
    t = time.time()
    
    with torch.no_grad():
        for batch_idx, (fields, coords, targets) in enumerate(train_loader):
            fields, coords, targets = fields.float().to(device), coords.float().to(device), targets.float().to(device)
            
            lats, lons, ldts = coords[:, 0].cpu().numpy(), coords[:, 1].cpu().numpy(), coords[:, 2].cpu().numpy()
            lats = np.round(linear([min(WIND_extent[0], PRES_extent[0]), max(WIND_extent[1], PRES_extent[1])], [-90, 90], lats), 2)
            lons = np.round(linear([min(WIND_extent[0], PRES_extent[0]), max(WIND_extent[1], PRES_extent[1])], [0, 359.75], lons), 2)
            ldts = np.round(linear([min(WIND_extent[0], PRES_extent[0]), max(WIND_extent[1], PRES_extent[1])], [6., 168.], ldts), 1)
            
            pred = model(fields, coords)
            l, indiv_l = criterion(pred, targets)
            indiv_losses_train += np.array(indiv_l.mean(axis=0))
            targets, preds = targets.cpu().numpy(), pred.cpu().numpy()
            
            for i, ldt in enumerate(ldts):
                ldt = str(ldt)
                lat, lon = lats[i], lons[i]
                target = targets[i]
                pred_mean_cnn = preds[i][::2]
                pred_std_cnn = preds[i][1::2]
                
                
                target_renorm = (target*train_set.target_std+train_set.target_mean + np.array([0, 0, lat, lon])).T
                pred_mean_renorm = (pred_mean_cnn*train_set.target_std+train_set.target_mean + np.array([0, 0, lat, lon])).T
                pred_std_renorm = np.sqrt((pred_std_cnn*train_set.target_std)**2).T
                
                train_targets[str(float(ldt))].append(target_renorm)
                train_preds_mean[str(float(ldt))].append(pred_mean_renorm)
                train_preds_std[str(float(ldt))].append(pred_std_renorm)
                
                
            if batch_idx%(max(len(train_loader)//5, 1))==0:
                print(f"Batch {batch_idx}/{len(train_loader)} ({time.time()-t:.2f} s)")
                
        print(f"Train done in {time.time()-t:.2f} s")
        t = time.time()
        
        for batch_idx, (fields, coords, targets) in enumerate(val_loader):
            fields, coords, targets = fields.float().to(device), coords.float().to(device), targets.float().to(device)
            
            lats, lons, ldts = coords[:, 0].cpu().numpy(), coords[:, 1].cpu().numpy(), coords[:, 2].cpu().numpy()
            lats = np.round(linear([min(WIND_extent[0], PRES_extent[0]), max(WIND_extent[1], PRES_extent[1])], [-90, 90], lats), 2)
            lons = np.round(linear([min(WIND_extent[0], PRES_extent[0]), max(WIND_extent[1], PRES_extent[1])], [0, 359.75], lons), 2)
            ldts = np.round(linear([min(WIND_extent[0], PRES_extent[0]), max(WIND_extent[1], PRES_extent[1])], [6., 168.], ldts), 1)
            
            pred = model(fields, coords)
            l, indiv_l = criterion(pred, targets)
            indiv_losses_val += np.array(indiv_l.mean(axis=0))
            targets, preds = targets.cpu().numpy(), pred.cpu().numpy()
            
            for i, ldt in enumerate(ldts):
                ldt = str(ldt)
                lat, lon = lats[i], lons[i]
                target = targets[i]
                pred_mean_cnn = preds[i][::2]
                pred_std_cnn = preds[i][1::2]
                
                target_renorm = (target*val_set.target_std + val_set.target_mean + np.array([0, 0, lat, lon])).T
                pred_mean_renorm = (pred_mean_cnn*val_set.target_std + val_set.target_mean + np.array([0, 0, lat, lon])).T
                pred_std_renorm = np.sqrt((pred_std_cnn*val_set.target_std)**2).T
                
                val_targets[str(float(ldt))].append(target_renorm)
                val_preds_mean[str(float(ldt))].append(pred_mean_renorm)
                val_preds_std[str(float(ldt))].append(pred_std_renorm)
                
            if batch_idx%(max(len(val_loader)//5, 1))==0:
                print(f"Batch {batch_idx}/{len(val_loader)} ({time.time()-t:.2f} s)")
                
        print(f"Val done in {time.time()-t:.2f} s")
        t = time.time()
        
        for batch_idx, (fields, coords, targets) in enumerate(test_loader):
            fields, coords, targets = fields.float().to(device), coords.float().to(device), targets.float().to(device)
            
            lats, lons, ldts = coords[:, 0].cpu().numpy(), coords[:, 1].cpu().numpy(), coords[:, 2].cpu().numpy()
            lats = np.round(linear([min(WIND_extent[0], PRES_extent[0]), max(WIND_extent[1], PRES_extent[1])], [-90, 90], lats), 2)
            lons = np.round(linear([min(WIND_extent[0], PRES_extent[0]), max(WIND_extent[1], PRES_extent[1])], [0, 359.75], lons), 2)
            ldts = np.round(linear([min(WIND_extent[0], PRES_extent[0]), max(WIND_extent[1], PRES_extent[1])], [6., 168.], ldts), 1)
            
            pred = model(fields, coords)
            l, indiv_l = criterion(pred, targets)
            indiv_losses_test += np.array(indiv_l.mean(axis=0))
            targets, preds = targets.cpu().numpy(), pred.cpu().numpy()
            
            for i, ldt in enumerate(ldts):
                ldt = str(ldt)
                lat, lon = lats[i], lons[i]
                target = targets[i]
                pred_mean_cnn = preds[i][::2]
                pred_std_cnn = preds[i][1::2]
                
                target_renorm = (target*test_set.target_std + test_set.target_mean + np.array([0, 0, lat, lon])).T
                pred_mean_renorm = (pred_mean_cnn*test_set.target_std + test_set.target_mean + np.array([0, 0, lat, lon])).T
                pred_std_renorm = np.sqrt((pred_std_cnn*test_set.target_std)**2).T
                
                test_targets[str(float(ldt))].append(target_renorm)
                test_preds_mean[str(float(ldt))].append(pred_mean_renorm)
                test_preds_std[str(float(ldt))].append(pred_std_renorm)
            
            if batch_idx%(max(len(test_loader)//5, 1))==0:
                print(f"Batch {batch_idx}/{len(test_loader)} ({time.time()-t:.2f} s)")
                
        print(f"Test done in {time.time()-t:.2f} s")
    
    indiv_losses_train /= len(train_loader)
    indiv_losses_val /= len(val_loader)
    indiv_losses_test /= len(test_loader)
    
    train_targets = {key: np.array(val) for key, val in train_targets.items()}
    train_preds_mean = {key: np.array(val) for key, val in train_preds_mean.items()}
    train_preds_std = {key: np.array(val) for key, val in train_preds_std.items()}
    
    val_targets = {key: np.array(val) for key, val in val_targets.items()}
    val_preds_mean = {key: np.array(val) for key, val in val_preds_mean.items()}
    val_preds_std = {key: np.array(val) for key, val in val_preds_std.items()}
    
    test_targets = {key: np.array(val) for key, val in test_targets.items()}
    test_preds_mean = {key: np.array(val) for key, val in test_preds_mean.items()}
    test_preds_std = {key: np.array(val) for key, val in test_preds_std.items()}
    
    train_rmse_cnn = {key: np.sqrt(np.mean((train_targets[key] - train_preds_mean[key])**2, axis=0)) for key in train_targets.keys()}
    val_rmse_cnn = {key: np.sqrt(np.mean((val_targets[key] - val_preds_mean[key])**2, axis=0)) for key in val_targets.keys()}
    test_rmse_cnn = {key: np.sqrt(np.mean((test_targets[key] - test_preds_mean[key])**2, axis=0)) for key in test_targets.keys()}
    
    
    train_crps_cnn = {}
    val_crps_cnn = {}
    test_crps_cnn = {}
    spread_skill_train = {}
    spread_skill_val = {}
    spread_skill_test = {}
    pit_pts_train = {}
    pit_pts_val = {}
    pit_pts_test = {}
    
    
    criterion = lambda x,y,z: CRPSNumpy(x,y,z, reduction='none')
    
    for ldt in train_targets.keys():
        m_train, std_train = train_preds_mean[ldt], train_preds_std[ldt]
        m_val, std_val = val_preds_mean[ldt], val_preds_std[ldt]
        m_test, std_test = test_preds_mean[ldt], test_preds_std[ldt]
        
        train_crps_cnn[ldt] = criterion(m_train, std_train, train_targets[ldt])
        val_crps_cnn[ldt] = criterion(m_val, std_val, val_targets[ldt])
        test_crps_cnn[ldt] = criterion(m_test, std_test, test_targets[ldt])
        
        spread_skill_train[ldt] = np.array([[calculate_spread_skill(train_targets[ldt][:, i], m_train[:, i], std_train[:, i], train_rmse_cnn[ldt][i])] for i in range(4)]).squeeze()
        spread_skill_val[ldt] = np.array([[calculate_spread_skill(val_targets[ldt][:, i], m_val[:, i], std_val[:, i], val_rmse_cnn[ldt][i])] for i in range(4)]).squeeze()
        spread_skill_test[ldt] = np.array([[calculate_spread_skill(test_targets[ldt][:, i], m_test[:, i], std_test[:, i], test_rmse_cnn[ldt][i])] for i in range(4)]).squeeze()

        pit_pts_train[ldt] = [get_pit_points(train_targets[ldt][:, i], m_train[:, i], std_train[:, i]) for i in range(4)]
        if int(float(ldt)) == 6:
            print([pit_pts_train[ldt][i]['pit_counts'] for i in range(4)])
        pit_pts_val[ldt] = [get_pit_points(val_targets[ldt][:, i], m_val[:, i], std_val[:, i]) for i in range(4)]
        pit_pts_test[ldt] = [get_pit_points(test_targets[ldt][:, i], m_test[:, i], std_test[:, i]) for i in range(4)]
    
    for ldt in train_targets.keys():
        #with open(f"{save_path}/Final_comp/{model_name}_CRPS_pres_{pres}_ldt_{int(float(ldt))}_epochs_{epochs}_lr_{learning_rate}_optim_{optim}"\
        #                + f"_sched_{sched}_{'_'.join(train_seasons)}.txt", 'w') as f:
        #    
        #    f.write(f"Train losses: {np.round(indiv_losses_train, 3)}\n")
        #    f.write(f"Val losses: {np.round(indiv_losses_val, 3)}\n")
        #    f.write(f"Test losses: {np.round(indiv_losses_test, 3)}\n")
        #    f.write(f"\n")
        #    f.write(f"Train RMSE CNN (ensemble mean error): {np.round(train_rmse_cnn[ldt], 3)}\n")
        #    f.write(f"Val RMSE CNN (ensemble mean error): {np.round(val_rmse_cnn[ldt], 3)}\n")
        #    f.write(f"Test RMSE CNN (ensemble mean error): {np.round(test_rmse_cnn[ldt], 3)}\n")
        #    f.write(f"\n")
        #    f.write(f"Train CRPS CNN (mean CRPS): {np.round(np.mean(train_crps_cnn[ldt], axis=0), 3)}\n")
        #    f.write(f"Val CRPS CNN (mean CRPS): {np.round(np.mean(val_crps_cnn[ldt], axis=0), 3)}\n")
        #    f.write(f"Test CRPS CNN (mean CRPS): {np.round(np.mean(test_crps_cnn[ldt], axis=0), 3)}\n")
        #    f.write(f"\n")
        
        fig, axs = plt.subplot_mosaic([['Wind', 'Pressure'], ['Latitude', 'Longitude']], figsize=(15,15), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
        
        ss_train = spread_skill_train[ldt]
        ss_val = spread_skill_val[ldt]
        ss_test = spread_skill_test[ldt]
        
        axs['Wind'].plot(ss_train[0,0][ss_train[0,0]>-999], ss_train[0,1][ss_train[0,1]>-999], label='Train', color='blue', marker='x')
        axs['Wind'].plot(ss_val[0,0][ss_val[0,0]>-999], ss_val[0,1][ss_val[0,1]>-999], label='Val', color='orange', marker='o')
        axs['Wind'].plot(ss_test[0,0][ss_test[0,0]>-999], ss_test[0,1][ss_test[0,1]>-999], label='Test', color='green', marker='D')
        axs['Wind'].plot(axs['Wind'].get_xlim(), axs['Wind'].get_xlim(), label='y=x', color='black', linestyle='-', alpha=0.5)
        axs['Wind'].set_title(f"Wind speed", fontsize=18)
        axs['Wind'].set_xlabel('Spread (uncertainty, m/s)', fontsize=17)
        axs['Wind'].set_ylabel('Skill (RMSE, m/s)', fontsize=17)
        axs['Wind'].annotate("Overconfident model", xy=(0.5, 0.95), xycoords='axes fraction', ha='center', fontsize=18, color='black')
        axs['Wind'].annotate("Underconfident model", xy=(0.5, 0.02), xycoords='axes fraction', ha='center', fontsize=18, color='black')
        axs['Wind'].legend(fontsize=17)
        
        axs['Pressure'].plot(ss_train[1,0][ss_train[1,0]>-999], ss_train[1,1][ss_train[1,1]>-999], label='Train', color='blue', marker='x')
        axs['Pressure'].plot(ss_val[1,0][ss_val[1,0]>-999], ss_val[1,1][ss_val[1,1]>-999], label='Val', color='orange', marker='o')
        axs['Pressure'].plot(ss_test[1,0][ss_test[1,0]>-999], ss_test[1,1][ss_test[1,1]>-999], label='Test', color='green', marker='D')
        axs['Pressure'].plot(axs['Pressure'].get_xlim(), axs['Pressure'].get_xlim(), label='y=x', color='black', linestyle='-', alpha=0.5)
        axs['Pressure'].set_title(f"Minimum sea-level pressure", fontsize=18)
        axs['Pressure'].set_xlabel('Spread (uncertainty, Pa)', fontsize=17)
        axs['Pressure'].set_ylabel('Skill (RMSE, Pa)', fontsize=17)
        axs['Pressure'].annotate("Overconfident model", xy=(0.5, 0.95), xycoords='axes fraction', ha='center', fontsize=18, color='black')
        axs['Pressure'].annotate("Underconfident model", xy=(0.5, 0.02), xycoords='axes fraction', ha='center', fontsize=18, color='black')
        axs['Pressure'].legend(fontsize=17)
        
        axs['Latitude'].plot(ss_train[2,0][ss_train[2,0]>-999], ss_train[2,1][ss_train[2,1]>-999], label='Train', color='blue', marker='x')
        axs['Latitude'].plot(ss_val[2,0][ss_val[2,0]>-999], ss_val[2,1][ss_val[2,1]>-999], label='Val', color='orange', marker='o')
        axs['Latitude'].plot(ss_test[2,0][ss_test[2,0]>-999], ss_test[2,1][ss_test[2,1]>-999], label='Test', color='green', marker='D')
        axs['Latitude'].plot(axs['Latitude'].get_xlim(), axs['Latitude'].get_xlim(), label='y=x', color='black', linestyle='-', alpha=0.5)
        axs['Latitude'].set_title(f"Latitude", fontsize=18)
        axs['Latitude'].set_xlabel('Spread (uncertainty, degrees north)', fontsize=17)
        axs['Latitude'].set_ylabel('Skill (RMSE, degrees north)', fontsize=17)
        axs['Latitude'].annotate("Overconfident model", xy=(0.5, 0.95), xycoords='axes fraction', ha='center', fontsize=18, color='black')
        axs['Latitude'].annotate("Underconfident model", xy=(0.5, 0.02), xycoords='axes fraction', ha='center', fontsize=18, color='black')
        axs['Latitude'].legend(fontsize=17)
        
        axs['Longitude'].plot(ss_train[3,0][ss_train[3,0]>-999], ss_train[3,1][ss_train[3,1]>-999], label='Train', color='blue', marker='x')
        axs['Longitude'].plot(ss_val[3,0][ss_val[3,0]>-999], ss_val[3,1][ss_val[3,1]>-999], label='Val', color='orange', marker='o')
        axs['Longitude'].plot(ss_test[3,0][ss_test[3,0]>-999], ss_test[3,1][ss_test[3,1]>-999], label='Test', color='green', marker='D')
        axs['Longitude'].plot(axs['Longitude'].get_xlim(), axs['Longitude'].get_xlim(), label='y=x', color='black', linestyle='-', alpha=0.5)
        axs['Longitude'].set_title(f"Longitude", fontsize=18)
        axs['Longitude'].set_xlabel('Spread (uncertainty, degrees east)', fontsize=17)
        axs['Longitude'].set_ylabel('Skill (RMSE, degrees east)', fontsize=17)
        axs['Longitude'].annotate("Overconfident model", xy=(0.5, 0.95), xycoords='axes fraction', ha='center', fontsize=18, color='black')
        axs['Longitude'].annotate("Underconfident model", xy=(0.5, 0.02), xycoords='axes fraction', ha='center', fontsize=18, color='black')
        axs['Longitude'].legend(fontsize=17)
        
        fig.suptitle(f"Spread-skill for {model_name} at {int(float(ldt))}h", fontsize=20)
        #fig.savefig(f"{save_path}/Final_comp/{model_name}_spread_skill_pres_{pres}_ldt_{int(float(ldt))}_epochs_{epochs}_lr_{learning_rate}_optim_{optim}"\
        #                + f"_sched_{sched}_{'_'.join(train_seasons)}.pdf")
        plt.close(fig)
        
        dict_pits = {"Train": {"Wind": pit_pts_train[ldt][0],
                               "Pres": pit_pts_train[ldt][1],
                               "Lat": pit_pts_train[ldt][2],
                               "Lon": pit_pts_train[ldt][3]},
                     "Val": {"Wind": pit_pts_val[ldt][0],
                             "Pres": pit_pts_val[ldt][1],
                             "Lat": pit_pts_val[ldt][2],
                             "Lon": pit_pts_val[ldt][3]},
                     "Test": {"Wind": pit_pts_test[ldt][0],
                              "Pres": pit_pts_test[ldt][1],
                              "Lat": pit_pts_test[ldt][2],
                              "Lon": pit_pts_test[ldt][3]}
                    }
        for c in ["Train", "Val", "Test"]:
            dict_pit = dict_pits[c]
            if c=='Train' and int(float(ldt))==6:
                print([dict_pit[key]['pit_centers'] for key in dict_pit.keys()])
            plot_pit_dict(dict_pit, bar_label=["Wind", "Pres", "Lat", "Lon"], title=f"PIT points for {model_name} at lead time {int(float(ldt))}h ({c})",
                      model=model_name, ldt=int(float(ldt)), split=c)
            
    print("finished")        
      
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    parser = ArgumentParser()
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--crps", action="store_true")
    
    args = parser.parse_args()
    small = args.small
    crps = args.crps
    
    model_name = "graphcast"
    pres = True
    epochs = 90
    learning_rate = 0.01
    optim = "adam"
    sched = "cosine_annealing"
    
    train_seasons = ['2000', '2001', '2002', '2004', '2005', '2007']
    val_seasons = ['2003', '2006']
    test_seasons = '2008'
    
    if not crps:
        final_test_deterministic(model_name, pres, epochs, learning_rate, optim, sched, device, 
                train_seasons, val_seasons, test_seasons=test_seasons, small=small,
                save_path='/users/lpoulain/louis/plots/cnn/',
                data_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/",
                df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv")
    else:
        final_test_probabilistic(model_name, pres, epochs, learning_rate, optim, sched, device, 
                train_seasons, val_seasons, test_seasons=test_seasons, small=small,
                save_path='/users/lpoulain/louis/plots/cnn/',
                data_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/",
                df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv")
    
