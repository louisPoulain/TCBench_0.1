import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import time, random, sys
from argparse import ArgumentParser

from scipy.special import erf
from torch.utils.data import DataLoader
from cnn_loaders import CNN4PP_Dataset, linear
from utils.main_utils import global_wind_bins, global_pres_bins
from matplotlib.colors import LogNorm

def plot_deterministic(model_name, train_seasons, val_seasons, test_seasons, model, criterion, train_losses, val_losses,
                       test_set, test_loader, lr, optim, sched, epochs, device, save_path, retrain=False):
    
    test_seasons = [str(test_seasons)] if not isinstance(test_seasons, list) else test_seasons
    loss_name = "MSE"
    
    fig, axs = plt.subplot_mosaic([["loss", "loss"]], figsize=(10, 10),
                                  gridspec_kw={"wspace": 0.30, "hspace": 0.3})
    
    axs['loss'].plot(train_losses, label="train losses")
    axs['loss'].plot(val_losses, label="val losses")
    axs['loss'].set_xlabel("Epoch")
    axs['loss'].set_ylabel(f"Loss ({loss_name})")
    axs['loss'].set_title(f"Losses for {model_name}")
    axs['loss'].legend()
    
    fig.suptitle(f"Model: {model_name} | Train: {', '.join(train_seasons)}, Val: {', '.join(val_seasons)},"\
               + f" Test: {', '.join(test_seasons)}\nCNN post-processing ({loss_name} loss)")
    fig.savefig(f"{save_path}/Figs/Losses/losses_{model_name}_{loss_name}_train_{'_'.join(sorted(train_seasons))}"+\
                f"_e_{epochs}_lr_{lr}_optim_{optim}_sched_{sched}{'_retrain' if retrain else ''}.png", dpi=500, bbox_inches='tight')
    plt.close(fig)
    
    indiv_losses = np.array([0., 0., 0., 0.])
    test_targets = {str(float(i)): [] for i in range(6, 174, 6)}
    test_preds = {str(float(i)): [] for i in range(6, 174, 6)}
    
    WIND_extent = test_set.wind_extent
    PRES_extent = test_set.pres_extent

    t = time.time()
    with torch.no_grad():
        for batch_idx, (fields, coords, targets) in enumerate(test_loader):
            fields, coords, targets = fields.float().to(device), coords.float().to(device), targets.float().to(device)
            
            lats, lons, ldts = coords[:, 0].cpu().numpy(), coords[:, 1].cpu().numpy(), coords[:, 2].cpu().numpy()
            lats = np.round(linear([min(WIND_extent[0], PRES_extent[0]), max(WIND_extent[1], PRES_extent[1])], [-90, 90], lats), 2)
            lons = np.round(linear([min(WIND_extent[0], PRES_extent[0]), max(WIND_extent[1], PRES_extent[1])], [0, 359.75], lons), 2)
            ldts = np.round(linear([min(WIND_extent[0], PRES_extent[0]), max(WIND_extent[1], PRES_extent[1])], [6., 168.], ldts), 1)
            
            pred = model(fields, coords)
            l = criterion(pred, targets).cpu()
            indiv_losses += np.array(l.mean(axis=0))
            
            targets, preds = targets.cpu().numpy(), pred.cpu().numpy()
            
            for i, ldt in enumerate(ldts):
                ldt = str(ldt)
                lat, lon = lats[i], lons[i]
                target = targets[i]
                pred = preds[i]
                # for wind/pres: var = var*std+mean
                # for lat/lon: var = var*divisor + input
                target_renorm = (target*test_set.target_std+test_set.target_mean + np.array([0, 0, lat, lon])).T
                pred_renorm = (pred*test_set.target_std+test_set.target_mean + np.array([0, 0, lat, lon])).T
                test_targets[str(float(ldt))].append(target_renorm)
                test_preds[str(float(ldt))].append(pred_renorm)
                
                
            if batch_idx%(max(len(test_loader)//5, 1))==0:
                print(f"Batch {batch_idx}/{len(test_loader)} ({time.time()-t:.2f} s)")
            
    indiv_losses /= len(test_loader)
    print(indiv_losses)
    test_loss = indiv_losses
    test_targets = {key: np.array(val) for key, val in test_targets.items()}
    test_preds = {key: np.array(val) for key, val in test_preds.items()}
    
    test_mse = {key: np.sqrt(np.mean((test_targets[key] - test_preds[key])**2, axis=0)) for key in test_targets.keys()}
    test_mae = {key: np.mean(np.abs(test_targets[key] - test_preds[key]), axis=0) for key in test_targets.keys()}
    test_r2 = {key: 1 - np.sum((test_targets[key] - test_preds[key])**2, axis=0) / np.sum((test_targets[key] - np.mean(test_targets[key], axis=0))**2, axis=0)\
                for key in test_targets.keys()}
    
    for ldt in test_mse.keys():
        
        data_targets = test_targets[ldt]
        data_preds = test_preds[ldt]
        mse, mae, r2 = test_mse[ldt], test_mae[ldt], test_r2[ldt]
        
        plot_hist2d(data_preds, data_targets, model_name, train_seasons, val_seasons, test_seasons, test_loss, 
                    loss_name, epochs, lr, optim, sched, ldt, mse, mae, r2, retrain=retrain)
        plot_hist1d(data_preds, data_targets, model_name, train_seasons, val_seasons, test_seasons, test_loss,
                    loss_name, epochs, lr, optim, sched, ldt, mse, mae, r2, retrain=retrain)
    
    
    
    
def plot_probabilistic(model_name, train_seasons, val_seasons, test_seasons, model, criterion, train_losses, val_losses,
                       test_set, test_loader, lr, optim, sched, epochs, device, save_path, retrain=False):
    
    test_seasons = [str(test_seasons)] if not isinstance(test_seasons, list) else test_seasons
    loss_name = "CRPS"
    
    fig, axs = plt.subplot_mosaic([["loss", "loss"]], figsize=(10, 10),
                                  gridspec_kw={"wspace": 0.30, "hspace": 0.3})
    
    
    param_names = ["wind", "pres", r"$\Delta$"+"lat", r"$\Delta$"+"lon"]
    colors = ["red", "blue", "green", "orange"]
    for i in range(4):
        axs['loss'].plot(train_losses[:, i], label=f"Train loss ({param_names[i]})", c=colors[i])
        axs['loss'].plot(val_losses[:, i], label=f"Val loss ({param_names[i]})", c=colors[i], linestyle='--')
    axs['loss'].set_xlabel("Epoch")
    axs['loss'].set_ylabel(f"Loss ({loss_name})")
    axs['loss'].set_title(f"Losses for {model_name}")
    axs['loss'].legend()
    
    fig.suptitle(f"Model: {model_name} | Train: {', '.join(train_seasons)}, Val: {', '.join(val_seasons)},"\
               + f" Test: {', '.join(test_seasons)}\nCNN post-processing ({loss_name} loss)")
    fig.savefig(f"{save_path}/Figs/Losses/losses_{model_name}_{loss_name}_train_{'_'.join(sorted(train_seasons))}"+\
                f"_e_{epochs}_lr_{lr}_optim_{optim}_sched_{sched}{'_retrain' if retrain else ''}.png", dpi=500, bbox_inches='tight')
    plt.close(fig)
    
    test_loss = 0.0
    indiv_losses = 0.0
    test_targets = {str(float(i)): [] for i in range(6, 174, 6)}
    test_preds = {str(float(i)): [] for i in range(6, 174, 6)}
    
    WIND_extent = test_set.wind_extent
    PRES_extent = test_set.pres_extent

    t = time.time()
    with torch.no_grad():
        for batch_idx, (fields, coords, targets) in enumerate(test_loader):
            fields, coords, targets = fields.float().to(device), coords.float().to(device), targets.float().to(device)
            
            lats, lons, ldts = coords[:, 0].cpu().numpy(), coords[:, 1].cpu().numpy(), coords[:, 2].cpu().numpy()
            lats = np.round(linear([min(WIND_extent[0], PRES_extent[0]), max(WIND_extent[1], PRES_extent[1])], [-90, 90], lats), 2)
            lons = np.round(linear([min(WIND_extent[0], PRES_extent[0]), max(WIND_extent[1], PRES_extent[1])], [0, 359.75], lons), 2)
            ldts = np.round(linear([min(WIND_extent[0], PRES_extent[0]), max(WIND_extent[1], PRES_extent[1])], [6., 168.], ldts), 1)
            
            pred = model(fields, coords)
            l, indiv_l = criterion(pred, targets)
            test_loss += l.cpu().mean().item()
            indiv_losses += indiv_l.mean(axis=0)
            
            targets, preds = targets.cpu().numpy(), pred.cpu().numpy()
            
            for i, ldt in enumerate(ldts):
                ldt = str(ldt)
                lat, lon = lats[i], lons[i]
                targets_means = targets[i] 
                preds_means = preds[i][::2] # we get only the mean currently
                targets_renorm = targets_means*test_set.target_std+test_set.target_mean + np.array([0, 0, lat, lon])
                preds_renorm = preds_means*test_set.target_std+test_set.target_mean + np.array([0, 0, lat, lon])
                test_targets[ldt].append(targets_renorm) 
                test_preds[ldt].append(preds_renorm)
                
            if batch_idx%(max(len(test_loader)//5, 1))==0:
                print(f"Batch {batch_idx}/{len(test_loader)} ({time.time()-t:.2f} s)")
    
    print("Test loss (global): ", test_loss/len(test_loader))
    indiv_losses /= len(test_loader)
    print("Test loss (individual): ", indiv_losses)
    test_loss = indiv_losses
    test_targets = {key: np.array(val) for key, val in test_targets.items()}
    test_preds = {key: np.array(val) for key, val in test_preds.items()}
    
    test_rmse = {key: np.sqrt(np.mean((test_targets[key] - test_preds[key])**2, axis=0)) for key in test_targets.keys()}
    test_mae = {key: np.mean(np.abs(test_targets[key] - test_preds[key]), axis=0) for key in test_targets.keys()}
    test_r2 = {key: 1 - np.sum((test_targets[key] - test_preds[key])**2, axis=0) / np.sum((test_targets[key] - np.mean(test_targets[key], axis=0))**2, axis=0)\
                for key in test_targets.keys()}
    
    #test_rmse = {key: np.sqrt(np.mean(MySquaredError(test_targets[key], test_preds[key]), axis=0)) for key in test_targets.keys()}
    #test_mae = {key: np.mean(MyAbsoluteError(test_targets[key], test_preds[key]), axis=0) for key in test_targets.keys()}
    #test_r2 = {key: 1 - np.sum(MySquaredError(test_targets[key], test_preds[key]), axis=0) / np.sum(MySquaredError(test_targets[key], np.mean(test_targets[key], axis=0)), axis=0)\
    #            for key in test_targets.keys()}
    
    
    for ldt in test_rmse.keys():
        
        print("Lead time: ", int(float(ldt)))
        data_targets = test_targets[ldt]
        data_preds = test_preds[ldt]
        rmse, mae, r2 = test_rmse[ldt], test_mae[ldt], test_r2[ldt]
        
        plot_hist2d(data_preds, data_targets, model_name, train_seasons, val_seasons, test_seasons, test_loss, 
                    loss_name, epochs, lr, optim, sched, ldt, rmse, mae, r2, retrain=retrain)
        plot_hist1d(data_preds, data_targets, model_name, train_seasons, val_seasons, test_seasons, test_loss,
                    loss_name, epochs, lr, optim, sched, ldt, rmse, mae, r2, retrain=retrain)
        
        
        
def plot_hist2d(data_preds, data_targets, model_name, train_seasons, val_seasons, test_seasons, test_loss, loss_name,
                epochs, lr, optim, sched, ldt, rmse, mae, r2, retrain=False,
                save_path="/users/lpoulain/louis/plots/cnn/"):
    
    fig, axs = plt.subplot_mosaic([["wind","pres"],
                                   ["lat", "lon"]], figsize=(14, 9), gridspec_kw={"wspace": 0.3, "hspace": 0.3})
    
    # define bins
    bins_wind = global_wind_bins
    bins_pres = global_pres_bins
    r_lat = 1.5 # this is only for visualisation purposes
    r_lon = 6
    bins_lat = np.arange(-90-r_lat/2, 90+r_lat, r_lat)
    bins_lon = np.arange(-r_lon/2, 360+r_lon, r_lon)
    
    # define histograms
    hist_wind, x_e_wind, y_e_wind = np.histogram2d(data_preds[:, 0], data_targets[:, 0], bins=(bins_wind, bins_wind))
    x_wind, y_wind = np.meshgrid(x_e_wind, y_e_wind)
    
    hist_pres, x_e_pres, y_e_pres = np.histogram2d(data_preds[:, 1], data_targets[:, 1], bins=(bins_pres, bins_pres))
    x_pres, y_pres = np.meshgrid(x_e_pres, y_e_pres)
    
    hist_lat, x_e_lat, y_e_lat = np.histogram2d(data_preds[:, 2], data_targets[:, 2], bins=(bins_lat, bins_lat))
    x_lat, y_lat = np.meshgrid(x_e_lat, y_e_lat)
    
    hist_lon, x_e_lon, y_e_lon = np.histogram2d(data_preds[:, 3], data_targets[:, 3], bins=(bins_lon, bins_lon))
    x_lon, y_lon = np.meshgrid(x_e_lon, y_e_lon)
    
        
    wnd = axs['wind'].pcolormesh(x_wind, y_wind, hist_wind.T, cmap='plasma', norm=LogNorm())
    cbar = plt.colorbar(wnd, label='Density')
    wind_extent = [min(data_preds[:, 0].min(), data_targets[:, 0].min()),
                max(data_preds[:, 0].max(), data_targets[:, 0].max())]
    axs['wind'].plot(wind_extent, wind_extent, color="black", label="Identity")
    axs['wind'].set_xlim(wind_extent)
    axs['wind'].set_ylim(wind_extent)
    axs['wind'].set_xlabel("Post-processed wind (m/s)")
    axs['wind'].set_ylabel("Observed wind (m/s)")
    axs['wind'].set_title(f"Wind density histogram\nRMSE: {rmse[0]:.2f} m/s | MAE: {mae[0]:.2f} m/s | R2: {r2[0]:.3f}")
    
    
    prs = axs['pres'].pcolormesh(x_pres, y_pres, hist_pres.T, cmap='plasma', norm=LogNorm())
    cbar = plt.colorbar(prs, label='Density')
    pres_extent = [min(data_preds[:, 1].min(), data_targets[:, 1].min()),
                max(data_preds[:, 1].max(), data_targets[:, 1].max())]
    axs['pres'].plot(pres_extent, pres_extent, color="black", label="Identity")
    axs['pres'].set_xlim(pres_extent)
    axs['pres'].set_ylim(pres_extent)
    axs['pres'].set_xlabel("Post-processed pressure (Pa)")
    axs['pres'].set_ylabel("Observed pressure (Pa)")
    axs['pres'].set_title(f"Pressure density histogram\nRMSE: {rmse[1]:.2f} Pa | MAE: {mae[1]:.2f} Pa | R2: {r2[1]:.3f}")
    
    
    lat = axs['lat'].pcolormesh(x_lat, y_lat, hist_lat.T, cmap='plasma', norm=LogNorm())
    cbar = plt.colorbar(lat, label='Density')
    lat_extent = [min(data_preds[:, 2].min(), data_targets[:, 2].min()),
                    max(data_preds[:, 2].max(), data_targets[:, 2].max())]
    axs['lat'].plot(lat_extent, lat_extent, color="black", label="Identity", alpha=0.5, linestyle='--')
    axs['lat'].set_xlim(lat_extent)
    axs['lat'].set_ylim(lat_extent)
    axs['lat'].set_xlabel("Post-processed latitude (° north)")
    axs['lat'].set_ylabel("Observed latitude (° north)")
    axs['lat'].set_title(f"Latitude density histogram\nRMSE: {rmse[2]:.2f}° | MAE: {mae[2]:.2f}° | R2: {r2[2]:.3f}")
    
    
    lon = axs['lon'].pcolormesh(x_lon, y_lon, hist_lon.T, cmap='plasma', norm=LogNorm())
    cbar = plt.colorbar(lon, label='Density')
    lon_extent = [min(data_preds[:, 3].min(), data_targets[:, 3].min()),
                    max(data_preds[:, 3].max(), data_targets[:, 3].max())]
    axs['lon'].plot(lon_extent, lon_extent, color="black", label="Identity", alpha=0.5, linestyle='--')
    axs['lon'].set_xlim(lon_extent)
    axs['lon'].set_ylim(lon_extent)
    axs['lon'].set_xlabel("Post-processed longitude (° east)")
    axs['lon'].set_ylabel("Observed longitude (° east)")
    axs['lon'].set_title(f"Longitude density histogram\nRMSE: {rmse[3]:.2f}° | MAE: {mae[3]:.2f}° | R2: {r2[3]:.3f}")
    

    fig.suptitle(f"Model: {model_name} | Train: {', '.join(train_seasons)}, Val: {', '.join(val_seasons)},"\
            + f" Test: {', '.join(test_seasons)}\nCNN post-processing ({loss_name} loss) - Lead time: {int(float(ldt))}h")
    fig.savefig(f"{save_path}/Figs/Histograms/{int(float(ldt))}h/hist2d_{model_name}_ldt_{int(float(ldt))}_{loss_name}_train_{'_'.join(sorted(train_seasons))}"+\
            f"_e_{epochs}_lr_{lr}_optim_{optim}_sched_{sched}{'_retrain' if retrain else ''}.png", dpi=500, bbox_inches='tight')
    plt.close(fig)
    
    
def plot_hist1d(data_preds, data_targets, model_name, train_seasons, val_seasons, test_seasons, test_loss, loss_name,
                epochs, lr, optim, sched, ldt, rmse, mae, r2, retrain=False,
                save_path="/users/lpoulain/louis/plots/cnn/"):
    
    fig, axs = plt.subplot_mosaic([["wind","pres"],
                                    ["lat", "lon"]], figsize=(14, 9), gridspec_kw={"wspace": 0.20, "hspace": 0.3})
    
    # define bins
    bins_wind = global_wind_bins
    bins_pres = global_pres_bins
    r_lat = 1.5 # this is only for visualisation purposes
    r_lon = 6
    bins_lat = np.arange(-90-r_lat/2, 90+r_lat, r_lat)
    bins_lon = np.arange(-r_lon/2, 360+r_lon, r_lon)
    
    # define histograms
    
    axs['wind'].hist(data_targets[:, 0], bins=bins_wind, label="Observed wind distribution", alpha=0.5)
    axs['wind'].hist(data_preds[:, 0], bins=bins_wind, label="Post-processed wid distribution", alpha=0.7)
    wind_extent = [min(data_preds[:, 0].min(), data_targets[:, 0].min()),
                   max(data_preds[:, 0].max(), data_targets[:, 0].max())]
    axs['wind'].set_xlim(wind_extent)
    axs['wind'].set_xlabel("Wind (m/s)")
    axs['wind'].set_ylabel("Occurences")
    axs['wind'].set_title(f"Wind density histogram\nRMSE: {rmse[0]:.2f} m/s | MAE: {mae[0]:.2f} m/s | R2: {r2[0]:.3f}")
    axs['wind'].legend()
    
    
    axs['pres'].hist(data_targets[:, 1], bins=bins_pres, label="Observed pressure distribution", alpha=0.5)
    axs['pres'].hist(data_preds[:, 1], bins=bins_pres, label="Post-processed pressure distribution", alpha=0.7)
    pres_extent = [min(data_preds[:, 1].min(), data_targets[:, 1].min()),
                   max(data_preds[:, 1].max(), data_targets[:, 1].max())]
    axs['pres'].set_xlim(pres_extent)
    axs['pres'].set_xlabel("Pressure (Pa)")
    axs['pres'].set_ylabel("Occurences")
    axs['pres'].set_title(f"Pressure density histogram\nRMSE: {rmse[1]:.2f} Pa | MAE: {mae[1]:.2f} Pa | R2: {r2[1]:.3f}")
    axs['pres'].legend()
    
    
    axs['lat'].hist(data_targets[:, 2], bins=bins_lat, label="Observed latitude distribution", alpha=0.5)
    axs['lat'].hist(data_preds[:, 2], bins=bins_lat, label="Post-processed latitude distribution", alpha=0.7)
    lat_extent = [min(data_preds[:, 2].min(), data_targets[:, 2].min()),
                  max(data_preds[:, 2].max(), data_targets[:, 2].max())]
    axs['lat'].set_xlim(lat_extent)
    axs['lat'].set_xlabel("Latitude (° north)")
    axs['lat'].set_ylabel("Occurences")
    axs['lat'].set_title(f"Latitude density histogram\nRMSE: {rmse[2]:.2f}° | MAE: {mae[2]:.2f}° | R2: {r2[2]:.3f}")
    axs['lat'].legend()
    
    
    axs['lon'].hist(data_targets[:, 3], bins=bins_lon, label="Observed longitude distribution", alpha=0.5)
    axs['lon'].hist(data_preds[:, 3], bins=bins_lon, label="Post-processed longitude distribution", alpha=0.7)
    lon_extent = [min(data_preds[:, 3].min(), data_targets[:, 3].min()),
                  max(data_preds[:, 3].max(), data_targets[:, 3].max())]
    axs['lon'].set_xlim(lon_extent)
    axs['lon'].set_xlabel("Longitude (° east)")
    axs['lon'].set_ylabel("Occurences")
    axs['lon'].set_title(f"Longitude density histogram\nRMSE: {rmse[3]:.2f}° | MAE: {mae[3]:.2f}° | R2: {r2[3]:.3f}")
    axs['lon'].legend()
    
    
    fig.suptitle(f"Model: {model_name} | Train: {', '.join(train_seasons)}, Val: {', '.join(val_seasons)},"\
            + f" Test: {', '.join(test_seasons)}\nCNN post-processing ({loss_name} loss) - Lead time: {int(float(ldt))}h")
    fig.savefig(f"{save_path}/Figs/Histograms/{int(float(ldt))}h/hist1d_{model_name}_ldt_{int(float(ldt))}_{loss_name}_train_{'_'.join(sorted(train_seasons))}"+\
            f"_e_{epochs}_lr_{lr}_optim_{optim}_sched_{sched}{'_retrain' if retrain else ''}.png", dpi=500, bbox_inches='tight')
    plt.close(fig)


def CRPSNormal(y_pred, y_true, reduction="mean"):
    # taken from https://github.com/WillyChap/ARML_Probabilistic/blob/main/Coastal_Points/Testing_and_Utility_Notebooks/CRPS_Verify.ipynb#54dc5b06b84f0a023e394ae24e9e0e1ea49301e1
    
    mu = y_pred[:, 0]
    sigma = y_pred[:, 1]
    
    # prevent negative sigmas
    sigma = torch.sqrt(sigma.pow(2))
    
    loc = (y_true - mu) / sigma
    pdf = torch.exp(-0.5 * loc.pow(2)) / torch.sqrt(2 * torch.from_numpy(np.array(np.pi)))
    cdf = 0.5 * (1.0 + torch.erf(loc / torch.sqrt(torch.tensor(2.))))
    
    # compute CRPS for each (input, truth) pair
    crps = sigma * (loc * (2. * cdf - 1.) + 2. * pdf -1. / torch.from_numpy(np.array(np.sqrt(np.pi))))
    
    return crps.mean() if reduction=="mean" else crps



def CRPSLoss(y_pred, y_true, reduction='mean'):
    
    loss = 0.
    indiv_losses = np.empty((y_true.shape))
    
    for i in range(0, y_true.shape[1]):
        l = CRPSNormal(y_pred[:, i*2:(i+1)*2], y_true[:, i], reduction=reduction)
        loss += l
        indiv_losses[:, i] = l.detach().cpu().numpy()
    
    return loss / (y_true.shape[1] // 2), indiv_losses


def CRPSNormal_np(mu, sigma, y, reduction='mean'):
        sigma = np.sqrt(sigma**2)
        loc = (y - mu) / sigma
        pdf = np.exp(-0.5 * loc**2) / np.sqrt(2 * np.pi)
        cdf = 0.5 * (1.0 + erf(loc / np.sqrt(2)))
        crps = sigma * (loc * (2. * cdf - 1.) + 2. * pdf -1. / np.sqrt(np.pi))
        return crps.mean() if reduction=="mean" else crps
    

def CRPSNumpy(mu_pred, sigma_pred, y_true, reduction="mean"):
    
    indiv_losses = np.empty((y_true.shape))
    
    for i in range(0, y_true.shape[1]):
        l = CRPSNormal_np(mu_pred[:, i], sigma_pred[:, i], y_true[:, i], reduction=reduction)
        indiv_losses[:, i] = l
    
    return indiv_losses


def MySquaredError(x, y, kt=False):
    if kt:
        return ((np.abs(x-y)-2.5) ** 2) * (np.abs(x-y) > 2.5)
    else:
        return ((np.abs(x-y)-2.5*0.514444) ** 2) * (np.abs(x-y) > 2.5*0.514444)
    
    
def MyAbsoluteError(x, y, kt=False):
    if kt:
        return np.abs(x-y) * (np.abs(x-y) > 2.5)
    else:
        return (np.abs(x-y)-2.5*0.514444) * (np.abs(x-y) > 2.5*0.514444)
        

def load_and_plot(model_name, pres, epochs, learning_rate, optim, sched, train_losses, val_losses, device, 
                  train_seasons, val_seasons, test_seasons=2008, crps=False, small=False,
                  save_path='/users/lpoulain/louis/plots/cnn',
                  data_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/",
                  df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv"):
    
    model = torch.load(f"{save_path}/Models/{model_name}{'_crps' if crps else ''}_pres_{pres}_epochs_{epochs}_lr_{learning_rate}_optim_{optim}"\
                     + f"_sched_{sched}_{'_'.join(train_seasons)}.pt", map_location=device)
    
    test_set = CNN4PP_Dataset(data_path, model_name, df_path, seasons=test_seasons, pres=pres, train_seasons=train_seasons, create_input=False)
    if not small:
        test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=1)
    else:
        test_loader = DataLoader(torch.utils.data.Subset(test_set, random.sample(range(0, len(test_set)), len(test_set)//50)),
                            batch_size=512, shuffle=False, num_workers=1)
    if crps:
        criterion = lambda x,y: CRPSLoss(x, y, reduction='none')
    else:
        criterion = nn.MSELoss(reduction='none')
    
    if not crps:
        plot_deterministic(model_name=model_name, train_seasons=train_seasons, val_seasons=val_seasons, test_seasons=test_seasons, model=model, 
                           criterion=criterion, train_losses=train_losses, val_losses=val_losses, test_set=test_set, test_loader=test_loader, 
                           lr=learning_rate, optim=optim, sched=sched, epochs=epochs, device=device, save_path=save_path)
    else:
        plot_probabilistic(model_name=model_name, train_seasons=train_seasons, val_seasons=val_seasons, test_seasons=test_seasons, model=model,
                           criterion=criterion, train_losses=train_losses, val_losses=val_losses, test_set=test_set, test_loader=test_loader,
                           lr=learning_rate, optim=optim, sched=sched, epochs=epochs, device=device, save_path=save_path)
    
    
    
if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--small", action="store_true", help="Use a smaller test set")
    parser.add_argument("--crps", action="store_true", help="Use CRPS loss instead of MSE")
    args = parser.parse_args()
    
    small = args.small
    crps = args.crps
    
    model_name = 'graphcast'
    pres = True
    epochs = 90
    learning_rate = 0.01
    optim = 'adam'
    sched = 'cosine_annealing'

    device = "cpu"
    
    train_seasons = sorted(['2000','2002','2003','2004','2006','2007'])
    val_seasons = ['2001', '2005']
    test_seasons = 2008
    
    train_losses = np.load(f"/users/lpoulain/louis/plots/cnn/Figs/Losses/{model_name}_pres_{pres}_epochs_{epochs}_lr_{learning_rate}_optim_{optim}_sched_{sched}"\
                    + f"_{'_'.join(sorted(train_seasons))}_train_losses{'_crps' if crps else ''}.npy")
                    
    val_losses = np.load(f"/users/lpoulain/louis/plots/cnn/Figs/Losses/{model_name}_pres_{pres}_epochs_{epochs}_lr_{learning_rate}_optim_{optim}_sched_{sched}"\
                    + f"_{'_'.join(sorted(train_seasons))}_val_losses{'_crps' if crps else ''}.npy")

    load_and_plot(model_name=model_name, pres=pres, epochs=epochs, learning_rate=learning_rate, optim=optim, sched=sched, train_losses=train_losses, 
                  val_losses=val_losses, device=device, train_seasons=train_seasons, val_seasons=val_seasons, test_seasons=test_seasons, small=small,
                  crps=crps)