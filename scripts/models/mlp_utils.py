from loading_utils import statistics_loader, stats_list, stats_fcts
from xgboost_utils import split_data_new
from utils.main_utils import multiline_label
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys


    

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight.data, 1)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data, gain=init.calculate_gain('leaky_relu', 0.01))
        init.constant_(m.bias.data, 0)
        


def create_dataset_mlp(data_path, model_name, lead_time, seasons, split_ratio:list,
            stats=[], stats_wind=["max"], stats_pres=["min"],
            df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv",
            save_path="/users/lpoulain/louis/plots/mlp/"
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
                                                                                                    seasons[0], df_path, save_path)
    
    if len(seasons)>1:
        for season in seasons[1:]:
            tmp_wind_stats, tmp_pres_stats, tmp_wnd_truth, tmp_pres_truth, tmp_ids = statistics_loader(data_path, model_name, lead_time,
                                                                                                    season, df_path, save_path)
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
    
    dtrain_wind, dval_wind, dtest_wind,\
    dtrain_pres, dval_pres, dtest_pres,\
    truth_wind_train, truth_wind_val, truth_wind_test,\
    truth_pres_train, truth_pres_val, truth_pres_test = split_data(idx_train, idx_val, stats_wind_list, stats_pres_list, 
                                                                   truth_wind_list, truth_pres_list, stats_wind, stats_pres)
    
    mean_wind, std_wind = np.mean(dtrain_wind, axis=0, keepdims=True), np.std(dtrain_wind, axis=0, keepdims=True)
    mean_pres, std_pres = np.mean(dtrain_pres, axis=0, keepdims=True), np.std(dtrain_pres, axis=0, keepdims=True)
    dtrain_wind, dval_wind, dtest_wind = (dtrain_wind-mean_wind)/std_wind, (dval_wind-mean_wind)/std_wind, (dtest_wind-mean_wind)/std_wind
    dtrain_pres, dval_pres, dtest_pres = (dtrain_pres-mean_pres)/std_pres, (dval_pres-mean_pres)/std_pres, (dtest_pres-mean_pres)/std_pres

    return dtrain_wind, dval_wind, dtest_wind, dtrain_pres, dval_pres, dtest_pres, truth_wind_train,\
            truth_wind_val, truth_wind_test, truth_pres_train, truth_pres_val, truth_pres_test,\
            stats, stats_wind, stats_pres, nb_tc
            
            
            
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout_rate=0.0):
        super(MLP, self).__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_units[0]))
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Tanh())
        
        for i in range(1, len(hidden_units)):
            layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_units[-1], 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.model(x)
        return x
    
    

class MLP_normal(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout_rate=0.0):
        super(MLP_normal, self).__init__()
        self.epsilon = 1e-6

        layers = []
        layers.append(nn.Linear(input_dim, hidden_units[0]))
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.LeakyReLU())
        
        for i in range(1, len(hidden_units)):
            layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.LeakyReLU())
        
        layers.append(nn.Linear(hidden_units[-1], 2))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.model(x)
        x = torch.cat((x[:, 0].view(x.shape[0],1), nn.ReLU()(x[:, 1].view(x.shape[0],1))+self.epsilon), dim=1)
        return x


class MLP_shash(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout_rate=0.0):
        super(MLP_shash, self).__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_units[0]))
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Tanh())
        
        for i in range(1, len(hidden_units)):
            layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_units[-1], 4))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.model(x)
        return x


def NormalLossMLP(outputs, targets):
    loss = nn.GaussianNLLLoss()(outputs[:, 0], targets, outputs[:, 1])
    return loss


def LogShash(loc, scale, skew, tau, y):
    # see Jones, M. C. & Pewsey, A., Sinh-arcsinh distributions,
    # Biometrika, Oxford University Press, 2009, 96, 761-780.
    # DOI: 10.1093/biomet/asp053
    # p762. eq 1 and 2 + transform with loc and scale
    # here: loc = xi, scale = eta, skew = eps, tau = delta
    
    val = (y-loc)/scale
    S = lambda x: torch.sinh(tau*torch.asinh(x)-skew)
    C = lambda x: torch.cosh(tau*torch.asinh(x)-skew)
    first = lambda x: (torch.pi*2*(1 + x.pow(2))).pow(-1/2) * tau
    second = lambda x: tau*torch.asinh(x)-skew
    third = lambda x: torch.clamp(torch.exp(-S(x).pow(2)/2), 1e-10)
    # Shash(x) = first(x) * cosh(second(x)) * third(x)
    # with loc and scale: Shash(y) = 1 / scale * first(y) * cosh(second(y)) * third(y) , with y = (x-loc)/scale
    # to avoid numerical issues with log(cosh), we use logaddexp (log(cosh(x)) = logaddexp(x, -x) - log(2))
    return torch.log(1/scale) + torch.log(first(val)) +\
           torch.logaddexp(second(val), -second(val)) - torch.log(torch.tensor(2.)) +\
           torch.log(third(val))

def ShashLossMLP(outputs, targets):
    
    loc = outputs[:, 0].view(-1,1)
    scale = torch.clamp(outputs[:, 1], min=1e-6).view(-1,1)
    skew = outputs[:, 2].view(-1,1)
    tau = torch.clamp(outputs[:, 3], min=1e-6).view(-1,1)
    
    loss = -LogShash(loc, scale, skew, tau, targets)
    
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        print(torch.cat((loc, scale, skew, tau), dim=1))
        print(targets)
        print(loss)
        print("Nan/Inf in ShashLossMLP")
        sys.exit()

    return loss.mean()


def ShashMoments(loc, scale, skew, tau):
    from shash_helpers import mean, stddev
    
    Mean = mean(loc, scale, skew, tau)
    Std = stddev(loc, scale, skew, tau)
    
    return Mean, Std



def mlp_dataloader(dtrain, dval, dtest, truth_train, truth_val, truth_test, batch_size):
    
    dtrain_dataset = torch.utils.data.TensorDataset(torch.from_numpy(dtrain), torch.from_numpy(truth_train.reshape(-1,1)))
    dtrain_loader = DataLoader(dtrain_dataset, batch_size=batch_size, shuffle=True)
    
    dval_dataset = torch.utils.data.TensorDataset(torch.from_numpy(dval), torch.from_numpy(truth_val.reshape(-1,1)))
    dval_loader = DataLoader(dval_dataset, batch_size=batch_size, shuffle=False)
    
    dtest_dataset = torch.utils.data.TensorDataset(torch.from_numpy(dtest), torch.from_numpy(truth_test.reshape(-1,1)))
    dtest_loader = DataLoader(dtest_dataset, batch_size=1, shuffle=False)
    
    return dtrain_loader, dval_loader, dtest_loader



def train_mlp(model, train_loader, val_loader, criterion, optimizer, num_epochs, scheduler=None, var="wind"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_losses = []
    val_losses = []

    
    # train for 2 epochs to avoid problems with weights init
    for e in range(2):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.float().to(device), targets.float().to(device)
            optimizer.zero_grad()
            
            
            outputs = model(inputs)
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("Nan/Inf in train_mlp")
                print("old out", old_out)
                print("old grad", old_grad)
                print("old loss", loss)
                print("inputs", inputs)
                sys.exit()
            old_out = outputs.detach().clone()
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            old_grad = [param.grad for param in model.parameters()]
    
    with trange(1, num_epochs + 1, desc=f'Training {var}', unit='epoch') as t:
        for epoch in t:
            model.train()
            running_loss = 0.0
            
            with tqdm(train_loader, desc=f'Train epoch {epoch}',
              unit='batch', leave=False) as t1:
                for batch_idx, (inputs, targets) in enumerate(t1):
                    inputs, targets = inputs.float().to(device), targets.float().to(device)
                    optimizer.zero_grad()
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)
            
            # Calculate loss on validation dataset
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                with tqdm(val_loader, desc=f'Val. epoch {epoch}',
                unit='batch', leave=False) as t2:
                    for batch_idx, (inputs, targets) in enumerate(t2):
                        inputs, targets = inputs.float().to(device), targets.float().to(device)
                        val_outputs = model(inputs)
                        val_loss += criterion(val_outputs, targets).item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Update scheduler if provided
            if scheduler is not None:
                scheduler.step()

    return train_losses, val_losses



def plot_mlp(mlp_wind, mlp_pres, train_losses, val_losses, test_loader_wind, test_loader_pres, model_name, seasons, num_tcs,
             stats = [], stats_wind = [], stats_pres = [], mlp_params={}, lead_time=0,
             save_path="/users/lpoulain/louis/plots/mlp/"):
    
    plot_fct_dict = {"mlp": plot_mlp_deterministic, "mlp_normal": plot_mlp_normal, "mlp_shash": plot_mlp_shash}
    mlp_type = mlp_params['mlp_type']
    
    plot_fct_dict[mlp_type](mlp_wind, mlp_pres, train_losses, val_losses, test_loader_wind, test_loader_pres, model_name, seasons, num_tcs,
                            stats = stats, stats_wind = stats_wind, stats_pres = stats_pres, mlp_params=mlp_params, lead_time=lead_time,
                            save_path=save_path)
    


def plot_mlp_deterministic(mlp_wind, mlp_pres, train_losses, val_losses, test_loader_wind, test_loader_pres, model_name, seasons, num_tcs,
             stats = [], stats_wind = [], stats_pres = [], mlp_params={}, lead_time=0,
             save_path="/users/lpoulain/louis/plots/mlp/"):
    
    lr, layers, epochs, sched = mlp_params['lr'], mlp_params['layers'], mlp_params['epochs'], mlp_params['scheduler']
    if len(stats)>0:
        for s in stats:
            stats_wind.remove(s)
            stats_pres.remove(s)
    
    # Create the figure and axes using plt.subplot_mosaic
    fig, axs = plt.subplot_mosaic([['wind_train_val', 'pres_train_val'], ['wind_hist', 'pres_hist']], figsize=(10, 8))

    # Plot train/val losses for wind variable
    axs['wind_train_val'].plot(train_losses['wind'], label='Train Loss')
    axs['wind_train_val'].plot(val_losses['wind'], label='Val Loss')
    axs['wind_train_val'].set_title('Train/Val Losses - Wind')
    axs['wind_train_val'].legend()
    axs['wind_train_val'].set_yscale('log')

    # Plot train/val losses for pres variable
    axs['pres_train_val'].plot(train_losses['pres'], label='Train Loss')
    axs['pres_train_val'].plot(val_losses['pres'], label='Val Loss')
    axs['pres_train_val'].set_title('Train/Val Losses - Pressure')
    axs['pres_train_val'].legend()
    axs['pres_train_val'].set_yscale('log')

    # Get the outputs of the model on the test set
    test_outputs_wind = []
    truth_wind = []
    test_outputs_pres = []
    truth_pres = []
    with torch.no_grad():
        for test_inputs, target in test_loader_wind:
            test_inputs, target = test_inputs.float(), target.float()
            test_outputs_wind.append(mlp_wind(test_inputs).item())
            truth_wind.append(target.item())
            
        for test_inputs, target in test_loader_pres:
            test_inputs, target = test_inputs.float(), target.float()
            test_outputs_pres.append(mlp_pres(test_inputs).item())
            truth_pres.append(target.item())

    # Plot histograms for wind variable
    axs['wind_hist'].hist(test_outputs_wind, bins=100, alpha=0.5, label='Model outputs')
    axs['wind_hist'].hist(truth_wind, bins=100, alpha=0.5, label='Truth')
    axs['wind_hist'].set_title('Histogram - Wind')
    axs['wind_hist'].legend()

    # Plot histograms for pres variable
    axs['pres_hist'].hist(test_outputs_pres, bins=100, alpha=0.5, label='Model outputs')
    axs['pres_hist'].hist(truth_pres, bins=100, alpha=0.5, label='Truth')
    axs['pres_hist'].set_title('Histogram - Pressure')
    axs['pres_hist'].legend()
    
    # Set the title of the figure
    st = fig.suptitle(f"{model_name} | Seasons: {', '.join(seasons)} ({num_tcs} TCs) | Deterministic model")
    st.set_y(0.98)
    fig.tight_layout()

    # Save the figure
    fig.savefig(save_path + "Figs/" + f"mlp_{model_name}_{lead_time}h_{'_'.join(s for s in seasons)}_layers_"+\
            f"{'.'.join(str(l) for l in layers)}_lr_{lr}_epochs_{epochs}_sched_{sched}"+\
            (f"_{'_'.join(stat for stat in stats)}" if len(stats)>0 else "")+\
            (f"_w_{'_'.join(stat for stat in stats_wind)}" if len(stats_wind)>0 else "") +\
            (f"_p_{'_'.join(stat for stat in stats_pres)}" if len(stats_pres)>0 else "") +".png", dpi=500)
    
    
    
def plot_mlp_normal(mlp_wind, mlp_pres, train_losses, val_losses, test_loader_wind, test_loader_pres, model_name, seasons, num_tcs,
             stats = [], stats_wind = [], stats_pres = [], mlp_params={}, lead_time=0,
             save_path="/users/lpoulain/louis/plots/mlp/"):
    
    lr, layers, epochs, sched = mlp_params['lr'], mlp_params['layers'], mlp_params['epochs'], mlp_params['scheduler']
    if len(stats)>0:
        for s in stats:
            stats_wind.remove(s)
            stats_pres.remove(s)
    
    # Create the figure and axes using plt.subplot_mosaic
    fig, axs = plt.subplot_mosaic([['wind_train_val', 'pres_train_val'], ['wind_hist', 'pres_hist']], figsize=(10, 8))

    # Plot train/val losses for wind variable
    axs['wind_train_val'].plot(train_losses['wind'], label='Train Loss')
    axs['wind_train_val'].plot(val_losses['wind'], label='Val Loss')
    axs['wind_train_val'].set_title('Train/Val Losses - Wind')
    axs['wind_train_val'].legend()

    # Plot train/val losses for pres variable
    axs['pres_train_val'].plot(train_losses['pres'], label='Train Loss')
    axs['pres_train_val'].plot(val_losses['pres'], label='Val Loss')
    axs['pres_train_val'].set_title('Train/Val Losses - Pressure')
    axs['pres_train_val'].legend()

    # Get the outputs of the model on the test set
    standardized_res_wind = []
    standardized_res_pres = []
    
    with torch.no_grad():
        for test_inputs, target in test_loader_wind:
            test_inputs, target = test_inputs.float(), target.float()
            output = mlp_wind(test_inputs)
            standardized_res_wind.append((target.item()-output[:,0].item())/output[:,1].item())
            #truth_wind.append(target.item())
            
        for test_inputs, target in test_loader_pres:
            test_inputs, target = test_inputs.float(), target.float()
            output = mlp_pres(test_inputs)
            standardized_res_pres.append((target.item()-output[:,0].item())/output[:,1].item())
            #truth_pres.append(target.item())
    
    standardized_res_wind = np.array(standardized_res_wind)       
    standardized_res_pres = np.array(standardized_res_pres)

    # Plot histograms for wind variable
    axs['wind_hist'].hist(standardized_res_wind, bins=100, alpha=0.5, label='Model prediction', weights=np.ones(len(standardized_res_wind))/len(standardized_res_wind))
    axs['wind_hist'].set_ylabel('Probability')
    axs['wind_hist'].annotate(f"Mean: {np.mean(standardized_res_wind):.3f}\nStd: {np.std(standardized_res_wind):.3f}", xy=(0.7, 0.7), xycoords='axes fraction')
    axs['wind_hist'].set_title(multiline_label('Standardized residuals distribution of predicted values - Wind'))
    axs['wind_hist'].legend()

    # Plot histograms for pres variable
    axs['pres_hist'].hist(standardized_res_pres, bins=100, alpha=0.5, label='Model prediction', weights=np.ones(len(standardized_res_pres))/len(standardized_res_pres))
    axs['pres_hist'].set_ylabel('Probability')
    axs['pres_hist'].annotate(f"Mean: {np.mean(standardized_res_pres):.3f}\nStd: {np.std(standardized_res_pres):.3f}", xy=(0.7, 0.7), xycoords='axes fraction')
    axs['pres_hist'].set_title(multiline_label('Standardized residuals distribution of predicted values - Pressure'))
    axs['pres_hist'].legend()
    
    # Set the title of the figure
    st = fig.suptitle(f"{model_name} | Seasons: {', '.join(seasons)} ({num_tcs} TCs) | Normal dist model")
    st.set_y(0.98)
    fig.tight_layout()

    # Save the figure
    fig.savefig(save_path + "Figs/" + f"mlp_normal_{model_name}_{lead_time}h_{'_'.join(s for s in seasons)}_layers_"+\
            f"{'.'.join(str(l) for l in layers)}_lr_{lr}_epochs_{epochs}_sched_{sched}"+\
            (f"_{'_'.join(stat for stat in stats)}" if len(stats)>0 else "")+\
            (f"_w_{'_'.join(stat for stat in stats_wind)}" if len(stats_wind)>0 else "") +\
            (f"_p_{'_'.join(stat for stat in stats_pres)}" if len(stats_pres)>0 else "") +".png", dpi=500)
    
    

def plot_mlp_shash(mlp_wind, mlp_pres, train_losses, val_losses, test_loader_wind, test_loader_pres, model_name, seasons, num_tcs,
             stats = [], stats_wind = [], stats_pres = [], mlp_params={}, lead_time=0,
             save_path="/users/lpoulain/louis/plots/mlp/"):
    
    lr, layers, epochs, sched = mlp_params['lr'], mlp_params['layers'], mlp_params['epochs'], mlp_params['scheduler']
    if len(stats)>0:
        for s in stats:
            stats_wind.remove(s)
            stats_pres.remove(s)
    
    # Create the figure and axes using plt.subplot_mosaic
    fig, axs = plt.subplot_mosaic([['wind_train_val', 'pres_train_val'], ['wind_hist', 'pres_hist']], figsize=(10, 8))

    # Plot train/val losses for wind variable
    axs['wind_train_val'].plot(train_losses['wind'], label='Train Loss')
    axs['wind_train_val'].plot(val_losses['wind'], label='Val Loss')
    axs['wind_train_val'].set_title('Train/Val Losses - Wind')
    axs['wind_train_val'].legend()

    # Plot train/val losses for pres variable
    axs['pres_train_val'].plot(train_losses['pres'], label='Train Loss')
    axs['pres_train_val'].plot(val_losses['pres'], label='Val Loss')
    axs['pres_train_val'].set_title('Train/Val Losses - Pressure')
    axs['pres_train_val'].legend()

    # Get the outputs of the model on the test set
    standardized_res_wind = []
    standardized_res_pres = []
    
    with torch.no_grad():
        for test_inputs, target in test_loader_wind:
            test_inputs, target = test_inputs.float(), target.float()
            output = mlp_wind(test_inputs)
            
            loc, scale, skew, tau = output[:,0].item(), output[:,1].item(), output[:,2].item(), output[:,3].item()
            Mean, Std = ShashMoments(loc, scale, skew, tau)
            
            standardized_res_wind.append((target.item()-Mean)/Std)
            #truth_wind.append(target.item())
            
        for test_inputs, target in test_loader_pres:
            test_inputs, target = test_inputs.float(), target.float()
            output = mlp_pres(test_inputs)
            
            loc, scale, skew, tau = output[:,0].item(), output[:,1].item(), output[:,2].item(), output[:,3].item()
            Mean, Std = ShashMoments(loc, scale, skew, tau)
            
            standardized_res_pres.append((target.item()-Mean)/Std)
            #truth_pres.append(target.item())
    
    standardized_res_wind = np.array(standardized_res_wind)       
    standardized_res_pres = np.array(standardized_res_pres)

    # Plot histograms for wind variable
    axs['wind_hist'].hist(standardized_res_wind, bins=100, alpha=0.5, label='Model prediction', weights=np.ones(len(standardized_res_wind))/len(standardized_res_wind))
    axs['wind_hist'].set_ylabel('Probability')
    axs['wind_hist'].annotate(f"Mean: {np.mean(standardized_res_wind):.3f}\nStd: {np.std(standardized_res_wind):.3f}", xy=(0.7, 0.7), xycoords='axes fraction')
    axs['wind_hist'].set_title(multiline_label('Standardized residuals distribution of predicted values - Wind'))
    axs['wind_hist'].legend()

    # Plot histograms for pres variable
    axs['pres_hist'].hist(standardized_res_pres, bins=100, alpha=0.5, label='Model prediction', weights=np.ones(len(standardized_res_pres))/len(standardized_res_pres))
    axs['pres_hist'].set_ylabel('Probability')
    axs['pres_hist'].annotate(f"Mean: {np.mean(standardized_res_pres):.3f}\nStd: {np.std(standardized_res_pres):.3f}", xy=(0.7, 0.7), xycoords='axes fraction')
    axs['pres_hist'].set_title(multiline_label('Standardized residuals distribution of predicted values - Pressure'))
    axs['pres_hist'].legend()
    
    # Set the title of the figure
    st = fig.suptitle(f"{model_name} | Seasons: {', '.join(seasons)} ({num_tcs} TCs) | SHASH model")
    st.set_y(0.98)
    fig.tight_layout()

    # Save the figure
    fig.savefig(save_path + "Figs/" + f"mlp_shash_{model_name}_{lead_time}h_{'_'.join(s for s in seasons)}_layers_"+\
            f"{'.'.join(str(l) for l in layers)}_lr_{lr}_epochs_{epochs}_sched_{sched}"+\
            (f"_{'_'.join(stat for stat in stats)}" if len(stats)>0 else "")+\
            (f"_w_{'_'.join(stat for stat in stats_wind)}" if len(stats_wind)>0 else "") +\
            (f"_p_{'_'.join(stat for stat in stats_pres)}" if len(stats_pres)>0 else "") +".png", dpi=500)
    

