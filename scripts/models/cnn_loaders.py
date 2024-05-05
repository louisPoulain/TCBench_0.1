import glob, os, sys, pickle, torch, subprocess
import pandas as pd
import numpy as np
import xarray as xr

from time import time
from torch.utils.data import Dataset
from utils.main_utils import get_lead_time, get_start_date_nc, get_tc_id_nc
from multiprocessing import Pool


DELTAS_PER_SEASON = {'p_True': {'2000': 39082, '2001': 33078, '2002': 39724, '2003': 52532, '2004': 56814, '2005': 49610, '2006': 46047, '2007': 34028},
                     'p_False': {'2000': 46620, '2001': 36623, '2002': 41296, '2003': 53617, '2004': 58405, '2005': 50519, '2006': 46605, '2007': 34343}}

Q1_WIND = 1.3 # m/s, estimated from 2005 in north Atlantic
MAX_WIND = 30 # m/s, estimated from 2005 in north Atlantic
MIN_PRES = 95000 # Pa, estimated from 2005 in north Atlantic
Q3_PRES = 102000 # Pa, estimated from 2005 in north Atlantic
linear = lambda orig, target, point: (target[1]-target[0])*point/(orig[1]-orig[0]) + (target[0]*orig[1] - orig[0]*target[1])/(orig[1]-orig[0])


def get_ibtracs_data(df, seasons=[], pres=True):
    
    if not isinstance(seasons, list):
        seasons = [seasons]
        
    df = df[df["SEASON"].isin([str(season) for season in seasons])]
    
    tc_ids = df["SID"].unique()
    valid_dates = {}
    
    wind_col = "USA_WIND"
    pres_cols = [col for col in df.columns if "_PRES" in col and "PRES_" not in col and col!="WMO_PRES"]
    pres_columns = {}
    for tc_id in tc_ids:
        tmp_df = df[df["SID"]==tc_id].iloc[1:] # we never predict the start of the TC
        idxs = [idx for idx in tmp_df.index if tmp_df.loc[idx, wind_col]!=" "]
        
        if pres:
            key = lambda x: np.count_nonzero(tmp_df[x].values.astype("string")!=" ")
            pres_col = sorted(pres_cols, key=key)[-1] # the one with the highest number of values reported
            idxs = [idx for idx in idxs if tmp_df.loc[idx, pres_col]!=" "] # remove rows with missing values
            pres_columns[tc_id] = pres_col

        valid_dates[tc_id] = tmp_df.loc[idxs, "ISO_TIME"].values
        
    return df, valid_dates, pres_columns
    


class CNN4PP_Dataset(Dataset):
    
    def __init__(self, data_path, model_name, ibtracs_path, seasons, pres=True, 
                 save_path="/users/lpoulain/louis/plots/cnn/", train_seasons=[], create_input=True, verif_input=False, noprint=False):
        
        
        
        if isinstance(seasons, str):
            seasons = [int(seasons)]
        if isinstance(seasons, int):
            seasons = [seasons]
        self.noprint = noprint
        self.pres = pres
        self.save_path = save_path
        self.seasons = seasons
        self.model_name = "pangu" if model_name in ["pangu", "panguweather"] else model_name
        self.data_folder = "panguweather" if model_name=="pangu" else model_name
        self.train_seasons = train_seasons
        #self.delta_lat_mean, self.delta_lon_mean = 0, 0
        #self.delta_lat_std, self.delta_lon_std = 0, 0
        #delta_lon_mean_sq, delta_lat_mean_sq = 0, 0

        ibtracs_df = pd.read_csv(ibtracs_path, na_filter=False, dtype="string")
        self.ibtracs_df, self.valid_dates, self.pres_columns = get_ibtracs_data(ibtracs_df, seasons, pres=pres)
        
        
        self.data_list = []
        self.data_list_scratch = []
        self.ll = [0] * len(self.seasons)
        #deltas_counter = 0
        
        for i, season in enumerate(seasons):
            if not os.path.isfile(os.path.join(save_path, f"Data/{self.data_folder}/netcdf/Data_list_{season}_p_{self.pres}.pkl")):
                data_list = []
                tmp_df = ibtracs_df[ibtracs_df["SEASON"] == str(season)]
                tc_ids = tmp_df["SID"].unique()
                for tc_id in tc_ids:
                    key = lambda x: (get_start_date_nc(x), get_lead_time(x))
                    tmp_paths = sorted(glob.glob(os.path.join(data_path, f"{self.data_folder}/{self.model_name}*{tc_id}*small.nc")), key=key)

                    for path in tmp_paths:
                        dates = [get_start_date_nc(path) + np.timedelta64(i, "h") for i in range(6, get_lead_time(path)+6, 6)]
                        valid_idxs = [i for i, date in enumerate(dates) if date in self.valid_dates[tc_id].astype("datetime64[ns]")]
                        if len(valid_idxs)==0:
                            continue
                        data_list.append((path, valid_idxs, len(valid_idxs)))
                        #data_list_scratch.append((f"/scratch/lpoulain/{season}/" + os.path.basename(path).replace(".nc", "_wind_pres.npy"\
                        #                        if self.pres else "_wind.npy"), valid_idxs, len(valid_idxs)))
                with open(os.path.join(save_path, f"Data/{self.data_folder}/netcdf/Data_list_{season}_p_{self.pres}.pkl"), "wb") as f:
                    pickle.dump(data_list, f)
            if not os.path.isfile(os.path.join(save_path, f"Data/{self.data_folder}/scratch/Data_list_{season}_p_{self.pres}.pkl")):
                data_list_scratch = [f"/scratch/lpoulain/{season}/{x}" for x in ["fields", "coords", "truth"]]
                with open(os.path.join(save_path, f"Data/{self.data_folder}/scratch/Data_list_{season}_p_{self.pres}.pkl"), "wb") as f:
                    pickle.dump(data_list_scratch, f)
            
            # load data_lists
            with open(os.path.join(save_path, f"Data/{self.data_folder}/netcdf/Data_list_{season}_p_{self.pres}.pkl"), "rb") as f:
                self.data_list.extend(pickle.load(f))
            with open(os.path.join(save_path, f"Data/{self.data_folder}/scratch/Data_list_{season}_p_{self.pres}.pkl"), "rb") as f:
                self.data_list_scratch.append(pickle.load(f))
            
        #for season in train_seasons:
        #    self.delta_lat_mean += np.load(f"/scratch/lpoulain/{season}/lat_diff_mean_p_{self.pres}.npy") * DELTAS_PER_SEASON[f"p_{self.pres}"][season]
        #    self.delta_lon_mean += np.load(f"/scratch/lpoulain/{season}/lon_diff_mean_p_{self.pres}.npy") * DELTAS_PER_SEASON[f"p_{self.pres}"][season]
        #    delta_lat_mean_sq += np.load(f"/scratch/lpoulain/{season}/lat_diff_meanx2_p_{self.pres}.npy") * DELTAS_PER_SEASON[f"p_{self.pres}"][season]
        #    delta_lon_mean_sq += np.load(f"/scratch/lpoulain/{season}/lon_diff_meanx2_p_{self.pres}.npy") * DELTAS_PER_SEASON[f"p_{self.pres}"][season]
        #    deltas_counter += DELTAS_PER_SEASON[f"p_{self.pres}"][season]
            
        #self.delta_lat_mean /= deltas_counter
        #self.delta_lon_mean /= deltas_counter
        #self.delta_lat_std = np.sqrt(delta_lat_mean_sq/deltas_counter - self.delta_lat_mean**2)
        #self.delta_lon_std = np.sqrt(delta_lon_mean_sq/deltas_counter - self.delta_lon_mean**2)
        
        #self.pkl2npy()
        #self.rename()
        if create_input:
            self.save_as_np_new()
        self.numel = []
        
        for i, season in enumerate(self.seasons):
            with open(os.path.join(f"/scratch/lpoulain/{season}/bs_{self.model_name}_p_{self.pres}.pkl"), "rb") as f:
                bs_tmp = pickle.load(f)
            
            self.ll[i] = (i, len(bs_tmp), bs_tmp)
            self.numel.append(sum(bs_tmp))
        if verif_input:
            self.check_input_list()
        #self.save_as_np() # in case some corrupted files were deleted
        
        self.get_target_normalisation_cst()
        self.get_input_normalisation_cst()
        
        #self.target_mean = np.concatenate((self.target_mean, [self.delta_lat_mean, self.delta_lon_mean]))
        #self.target_std = np.concatenate((self.target_std, [self.delta_lat_std, self.delta_lon_std]))
        self.wind_extent = (np.array([Q1_WIND, MAX_WIND]) - self.mean[0].mean()) / self.std[0].mean()
        if pres:
            self.pres_extent = (np.array([MIN_PRES, Q3_PRES]) - self.mean[1].mean()) / self.std[1].mean()
        
        
        
    def __len__(self):
        return sum(self.numel)
    
    
    def __getitem__(self, idx):
        l_tot = 0 # global index (ie which season)
        i_tot = 0
        i = 0 # local index (ie which file)
        if idx < self.numel[0]:
            l = 0
        else:
            while idx >= l_tot:
                l_tot += self.numel[i_tot]
                i_tot += 1
            l_tot -= self.numel[i_tot-1]
            i_tot -= 1
            l = l_tot
        while idx >= l:
            l += self.ll[i_tot][2][i]
            i += 1
        l -= self.ll[i_tot][2][i-1] # we want the biggest l such that l<=idx and ll[i]+l > idx
        i -= 1
        
        sel_idx = idx - l # which index to select in file i
        prefixes = self.data_list_scratch[i_tot]
            
        fields = np.load(prefixes[0] + f"_{self.model_name}_{i}_p_{self.pres}.npy", mmap_mode='r')[sel_idx]
        lat_lon_ldt = np.load(prefixes[1] + f"_{self.model_name}_{i}_p_{self.pres}.npy", mmap_mode='r')[sel_idx]
        truth = np.load(prefixes[2] + f"_{self.model_name}_{i}_p_{self.pres}.npy", mmap_mode='r')[sel_idx]
        
        # Normalize coords so that values fall in about the same range as wind/pres normalised
        coords = np.array([0., 0., 0.])
        coords[0] = linear([-90, 90], [min(self.wind_extent[0], self.pres_extent[0]), max(self.wind_extent[1], self.pres_extent[1])], lat_lon_ldt[0])
        coords[1] = linear([0, 359.75], [min(self.wind_extent[0], self.pres_extent[0]), max(self.wind_extent[1], self.pres_extent[1])], lat_lon_ldt[1])
        coords[2] = linear([6., 168.], [min(self.wind_extent[0], self.pres_extent[0]), max(self.wind_extent[1], self.pres_extent[1])], lat_lon_ldt[2])
        
        fields = (fields - np.array(self.mean)) / np.array(self.std)
        truth = (truth - np.array(self.target_mean)) / np.array(self.target_std)
        return torch.tensor(fields), torch.tensor(coords), torch.tensor(truth)
    
    
    def check_input_list(self):
        wrong_files = []
        for i, season in enumerate(self.seasons):
            with open(f"/scratch/lpoulain/{season}/bs_{self.model_name}_p_{self.pres}.pkl", "rb") as f:
                bs_tmp = pickle.load(f)
            with open(f"/users/lpoulain/louis/plots/cnn/Data/{self.data_folder}/netcdf/Data_list_{season}_p_{self.pres}.pkl", "rb") as f:
                tmp_data_list = pickle.load(f)
            with open(f"/users/lpoulain/louis/plots/cnn/Data/{self.data_folder}/scratch/Data_list_{season}_p_{self.pres}.pkl", "rb") as f:
                tmp_data_list_scratch = pickle.load(f)
            assert len(tmp_data_list) == len(bs_tmp), f"{len(tmp_data_list)}!={len(bs_tmp)}"
            for j in range(len(tmp_data_list)):
                for p in tmp_data_list_scratch:
                    if os.path.getsize(p + f"_{self.model_name}_{j}_p_{self.pres}.npy")==0:
                        wrong_files.append(tmp_data_list[j][0], f"empty size for {p}")
                    else:
                        arr = np.load(p + f"_{self.model_name}_{j}_p_{self.pres}.npy", mmap_mode='r')
                        if arr.shape[0]!=bs_tmp[j]:
                            wrong_files.append((tmp_data_list[j][0], f"shape issue with {p}_{j}_{self.pres} ({arr.shape[0]}!={bs_tmp[j]})"))
        if len(wrong_files)>0:
            print(f"Found {len(wrong_files)} corrupted files.")
            print(wrong_files)
        
                

    
    def get_target_normalisation_cst(self):
        tc_ids_all = list(self.valid_dates.keys())
        
        for season in self.train_seasons:
            if False in [os.path.isfile(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_{x}.npy") for x in\
                         ["target_mean_x", "target_mean_x_sq", "target_counter"]]:
                print(f"Computing normalisation constants for targets ({season}) ... ")
                tmp_ibtracs = self.ibtracs_df[self.ibtracs_df["SEASON"]==str(season)]
                tc_ids = [tc_id for tc_id in tc_ids_all if tc_id in tmp_ibtracs["SID"].values]
                count = 0
                meanX = [0, 0] if self.pres else [0]
                meanX_sq = [0, 0] if self.pres else [0]
                
                for tc_id in tc_ids:

                    tmp_df = tmp_ibtracs[tmp_ibtracs["SID"]==tc_id]
                    tmp_df = tmp_df[tmp_df["ISO_TIME"].astype("datetime64[ns]").isin(self.valid_dates[tc_id].astype("datetime64[ns]"))]
                    
                    for idx in tmp_df.index:
                        count += 1
                        wind = np.array(tmp_df.loc[idx, "USA_WIND"], dtype="float") * 0.514444
                        meanX[0] += wind
                        meanX_sq[0] += wind**2
                        
                        if self.pres:
                            pres = np.array(tmp_df.loc[idx, self.pres_columns[tc_id]], dtype="float") * 100
                            meanX[1] += pres
                            meanX_sq[1] += pres**2
                            
                np.save(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_target_mean_x.npy", np.array(meanX)/count)
                np.save(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_target_mean_x_sq.npy", np.array(meanX_sq)/count)
                np.save(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_target_counter.npy", np.array(count))
        
        c = 0
        self.target_mean = [0, 0] if self.pres else [0]
        self.target_mean_sq = [0, 0] if self.pres else [0]
        self.target_std = [0, 0] if self.pres else [0]
        
        for season in self.train_seasons:
            counter = np.load(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_target_counter.npy").item()
            c += counter
            
            tmp_mean = np.load(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_target_mean_x.npy")
            tmp_mean_sq = np.load(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_target_mean_x_sq.npy")
            
            self.target_mean[0] += tmp_mean[0]*counter
            self.target_mean_sq[0] += tmp_mean_sq[0]*counter
            
            if self.pres:
                self.target_mean[1] += tmp_mean[1]*counter
                self.target_mean_sq[1] += tmp_mean_sq[1]*counter
            
        self.target_mean[0] /= c
        self.target_mean_sq[0] /= c
        self.target_std[0] = np.sqrt(self.target_mean_sq[0] - self.target_mean[0]**2)
        if self.pres:
            self.target_mean[1] /= c
            self.target_mean_sq[1] /= c
            self.target_std[1] = np.sqrt(self.target_mean_sq[1] - self.target_mean[1]**2)
            
            
                    
    def get_input_normalisation_cst(self):
        # taken from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        
        
        tc_ids_all = list(self.valid_dates.keys())
        t = time()
        l_tot = 0
        for season in self.train_seasons:
            if False in [os.path.isfile(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_{x}.npy") for x in\
                         ["input_mean_x", "input_mean_x_sq", "input_counter"]]:
                print(f"Computing normalisation constants for inputs ({season}) ... ")
                meanX = [np.zeros((241, 241)), np.zeros((241, 241))] if self.pres else [np.zeros((241, 241))]
                meanX_sq = [np.zeros((241, 241)), np.zeros((241, 241))] if self.pres else [np.zeros((241, 241))]
                count = 0
                
                tc_ids = [tc_id for tc_id in tc_ids_all if tc_id in self.ibtracs_df[self.ibtracs_df["SEASON"]==str(season)]["SID"].values]
                tmp_paths = [path for path in self.data_list if get_tc_id_nc(path[0]) in tc_ids]
                l = len(tmp_paths)
                
                for i, (path, idxs, _) in enumerate(tmp_paths):
                    if (i+1)%(l//5)==0 or i==1:
                        print(f"Processing file {l_tot+i+1}/{len(self.data_list)} - Running time: {time()-t}s\n"+\
                        f"Expected time left: {(time()-t)/(i+1+l_tot)*(len(self.data_list)-i-1-l_tot):.2f}s (total)\n"+
                        f"                    {(time()-t)/(i+1)*(l-i-1):.2f}s (Season)")
                    
                    ds = xr.load_dataset(path)
            
                    for idx in idxs:
                        count += 1
                        ds_tmp = ds.isel(time=idx)
                        try:
                            wind_sq = ds_tmp.u10.values**2 + ds_tmp.v10.values**2
                        except AttributeError:
                            print(f"Error with file {path}")
                            del ds_tmp
                            continue
                        if np.count_nonzero(np.isnan(wind_sq))>0:
                            print(f"Nans in wind for file {path}")
                            del ds_tmp
                            count -= 1
                            continue
                        meanX[0] += wind_sq**0.5
                        meanX_sq[0] += wind_sq
                        
                        if self.pres:
                            pres = ds_tmp.msl.values
                            if np.count_nonzero(np.isnan(pres))>0:
                                print(f"Nans in pres for file {path}")
                                del ds_tmp
                                count -= 1
                                continue
                            meanX[1] += pres
                            meanX_sq[1] += pres**2
                    del ds
                np.save(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_input_mean_x.npy", np.array(meanX)/count)
                np.save(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_input_mean_x_sq.npy", np.array(meanX_sq)/count)
                np.save(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_input_counter.npy", np.array(count))
                
                l_tot += l
        
        c = 0
        self.mean = [np.zeros((241, 241)), np.zeros((241, 241))] if self.pres else [np.zeros((241, 241))]
        self.mean_sq = [np.zeros((241, 241)), np.zeros((241, 241))] if self.pres else [np.zeros((241, 241))]
        self.std = [np.zeros((241, 241)), np.zeros((241, 241))] if self.pres else [np.zeros((241, 241))]
        
        for season in self.train_seasons:
            counter = np.load(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_input_counter.npy").item()
            c += counter
            
            tmp_mean = np.load(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_input_mean_x.npy")
            tmp_mean_sq = np.load(self.save_path + f"Constants/{self.data_folder}/{season}_p_{self.pres}_input_mean_x_sq.npy")
            if np.count_nonzero(np.isnan(tmp_mean))>0:
                print(f"Error with season {season}")
                print(f"Counter = {counter}")
                raise ValueError("Nan values in mean")
            if np.count_nonzero(np.isnan(tmp_mean_sq))>0:
                print(f"Error with season {season}")
                print(f"Counter = {counter}")
                raise ValueError("Nan values in mean_sq")
            
            self.mean[0] += tmp_mean[0]*counter
            self.mean_sq[0] += tmp_mean_sq[0]*counter
            
            if self.pres:
                self.mean[1] += tmp_mean[1]*counter
                self.mean_sq[1] += tmp_mean_sq[1]*counter
                
        self.mean[0] /= c
        self.mean_sq[0] /= c
        self.std[0] = np.sqrt(self.mean_sq[0] - self.mean[0]**2)
        
        if self.pres:
            self.mean[1] /= c
            self.mean_sq[1] /= c
            self.std[1] = np.sqrt(self.mean_sq[1] - self.mean[1]**2)
    
    
    def save_as_np_new(self):
        save_folder = "/scratch/lpoulain/"
        for season in self.seasons:
            if not os.path.isdir(save_folder+f"{season}"):
                os.mkdir(save_folder+f"{season}")
            
            with open(f"/users/lpoulain/louis/plots/cnn/Data/{self.data_folder}/netcdf/Data_list_{season}_p_{self.pres}.pkl", 'rb') as fi:
                self.tmp_list = pickle.load(fi)
                
            idxs = np.arange(len(self.tmp_list))
            bs = [self.tmp_list[i][2] for i in idxs]
            
            if not len(glob.glob(f"/scratch/lpoulain/{season}/*{self.model_name}*_p_{self.pres}.npy"))==3*len(self.tmp_list):
                idxs = [i for i in idxs if False in [os.path.isfile(f"/scratch/lpoulain/{season}/{x}_{self.model_name}_{i}_p_{self.pres}.npy")\
                            for x in ["fields", "coords", "truth"]]]
                print(f"Saving data ({len(idxs)} instances) as numpy arrays ...")
                self.bs_tmp = [self.tmp_list[i][2] for i in idxs]
                if len(idxs)!=0:
                    with Pool(min(os.cpu_count()//2, len(idxs))) as p:
                        p.map(self.save_one_as_np_new, idxs)
            del self.tmp_list
            if not os.path.isfile(f"/scratch/lpoulain/{season}/bs_{self.model_name}_p_{self.pres}.pkl"):        
                with open(f"/scratch/lpoulain/{season}/bs_{self.model_name}_p_{self.pres}.pkl", "wb") as f:
                    pickle.dump(bs, f)
        if not self.noprint:
            print("All data saved as numpy arrays.")
                    
    
    def save_one_as_np_new(self, idx):
        ds = xr.open_dataset(self.tmp_list[idx][0])
        time0 = get_start_date_nc(self.tmp_list[idx][0])
        ds = ds.isel(time=self.tmp_list[idx][1])
        
        tc_id = get_tc_id_nc(self.tmp_list[idx][0])
        df = self.ibtracs_df[self.ibtracs_df["SID"]==tc_id]
        season = df["SEASON"].values[0]
        
        start_lat, start_lon = df[df["ISO_TIME"].astype("datetime64[ns]")==time0][["LAT", "LON"]].values[0]
        start_lat, start_lon = np.float32(start_lat), np.float32(start_lon)
        start_lon = start_lon + 360 if start_lon<0 else start_lon
        
        
        times = ds.time.values if isinstance(ds.time.values[0], np.datetime64) else\
                [np.datetime64(val + get_start_date_nc(self.tmp_list[idx][0])) for val in ds.time.values]
             
        end_lats = df[df["ISO_TIME"].astype("datetime64[ns]").isin(times)]["LAT"].values.astype(np.float32)
        end_lons = df[df["ISO_TIME"].astype("datetime64[ns]").isin(times)]["LON"].values.astype(np.float32)
        end_lons = [l + 360 if l<0 else l for l in end_lons]
        if len(end_lons)!=len(times):
            print(f"Error with file {self.tmp_list[idx][0]}")
            return
        
        ldts = [np.timedelta64((times[i]-time0), "h").astype(np.float32) for i in range(len(times))]
        coords_start = np.repeat(np.array([start_lat, start_lon], dtype=np.float32).reshape(1,-1), len(times), axis=0)
        coords = np.concatenate((coords_start, np.array(ldts).reshape(-1, 1)), axis=1)
        
        wind_truth = df[df["ISO_TIME"].astype("datetime64[ns]").isin(times)]["USA_WIND"].values.astype(np.float32) * 0.514444
        wind = np.sqrt(ds.u10.values**2 + ds.v10.values**2).astype(np.float32)
        
        if self.pres:
            pres = ds.msl.values.astype(np.float32)
            pres_truth = df[df["ISO_TIME"].astype("datetime64[ns]").isin(times)][self.pres_columns[tc_id]].values.astype(np.float32) * 100
        
        #truth = np.column_stack([wind_truth, pres_truth, [end_lat-start_lat for end_lat in end_lats], 
        #                            [end_lon-start_lon for end_lon in end_lons]]) if self.pres else\
        #        np.column_stack([wind_truth, [end_lat-start_lat for end_lat in end_lats], [end_lon-start_lon for end_lon in end_lons]])
        truth = np.column_stack([wind_truth, pres_truth]) if self.pres else wind_truth
        ds.close()
        del ds
        # Save shape for fields is b*c*h*w
        if self.pres:
            res_fields = np.concatenate([wind.reshape((wind.shape[0], 1, *wind.shape[1:])), pres.reshape((pres.shape[0], 1, *pres.shape[1:]))], 
                                        axis=1)
            res_coords = coords
            res_truths = truth
        else:
            res_fields = wind.reshape((wind.shape[0], 1, *wind.shape[1:]))
            res_coords = coords
            res_truths = truth
        
        np.save(f"/scratch/lpoulain/{season}/fields_{self.model_name}_{idx}_p_{self.pres}.npy", res_fields)
        np.save(f"/scratch/lpoulain/{season}/coords_{self.model_name}_{idx}_p_{self.pres}.npy", res_coords)
        np.save(f"/scratch/lpoulain/{season}/truth_{self.model_name}_{idx}_p_{self.pres}.npy", res_truths)
        if (idx+1)%(len(self.bs_tmp)//5) == 0:
            print(idx, flush=True)
            
    
    
    def rename_one(self, idx):
        prefixes = self.data_list_scratch[self.idx_season]      
        if not os.path.isfile(prefixes[0] + f"_{self.model_name}_{idx}_p_{self.pres}.npy"):
            fields = np.load(prefixes[0] + f"_{idx}_p_{self.pres}.npy")
            coords = np.load(prefixes[1] + f"_{idx}_p_{self.pres}.npy")
            truth = np.load(prefixes[2] + f"_{idx}_p_{self.pres}.npy")
            
            np.save(prefixes[0] + f"_{self.model_name}_{idx}_p_{self.pres}.npy", fields)
            np.save(prefixes[1] + f"_{self.model_name}_{idx}_p_{self.pres}.npy", coords)
            np.save(prefixes[2] + f"_{self.model_name}_{idx}_p_{self.pres}.npy", truth)
            subprocess.run(["rm", prefixes[0] + f"_{idx}_p_{self.pres}.npy"])
            subprocess.run(["rm", prefixes[1] + f"_{idx}_p_{self.pres}.npy"])
            subprocess.run(["rm", prefixes[2] + f"_{idx}_p_{self.pres}.npy"])
        
        
        
    def rename(self):
        for i, season in enumerate(self.seasons):
            with open(f"/users/lpoulain/louis/plots/cnn/Data/{self.data_folder}/netcdf/Data_list_{season}_p_{self.pres}.pkl", 'rb') as fi:
                self.tmp_list = pickle.load(fi)
            idxs = np.arange(len(self.tmp_list))
            self.idx_season = i
            with Pool(os.cpu_count()//2) as p:
                p.map(self.rename_one, idxs)
            
            bs = [self.tmp_list[i][2] for i in idxs]
            with open(f"/scratch/lpoulain/{season}/bs_{self.model_name}_p_{self.pres}.pkl", "wb") as f:
                pickle.dump(bs, f)
            subprocess.run(["rm", f"/scratch/lpoulain/{season}/bs_p_{self.pres}.pkl"])
    
    """
    def save_as_np(self):
        save_folder = "/scratch/lpoulain/"
        for season in self.seasons:
            if not os.path.isdir(save_folder+f"{season}"):
                os.mkdir(save_folder+f"{season}")
        idxs = [i for i, path in enumerate(self.data_list_scratch) if not os.path.isfile(path[0])]
        if len(idxs)!=0:
            print(f"Saving data ({len(idxs)} instances) as numpy arrays ...")
            with Pool(min(os.cpu_count()//2, len(idxs))) as p:
                p.map(self.save_one_as_np, idxs)
        print("All data saved as numpy arrays.")
    
    
    def save_one_as_np(self, idx):
        save_loc = self.data_list_scratch[idx][0]
        if not os.path.isfile(save_loc):
            ds = xr.open_dataset(self.data_list[idx][0])
            time0 = get_start_date_nc(self.data_list[idx][0])
            
            ds = ds.isel(time=self.data_list[idx][1])
            
            tc_id = get_tc_id_nc(self.data_list[idx][0])
            df = self.ibtracs_df[self.ibtracs_df["SID"]==tc_id]
            
            start_lat, start_lon = df[df["ISO_TIME"].astype("datetime64[ns]")==time0][["LAT", "LON"]].values[0]
            start_lat, start_lon = float(start_lat), float(start_lon)
            
            
            times = ds.time.values if isinstance(ds.time.values[0], np.datetime64) else\
                    [np.datetime64(val + get_start_date_nc(self.data_list[idx][0])) for val in ds.time.values]
                    
            end_lats = df[df["ISO_TIME"].astype("datetime64[ns]").isin(times)]["LAT"].values.astype(float)
            end_lons = df[df["ISO_TIME"].astype("datetime64[ns]").isin(times)]["LON"].values.astype(float)
            
            ldts = [np.timedelta64((times[i]-time0), "h").astype(float) for i in range(len(times))]
            coords_start = np.repeat(np.array([start_lat, start_lon]).reshape(1,-1), len(times), axis=0)
            coords = np.concatenate((coords_start, np.array(ldts).reshape(-1, 1)), axis=1)
            
            wind_truth = df[df["ISO_TIME"].astype("datetime64[ns]").isin(times)]["USA_WIND"].values.astype(float)
            wind = (ds.u10.values**2 + ds.v10.values**2)**0.5
            
            if self.pres:
                pres = ds.msl.values
                pres_truth = df[df["ISO_TIME"].astype("datetime64[ns]").isin(times)][self.pres_columns[tc_id]].values.astype(float)
            
            truth = np.column_stack([wind_truth, pres_truth, [end_lat-start_lat for end_lat in end_lats], 
                                     [end_lon-start_lon for end_lon in end_lons]]) if self.pres else\
                    np.column_stack([wind_truth, [end_lat-start_lat for end_lat in end_lats], [end_lon-start_lon for end_lon in end_lons]])

            ds.close()
            del ds
            if self.pres:
                np.save(save_loc, np.array([np.transpose(np.concatenate([wind.reshape(1, *wind.shape), pres.reshape(1, *pres.shape)], axis=0), (-2,-1,0,1)),
                                            coords, truth], dtype=object), allow_pickle=True)
            else:
                np.save(save_loc, np.array([np.transpose(wind.reshape(1, *wind.shape), (-2,-1,0,1)), coords, truth], dtype=object), allow_pickle=True)        
    
    
    
    """