import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob, os
import xarray as xr
import re
from scripts import utils
import random, subprocess

def flatten(arg):
    # flatten list of any depth into a list of depth 1
    if not isinstance(arg, list): # if not list
        return [arg]
    return [x for sub in arg for x in flatten(sub)] # recurse and collect


class TCData(Dataset):
    def __init__(self, data_path, model_name, mode="deterministic", small=False, data_type='train', normalize=True):
        
        assert data_type in ['train', 'test', 'val'], "data_type must be one of 'train', 'test', or 'val'"
        model_folder = (model_name if model_name != "pangu" else "panguweather") + "/PostProcessing"
        
        self.data_path = data_path
        self.small = small
        self.data_type = data_type
        self.model_name = model_name
        self.model_folder = model_folder

        data = [np.load(f"{self.data_path}/{model_folder}/{model_name}_{mode}_{self.data_type}_{suffix}.npy", mmap_mode='r') for \
                suffix in ["inp_fields", "inp_coords", "inp_ldt", "targets"]]
        if self.small:
            data = [arr[:100] for arr in data]
        
        #self.data = [torch.from_numpy(arr).float() for arr in data]
        #del data
        if normalize:
            self.calculate_norm_cst()
            self.normalize()

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        fields, coords, ldt, target = self.data[0][idx], self.data[1][idx], self.data[2][idx], self.data[3][idx]
        lat, lon = coords
        return torch.from_numpy(fields).float(), torch.from_numpy(lat).float(), torch.from_numpy(lon).float(), \
                torch.from_numpy(ldt).float(), torch.from_numpy(target).float()
    
    def summary(self):
        print("Summary: \n")
        print(f"Number of samples: {self.data[0].shape[0]}")
        print("Input mins (train): ", [f"{prefix}: {self.inputs_min[i].reshape(torch.numel(self.inputs_min[i]))}" for \
            i, prefix in enumerate(['fields', 'coords', 'ldt'])])
        print("Input maxs (train): ", [f"{prefix}: {self.inputs_max[i].reshape(torch.numel(self.inputs_max[i]))}" for \
            i, prefix in enumerate(['fields', 'coords', 'ldt'])])
        print(f"Target mins (train): {self.targets_min} (Delta lat, Delta lon, Wnd, Msl)")
        print(f"Target maxs (train): {self.targets_max} (Delta lat, Delta lon, Wnd, Msl)")

    
    def calculate_norm_cst(self):
        fields, coords, ldt = self.data[:3]
        targets = self.data[3]
        s = "_small" if self.small else ""
        if self.data_type == 'train':
            self.inputs_min = [fields.amin(dim=(0, 2, 3)).reshape((1,3,1,1)), coords.amin(dim=0).reshape((1,2)), ldt.amin(dim=0)]
            self.inputs_max = [fields.amax(dim=(0, 2, 3)).reshape(1,3,1,1), coords.amax(dim=0).reshape(1,2), ldt.amax(dim=0)]
        
            self.targets_min = targets.amin(dim=0)
            self.targets_max = targets.amax(dim=0)
            
            np.save(f"{self.data_path}/{self.model_folder}/{self.model_name}_train_inputs_min{s}.npy", np.asarray(self.inputs_min, dtype=object))
            np.save(f"{self.data_path}/{self.model_folder}/{self.model_name}_train_inputs_max{s}.npy", np.asarray(self.inputs_max, dtype=object))
            np.save(f"{self.data_path}/{self.model_folder}/{self.model_name}_train_targets_min{s}.npy", self.targets_min)
            np.save(f"{self.data_path}/{self.model_folder}/{self.model_name}_train_targets_max{s}.npy", self.targets_max)
            for i in range(targets.shape[1]):
                hist, bin_edges = np.histogram(targets[:, i], bins=100)
                np.save(f"{self.data_path}/{self.model_folder}/{self.model_name}_train_targets_hist_{i}{s}.npy", hist)
                np.save(f"{self.data_path}/{self.model_folder}/{self.model_name}_train_targets_bin_edges_{i}{s}.npy", bin_edges)
            for i in range(coords.shape[1]):
                hist, bin_edges = np.histogram(coords[:, i], bins=100)
                np.save(f"{self.data_path}/{self.model_folder}/{self.model_name}_train_coords_hist_{i}{s}.npy", hist)
                np.save(f"{self.data_path}/{self.model_folder}/{self.model_name}_train_coords_bin_edges_{i}{s}.npy", bin_edges)
            hist, bin_edges = np.histogram(ldt, bins=np.arange(6, 174, 6))
            np.save(f"{self.data_path}/{self.model_folder}/{self.model_name}_train_ldt_hist_{s}.npy", hist)
            np.save(f"{self.data_path}/{self.model_folder}/{self.model_name}_train_ldt_bin_edges_{s}.npy", bin_edges)
        else:
            self.inputs_min = np.load(f"{self.data_path}/{self.model_folder}/{self.model_name}_train_inputs_min{s}.npy", allow_pickle=True)
            self.inputs_max = np.load(f"{self.data_path}/{self.model_folder}/{self.model_name}_train_inputs_max{s}.npy", allow_pickle=True)
            self.targets_min = np.load(f"{self.data_path}/{self.model_folder}/{self.model_name}_train_targets_min{s}.npy")
            self.targets_max = np.load(f"{self.data_path}/{self.model_folder}/{self.model_name}_train_targets_max{s}.npy")
        
        print("Cst calculated")
        
    def normalize(self):
        # input
        self.data[0] = (self.data[0] - self.inputs_min[0]) / (self.inputs_max[0]-self.inputs_min[0])
        # targets
        self.data[3][:, 3:] = (self.data[3][:, 3:] - self.targets_min[3:]) / (self.targets_max[3:]-self.targets_min[3:])




def create_dataset(data_folder, model_name, tc_ids, df_tracks, vars, size=241):
    """
    Create a dataset from a list of track ids and a list of variables
    Since there are different pressure levels, vars is expected to be of the form {var1: [plevs], var2: [plevs], ...}
    For vars without pressure level, e.g. 't2m', the dict should be {'t2m': [0]}
    """
    
    tc_cols_wind = ["USA_WIND"]
    tc_cols_pres = ['WMO_PRES', 'USA_PRES', 'TOKYO_PRES', 'CMA_PRES', 'HKO_PRES', 'NEWDELHI_PRES',
                    'REUNION_PRES', 'BOM_PRES', 'NADI_PRES', 'WELLINGTON_PRES', 'DS824_PRES',
                    'TD9636_PRES', 'TD9635_PRES', 'NEUMANN_PRES', 'MLC_PRES']
    
    folder_name = model_name if model_name != "pangu" else "panguweather"
    type_ = "deterministic"
    vars_as_list = flatten([[(var+f"{int(vars[var][i])}" if plev!=0 else var) for i, plev in enumerate(vars[var])]\
                            for var in list(vars.keys())])
    print(vars_as_list)

    # Load the data
    key = lambda x: (utils.get_start_date_nc(x), utils.get_lead_time(x))
    
    c = 1
    for tc_id in tc_ids:
        if  c % 20 == 0:
            print(f"Processing {c}/{len(tc_ids)}")
        tc_tracks = df_tracks[df_tracks['SID'] == tc_id]
        save_name = f"{data_folder}/{folder_name}/PostProcessing/{model_name}_{tc_id}_{'_'.join(var for var in vars_as_list)}_{type_}"
        saved_files = [f"{save_name}_{suffix}.npy" for suffix in ["inp_fields", "inp_coords", "inp_ldt", "inp_start_time", "targets"]]
        
        if False in [os.path.isfile(saved_file) for saved_file in saved_files]:
            filenames_list = sorted(glob.glob(f"{data_folder}/{folder_name}/{model_name}_*_{tc_id}_small.nc"), key=key)
            inputs = []
            targets = []
            nb_samples = 0
            for filename in filenames_list:
                try:
                    data_tmp = xr.open_dataset(filename, engine="netcdf4")
                except OSError:
                    print(f"Corrupted file: {filename}")
                    subprocess.run(["rm", filename])
                    continue
                try:
                    var1 = list(data_tmp.data_vars)[0]
                except IndexError:
                    print(f"Empty file: {filename}")
                    raise IndexError
                
                # fourcastnet records also the intial time step
                sample_tmp = data_tmp[f"{var1}"].shape[0]-1 if model_name=="fourcastnetv2" else data_tmp[f"{var1}"].shape[0]
                nb_samples += sample_tmp

                target_tmp = np.zeros((sample_tmp, 4)) # delta lat, delta lon, wnd, msl
                input_tmp = (np.zeros((sample_tmp, len(vars_as_list), size, size)),
                            np.zeros((sample_tmp, 2)), 
                            np.empty((sample_tmp), dtype=int),
                            np.empty((sample_tmp), dtype='datetime64[h]')) # "image", lat/lon, ldt, start_time
                
                for i, var in enumerate(vars_as_list):
                    if model_name=="fourcastnetv2":
                        idx = list(data_tmp.coords).index(var)
                        val = data_tmp["__xarray_dataarray_variable__"].values[1:, 0, idx, :, :]
                    else:
                        plev_match = re.search(r'\d+', var)
                        plev = plev_match.group() if plev_match is not None else None
                        if plev in data_tmp.isobaricInhPa: # upper-air variable
                            plev_idx = plev_match.start()
                            idx = data_tmp.isobaricInhPa.index(plev)
                            val = data_tmp[f"{var[:plev_idx]}"].values[:, idx, :, :]
                        else:
                            try:
                                val = data_tmp[f"{var}"].values[:, :, :]
                            except KeyError:
                                print(data_tmp)
                                raise KeyError(f"Variable {var} not found in {filename}")
                    
                    input_tmp[0][:, i, :, :] = val
                if type(data_tmp["time"].values[0]) == np.datetime64:
                    data_tmp["time"] = [np.timedelta64(np.timedelta64(6*i, 'h'),'ns') for i in range(data_tmp["time"].shape[0])] 
                
                times = data_tmp["time"].values[1:] if model_name=="fourcastnetv2" else \
                    utils.get_start_date_nc(filename) + data_tmp["time"].values # time contains timedeltas for pangu/gpc
                
                start_iso_time = np.datetime64(times[0]-np.timedelta64(6, 'h'), 'h')
                
                tc_track_start = tc_tracks[tc_tracks['ISO_TIME'].astype("datetime64[ns]") == start_iso_time]
                idx_to_remove = []
                for i in range(sample_tmp):
                    tc_track_tmp = tc_tracks[tc_tracks['ISO_TIME'].astype("datetime64[ns]") == times[i]]
                    
                    # inputs 
                    ldt = np.timedelta64(times[i] - start_iso_time, 'h').astype(int) if i!=0 else 6
                    #try:
                    lat_start = np.float32(tc_track_start["LAT"].values[0]) # at start of previ
                    #except IndexError:
                    #    print(start_iso_time)
                    #    print(tc_track_start)
                    #    raise IndexError
                    input_tmp[1][i, 0] = lat_start
                    lon_start = np.float32(tc_track_start["LON"].values[0]) # at start of previ
                    lon_start = lon_start + 180 if lon_start < 0 else lon_start
                    input_tmp[1][i, 1] = lon_start

                    input_tmp[2][i] = ldt
                    input_tmp[3][i] = start_iso_time
                    
                    # targets
                    lat = np.float32(tc_track_tmp['LAT'].values[0])
                    target_tmp[i, 0] = lat-lat_start # np.sin(np.radians())
                    lon = np.float32(tc_track_tmp['LON'].values[0])
                    lon = lon + 180 if lon < 0 else lon
                    target_tmp[i, 1] = lon-lon_start # np.cos(np.radians())
                    #target_tmp[i, 2] = lon-lon_start # np.sin(np.radians())
                    wnd, msl = [], []
                    for col in tc_cols_wind:
                        wind = tc_track_tmp[col].values[0]
                        if wind not in ["", " "]:
                            wnd.append(wind)
                    for col in tc_cols_pres:
                        mslp = tc_track_tmp[col].values[0]
                        if mslp not in [' ', '', "", " "]:
                            msl.append(mslp)
                    if len(wnd) == 0:
                        wnd.append(9999)
                        idx_to_remove.append(i)
                    if len(msl) == 0:
                        msl.append(9999)
                        idx_to_remove.append(i)
                    
                    # wind is given in knots, convert to m/s
                    target_tmp[i, 2] = float(wnd[0]) * 0.5144444444444444
                    # pressure is in mb, convert to Pa
                    target_tmp[i, 3] = float(msl[0]) * 100
                idx_to_remove = list(set(idx_to_remove))
                target_tmp = np.delete(target_tmp, idx_to_remove, axis=0)
                input_tmp = tuple([np.delete(inp, idx_to_remove, axis=0) for inp in input_tmp])
                
                inputs.append(input_tmp)
                targets.append(target_tmp)
            final_inputs = [np.zeros((nb_samples, len(vars_as_list), size, size)),
                            np.zeros((nb_samples, 2)), 
                            np.empty((nb_samples), dtype=int),
                            np.empty((nb_samples), dtype='datetime64[h]')] # "image", lat/lon, ldt, start_time
            final_targets = np.concatenate(targets, axis=0)
            for i in range(len(inputs[0])):
                final_inputs[i] = np.concatenate([inp[i] for inp in inputs], axis=0)
            
            np.save(save_name+"_inp_fields.npy", final_inputs[0])
            np.save(save_name+"_inp_coords.npy", final_inputs[1])
            np.save(save_name+"_inp_ldt.npy", final_inputs[2])
            np.save(save_name+"_inp_start_time.npy", final_inputs[3])
            np.save(save_name+"_targets.npy", final_targets)
        c += 1            

def split_data(data_path, model_name, mode="deterministic", splits={"train_val":0.8, "test":0.2}, val_split=0.25):
    
    # tdb: make the split on the number of sample rather than the nb of TCs
    assert sum([splits[key] for key in splits]) == 1, "Split values must sum to 1"
    
    model_folder = (model_name if model_name != "pangu" else "panguweather") + "/PostProcessing"
    
    if "train_val" not in splits.keys():
        
        targetsFiles = [np.load(x, mmap_mode='r') for x in glob.glob(f"{data_path}/{model_folder}/{model_name}*_{mode}_targets.npy")]
        fieldsFiles = [np.load(x, mmap_mode='r') for x in glob.glob(f"{data_path}/{model_folder}/{model_name}*_{mode}_inp_fields.npy")]
        coordsFiles = [np.load(x, mmap_mode='r') for x in glob.glob(f"{data_path}/{model_folder}/{model_name}*_{mode}_inp_coords.npy")]
        ldtFiles = [np.load(x, mmap_mode='r') for x in glob.glob(f"{data_path}/{model_folder}/{model_name}*_{mode}_inp_ldt.npy")]
        shuff_lists = list(zip(targetsFiles, fieldsFiles, coordsFiles, ldtFiles))
        random.shuffle(shuff_lists)
        targetsFiles, fieldsFiles, coordsFiles, ldtFiles = zip(*shuff_lists)
        targetsFiles, fieldsFiles, coordsFiles, ldtFiles = list(targetsFiles), list(fieldsFiles), list(coordsFiles), list(ldtFiles)
        idx_test, idx_val = splitter(targetsFiles, splits)
        
        names = ["targets", "fields", "coords", "ldt"]
        test_files = {f"{names[i]}": x[:idx_test] for i, x in enumerate([targetsFiles, fieldsFiles, coordsFiles, ldtFiles])}
        val_files = {f"{names[i]}": x[idx_test:idx_val] for i, x in enumerate([targetsFiles, fieldsFiles, coordsFiles, ldtFiles])}
        train_files = {f"{names[i]}": x[idx_val:] for i, x in enumerate([targetsFiles, fieldsFiles, coordsFiles, ldtFiles])}
        for name in ["targets", "fields", "coords", "ldt"]:
            savename = name if name=="targets" else f"inp_{name}"
            np.save(f"{data_path}/{model_folder}/{model_name}_{mode}_test_{savename}.npy", np.concatenate(test_files[name], axis=0))
            np.save(f"{data_path}/{model_folder}/{model_name}_{mode}_val_{savename}.npy", np.concatenate(val_files[name], axis=0))
            np.save(f"{data_path}/{model_folder}/{model_name}_{mode}_train_{savename}.npy", np.concatenate(train_files[name], axis=0))
    
    else:
        assert val_split < 0.5, "fold_split must be less than 0.5 (portion of samples to use in val)"
        
        names = ["targets", "fields", "coords", "ldt"]
        
        if False in [os.path.isfile(f"{data_path}/{model_folder}/{model_name}_{mode}_val_train_filenames.txt"),
                     os.path.isfile(f"{data_path}/{model_folder}/{model_name}_{mode}_test_filenames.txt")]:
            filelist = glob.glob(f"{data_path}/{model_folder}/{model_name}*_{mode}_targets.npy")
            targetsFiles = [np.load(x, mmap_mode='r') for x in filelist]
            shuff_lists = list(zip(targetsFiles, filelist))
            random.shuffle(shuff_lists)
            targetsFiles, filelist = zip(*shuff_lists)
            targetsFiles, filelist = list(targetsFiles), list(filelist)
            
            idx_test, idx_val = splitter(targetsFiles, splits={"test":splits["test"], "val":splits["train_val"]*val_split})
            idx_val -= idx_test
            with open(f"{data_path}/{model_folder}/{model_name}_{mode}_val_train_filenames.txt", "w") as f:
                for filename in filelist[idx_test:]:
                    f.write(f"{'_'.join(filename.split('_')[:-1])}\n")
            with open(f"{data_path}/{model_folder}/{model_name}_{mode}_test_filenames.txt", "w") as f:
                for filename in filelist[:idx_test]:
                    f.write(f"{'_'.join(filename.split('_')[:-1])}\n")
                    
        with open(f"{data_path}/{model_folder}/{model_name}_{mode}_val_train_filenames.txt", "r") as f:
                val_train_filenames = [x.strip() for x in f.readlines()]
        with open(f"{data_path}/{model_folder}/{model_name}_{mode}_test_filenames.txt", "r") as f:
                test_filenames = [x.strip() for x in f.readlines()]
        targetsFiles_test = [np.load(x+"_targets.npy", mmap_mode='r') for x in test_filenames]
        fieldsFiles_test = [np.load(x+"_inp_fields.npy", mmap_mode='r') for x in test_filenames]
        coordsFiles_test = [np.load(x+"_inp_coords.npy", mmap_mode='r') for x in test_filenames]
        ldtFiles_test = [np.load(x+"_inp_ldt.npy", mmap_mode='r') for x in test_filenames]
        targetsFiles_val_train = [np.load(x+"_targets.npy", mmap_mode='r') for x in val_train_filenames]
        fieldsFiles_val_train = [np.load(x+"_inp_fields.npy", mmap_mode='r') for x in val_train_filenames]
        coordsFiles_val_train = [np.load(x+"_inp_coords.npy", mmap_mode='r') for x in val_train_filenames]
        ldtFiles_val_train = [np.load(x+"_inp_ldt.npy", mmap_mode='r') for x in val_train_filenames]
        shuff_lists = list(zip(targetsFiles_val_train, fieldsFiles_val_train, coordsFiles_val_train, ldtFiles_val_train))
        random.shuffle(shuff_lists)
        targetsFiles_val_train, fieldsFiles_val_train, coordsFiles_val_train, ldtFiles_val_train = zip(*shuff_lists)
        targetsFiles_val_train, fieldsFiles_val_train, coordsFiles_val_train, ldtFiles_val_train = \
                                                            list(targetsFiles_val_train), list(fieldsFiles_val_train),\
                                                            list(coordsFiles_val_train), list(ldtFiles_val_train)
        idx_val, _ = splitter(targetsFiles_val_train, splits={"test":val_split, "val":1-val_split})
        
        test_files = {f"{names[i]}": x for i, x in enumerate([targetsFiles_test, fieldsFiles_test,
                                                                         coordsFiles_test, ldtFiles_test])}
        val_files = {f"{names[i]}": x[:idx_val] for i, x in enumerate([targetsFiles_val_train, fieldsFiles_val_train,
                                                                       coordsFiles_val_train, ldtFiles_val_train])}
        train_files = {f"{names[i]}": x[idx_val:] for i, x in enumerate([targetsFiles_val_train, fieldsFiles_val_train,
                                                                    coordsFiles_val_train, ldtFiles_val_train])}
        for name in ["targets", "fields", "coords", "ldt"]:
            savename = name if name=="targets" else f"inp_{name}"
            np.save(f"{data_path}/{model_folder}/{model_name}_{mode}_test_{savename}.npy", np.concatenate(test_files[name], axis=0))
            np.save(f"{data_path}/{model_folder}/{model_name}_{mode}_val_{savename}.npy", np.concatenate(val_files[name], axis=0))
            np.save(f"{data_path}/{model_folder}/{model_name}_{mode}_train_{savename}.npy", np.concatenate(train_files[name], axis=0))
            print("1")
               

def splitter(l:list, splits:dict):
    lcopy = l.copy()
    length = sum([x.shape[0] for x in lcopy])
    idx_test, idx_val = 0, 0
    test_number = int(length*splits["test"])
    
    samples_test = 0
    while(samples_test < test_number):
        samples_test += lcopy.pop(0).shape[0]
        idx_test += 1
    
    #length = sum([x.shape[0] for x in l])
    val_number = int(length*splits["val"])
    idx_val = idx_test
    samples_val = 0
    while(samples_val < val_number and len(lcopy)!=0):
        samples_val += lcopy.pop(0).shape[0]
        idx_val += 1 
    
    return idx_test, idx_val