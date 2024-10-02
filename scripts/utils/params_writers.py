import pandas as pd
import numpy as np
import random, os

from main_utils import date_time_nn_to_netcdf


def subtract_ibtracs_iso_times(iso_time1:str, iso_time2:str) -> float:
    # returns the time difference in hours
    
    nb_hours = (np.datetime64(iso_time2, 'm') - np.datetime64(iso_time1, 'm')).astype(float)/60
    
    return nb_hours


def time_ibtracs_to_nn(time: str) -> str:
    # in ibtracs times are of the form hh:mm:ss
    # NNs take times inputs of the form hhmm
    time_split = time.split(":")
    time_new = str("".join(t for t in time_split[:2]))
    return time_new


def date_ibtracs_to_nn(date: str) -> str:
    # in ibtracs dates are of the yyyy-mm-dd
    # NNs take date inputs of the form yyyymmdd
    date_split = date.split("-")
    date_new = str("".join(date_split))
    return date_new


def get_all_iso_times(df: pd.DataFrame, TC_id=None, season=None):
    # to be removed: TC year
    assert season is not None or TC_id is not None, "TC_id or season must be specified"
    
    if TC_id is not None:
        df_TC = df[df["SID"]==TC_id]
    else:
        df_TC_tmp = df[df["SEASON"]==season]
        TC_id = df_TC_tmp["SID"].values[1]
        print(f"TC id: {TC_id} ({season})")
        df_TC = df[df["SID"]==TC_id]
        
            
    return df_TC["ISO_TIME"].values, TC_id


def write_params_for_tc(output_path: str, df: pd.DataFrame, TC_id=None, season=None, step=6, max_lead=168, **kwargs):

    debug = kwargs.get("debug", False)
    
    iso_times, TC_id = get_all_iso_times(df=df, TC_id=TC_id, season=season)
    end_iso_time = iso_times[-1]
    start_iso_time = iso_times[0]
    
    start_date, start_time = date_ibtracs_to_nn(start_iso_time.split(" ")[0]), time_ibtracs_to_nn(start_iso_time.split(" ")[1])
    dates, times, lead_times = [start_date], [start_time], [min(int(subtract_ibtracs_iso_times(start_iso_time, end_iso_time)), max_lead)]
    
    current_iso = start_iso_time
    i = 1
    if max_lead==6:
        for iso in iso_times[1:-1]:
            dates.append(date_ibtracs_to_nn(iso.split(" ")[0]))
            times.append(time_ibtracs_to_nn(iso.split(" ")[1]))
            lead_times.append(6)
            
    while lead_times[-1] > step:
        
        current_iso = iso_times[i]
        date, time = date_ibtracs_to_nn(current_iso.split(" ")[0]), time_ibtracs_to_nn(current_iso.split(" ")[1])
        
        lead_time = min(int(subtract_ibtracs_iso_times(current_iso, end_iso_time)), max_lead)
        
        dates.append(date)
        times.append(time)
        lead_times.append(lead_time)
        i += 1

    timesteps = [True] + [False for i in range(len(times)-1)]
    prev_t = times[0]
    for i, t in enumerate(times[1:]):
        if abs(int(t)//100-int(prev_t)//100)%step==0:
            prev_t = t
            timesteps[i+1] = True
            
    dates, times, lead_times = ["date"] + [int(d) for i, d in enumerate(dates) if timesteps[i]], \
                                ["time"] + [t for i, t in enumerate(times) if timesteps[i]], \
                                ["lead time"] + [int(lt) for i, lt in enumerate(lead_times) if timesteps[i]]
    ids = ["ArrayTaskID"] + [i for i in range(len(dates)-1)]

    filename = f"input_params_{TC_id}_step_{step}_max_{max_lead}h.txt"
    with open(output_path + filename, "w") as w:
        col_format = "{:<12}" + "{:<9}" + "{:<5}" + "{:<9}" + "\n"
        if debug:
            data = np.column_stack((ids[:10], dates[:10], times[:10], lead_times[:10]))
        else:
            data = np.column_stack((ids, dates, times, lead_times))
        for x in data:
            w.write(col_format.format(*x))
            

def write_several_seasons(output_path:str, seasons:list=[2016,2017,2018,2019,2020], step=6, max_lead=168, 
                          ibtracs_path:str="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv",
                          **kwargs):
    
    ibtracs_df = pd.read_csv(ibtracs_path, dtype="string", na_filter=False)
    if ibtracs_path[-6:-4]=="00":
        ibtracs_df = ibtracs_df.loc[1:]
    
    all_tcs = kwargs.get("all_tcs", False)
    as_range = kwargs.get("as_range", False)
    
    if as_range:
        seasons = list(range(int(seasons[0]), int(seasons[-1])+1))
    print(f"Seasons: {seasons}")
    
    inputs = []
    for season in seasons:
        df_year = ibtracs_df[ibtracs_df['SEASON']==str(season)]
        basins = df_year['BASIN'].unique()
        
        for basin in basins:
            sids = df_year[df_year['BASIN']==basin]['SID'].unique()
            if not all_tcs:
                sids = [random.sample(list(sids), 1)[0]] # select one tc for each basin randomly
            
            for sid in sids:
                fname = output_path + f"{season}/input_params_{sid}_step_{step}_max_{max_lead}h.txt"
                if not os.path.isfile(fname):
                    if not os.path.isdir(output_path+f"{season}/"):
                        os.mkdir(output_path+f"{season}/")
                    write_params_for_tc(output_path+f"{season}/", ibtracs_df, TC_id=sid, season=season, step=step, max_lead=max_lead)
                inputs.append(fname)
    return sorted(list(set(inputs))) # some TCs may change basin during their lifetime, so we need to remove duplicates



def write_one_year(output_path:str, season:int=2000, step:int=6, max_lead:int=168,
                   ibtracs_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv"):
    df = pd.read_csv(ibtracs_path, dtype="string", na_filter=False)
    df = df[df["SEASON"]==str(season)]
    iso_times = df["ISO_TIME"].unique()
    
    dates_times_lead_times = set()
    
    for iso_time in iso_times:
        sids = df[df["ISO_TIME"]==iso_time]["SID"].unique()
        for sid in sids:
            end_tc = df[df["SID"]==sid]["ISO_TIME"].values[-1]
            date, time = date_ibtracs_to_nn(iso_time.split(" ")[0]), time_ibtracs_to_nn(iso_time.split(" ")[1])
            ldt = min(int(subtract_ibtracs_iso_times(iso_time, end_tc)), max_lead)
            if ldt == 0:
                continue
            dates_times_lead_times.add((date, time, ldt))
    
    key = lambda x: (x[0], x[1], x[2])
    dates_times_lead_times = sorted(list(dates_times_lead_times), key=key)
    n = len(dates_times_lead_times)
    fnames = [output_path + f"{season}/input_params_{season}_step_{step}_max_{max_lead}h#{i}.txt" \
                for i in range(int(np.ceil(n/300)))]
    
    dates, times, ldts = zip(*dates_times_lead_times)
    dates, times, ldts = list(dates), list(times), list(ldts)
    if not os.path.isdir(output_path + f"{season}/"):
        os.mkdir(output_path + f"{season}/")
    for i, fname in enumerate(fnames):
        if not os.path.isfile(fname):
            with open(fname, "w") as w:
                col_format = "{:<12}" + "{:<9}" + "{:<5}" + "{:<9}" + "\n"
                
                dates_tmp, times_tmp, ldts_tmp = dates[i*300:(i+1)*300], times[i*300:(i+1)*300], ldts[i*300:(i+1)*300]
                l = len(dates_tmp)
                data = np.row_stack((["ArrayTaskID", "date", "time", "lead time"],np.column_stack((np.arange(0,l), dates_tmp, times_tmp, ldts_tmp))))
                for x in data:
                    w.write(col_format.format(*x))
    return fnames


## FOR MONIKA'S PROJECT

def write_params_for_period(output_path, start_date:str="20200301", start_time="0000", end_date:str="20200831", 
                            end_time="0000", max_lead:int=168, step:int=12, **kwargs):
    # step: start a new forecast every <step> hours
    # complementary to write_input_params_to_file but this function does not take care if th date/time is in IBTrACS (more general)
    
    debug = kwargs.get("debug", False)
    filename = f"input_params_{start_date}T{str(int(start_time)).zfill(2)}_to_{end_date}T{str(int(end_time)).zfill(2)}_step_{step}_{max_lead}.txt"
    
    dates, times = ["date", start_date], ["time", start_time]
    end_iso_time = np.datetime64(date_time_nn_to_netcdf(end_date, end_time, -step), 'h') # last starting date can only be -step hours before the last known location of the TC
    current_date = date_time_nn_to_netcdf(dates[-1], times[-1])
    
    ids, ldts = ["ArrayTaskID", 0], ["lead time", min(subtract_ibtracs_iso_times(current_date, end_iso_time), max_lead)]
    
    while subtract_ibtracs_iso_times(current_date, end_iso_time) >= max_lead: # >= because last iso correspond to last_time-step
        
        current_date = current_date + np.timedelta64(step, 'h')
        
        ids.append(ids[-1]+1)
        dates.append(str(current_date).split("T")[0].replace("-", ""))
        times.append(str(int(str(current_date).split("T")[1])*100).zfill(4))
        ldts.append(max_lead)
    
    ldt = max_lead
    while subtract_ibtracs_iso_times(current_date, end_iso_time) >= step: # >= because last iso correspond to last_time-step
        
        current_date = current_date + np.timedelta64(step, 'h')
        ldt = ldt-step
        
        ids.append(ids[-1]+1)
        dates.append(str(current_date).split("T")[0].replace("-", ""))
        times.append(str(int(str(current_date).split("T")[1])*100).zfill(4))
        ldts.append(ldt)
        
        
    with open(output_path+filename, 'w') as w:
        col_format = "{:<12}" + "{:<9}" + "{:<5}" + "{:<9}" + "\n"
    
        if debug:
            data = np.column_stack((ids[:10], dates[:10], times[:10], ldts[:10]))
        else:
            data = np.column_stack((ids, dates, times, ldts))
        for x in data:
            w.write(col_format.format(*x))
