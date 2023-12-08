from argparse import ArgumentParser
import os
import datetime
import numpy as np

# INPUT ARGUMENTS   

parser = ArgumentParser()

parser.add_argument('--date', type=str, help="input date")
parser.add_argument('--time', type=str, help="input hours")

parser.add_argument('--ldt', type=str, help="lead time")

parser.add_argument('--save_path', type=str, help="save path")

args = parser.parse_args()

date, time, ldt = args.date, int(args.time)//100, args.ldt
year, month, day = date[:4], date[4:6], date[6:]

date_start = np.datetime64(f"{year}-{month}-{day}T{str(time).zfill(2)}")
date_end = np.datetime64(np.datetime64(f"{year}-{month}-{day}T{str(time).zfill(2)}") + np.timedelta64(int(ldt), 'h'), 'h')
save_path = f"/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/fourcastnetv2/fourcastnetv2_{date_start}_to_{date_end}_ldt_{ldt}_6.nc"

# MODEL PATH AND INPUT DATA

# Set number of GPUs to use to 1
os.environ['WORLD_SIZE'] = '1'

model_registry = "/work/FAC/FGSE/IDYST/tbeucler/default/louis/earth2mip/examples/models"
os.makedirs(model_registry, exist_ok=True)
os.environ['MODEL_REGISTRY'] = model_registry

### we can import only once the env var are set

from earth2mip import registry, inference_ensemble
from earth2mip.initial_conditions import cds
import earth2mip.networks.fcnv2_sm as fcnv2_sm

### ===========================================
    
cds_api = os.path.join(os.path.expanduser("/users/lpoulain/louis/"), '.cdsapirc')


# INFERENCE

# Load fcnv2 model(s) from registry

package = registry.get_model("fcnv2_sm")
fcnv2_sm_inference_model = fcnv2_sm.load(package)

# Initial state data/time
start_time = datetime.datetime(int(year), int(month), int(day), time)

# Pangu datasource, this is much simplier since pangu only uses one timestep as an input
fcnv2_sm_data_source = cds.DataSource(fcnv2_sm_inference_model.in_channel_names)

fcnv2_sm_ds = inference_ensemble.run_basic_inference(
    fcnv2_sm_inference_model,
    n=int(ldt)//6, # fcnv2 is at 6 hour dt
    data_source=fcnv2_sm_data_source,
    time=start_time,
)

fcnv2_sm_ds.to_netcdf(save_path)