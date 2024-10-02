from argparse import ArgumentParser
from cnn_loaders import CNN4PP_Dataset
import sys
from utils.main_utils import str2list, str2bool

parser = ArgumentParser()

parser.add_argument("--data_path", type=str, default="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/")
parser.add_argument("--ibtracs_path", type=str, default="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv")
parser.add_argument("--save_path", type=str, default="/users/lpoulain/louis/plots/cnn/")

parser.add_argument("--model_name", type=str, default="graphcast")
parser.add_argument("--seasons", type=str2list, default="[2000,2001,2002]")

parser.add_argument("--pres", type=str2bool, help="whether to use pressure data", default="False")

args = parser.parse_args()
print(args)

data_path, ibtracs_path, save_path = args.data_path, args.ibtracs_path, args.save_path

model_name, seasons, pres = args.model_name, args.seasons, args.pres

data = CNN4PP_Dataset(data_path, model_name, ibtracs_path, seasons, pres, save_path, train_seasons=seasons)
                 