from argparse import ArgumentParser
from utils.main_utils import str2list, str2bool, str2intlist
from case_study_plot import trajectory, trajectory_with_pp

func_dict = {
    "trajectory": trajectory,
    "trajectory_with_pp": trajectory_with_pp
}
parser = ArgumentParser()


# data paths

parser.add_argument("--data_path", type=str, default="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/", help="Path to the data folder.")
parser.add_argument("--df_path", default="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv")
parser.add_argument("--plot_path", default="/users/lpoulain/louis/plots/case_studies/")


# function to use

parser.add_argument("--func", type=str, default="trajectory", help="Function to use for the case study.", choices=func_dict.keys())

# function arguments

parser.add_argument("--tc_id", type=str, default="2005236N23285", help="ID of the tropical cyclone to plot.", 
                    choices=["2000185N15117", "2005236N23285", "2008278N13261"])
parser.add_argument("--model_names", type=str2list, default=["pangu","graphcast","fourcastnetv2"])
parser.add_argument("--max_lead", type=str2intlist, default=[72], help="Maximum lead time to plot.")
parser.add_argument("--ldt_step", type=int, default=6, help="Time step between lead times.")

# specific arguments for pp plot

parser.add_argument("--pp_type", type=str, default="none", help="Type of post-processing to use.", choices=["linear", "xgboost", "cnn", 'none'])
parser.add_argument("--train_seasons", type=str2list, default=[])
parser.add_argument("--basin", type=str, default="all", help="Basin of the tropical cyclone.", choices=["NA", "WP", "EP", "NI", "SI", "SP", "SA", 'all'])

## for linear pp
parser.add_argument("--dim", type=int, default=2, help="Dimension of the linear post-processing.", choices=[1,2])

## for xgboost pp
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for the xgboost model.")
parser.add_argument("--depth", type=int, default=3, help="Depth of the trees for the xgboost model.")
parser.add_argument("--epochs", type=int, default=200, help="Number of epochs for the xgboost model.")
parser.add_argument("--gamma", type=float, default=0.0, help="Gamma for the xgboost model.")
parser.add_argument("--jsdiv", action="store_true", help="Use Jensen-Shannon divergence as loss function")
parser.add_argument("--sched", action="store_true", help="Use a scheduler for the learning rate [cos_annealing]")
parser.add_argument("--stats", type=str2list, default=[], help="Statistics to use for the xgboost post-processing.")
parser.add_argument("--stats_wind", type=str2list, default=[], help="Statistics to use for the wind in the xgboost post-processing.")
parser.add_argument("--stats_pres", type=str2list, default=[], help="Statistics to use for the pressure in the xgboost post-processing.")

## for cnn pp

# you need lr, epochs, sched
parser.add_argument("--optim", type=str, default="adam")
parser.add_argument("--crps", action="store_true", help="Use CRPS as loss function")
parser.add_argument("--pres", type=str2bool, default=True, help="Use pressure as input for the CNN.")


args = parser.parse_args()

pp_params = {"dim": args.dim,
             "jsdiv": args.jsdiv,
             "sched": args.sched,
             "stats": args.stats,
             "stats_wind": args.stats_wind,
             "stats_pres": args.stats_pres,
             "lr": args.lr,
             "depth": args.depth,
             "epochs": args.epochs,
             "gamma": args.gamma,
             "train_seasons": args.train_seasons if '' not in args.train_seasons and ' ' not in args.train_seasons else [],
             "basin": args.basin,
             "optim": args.optim,
             "crps": args.crps,
             "pres": args.pres}

tc_id = args.tc_id
model_names = args.model_names
max_lead = args.max_lead
if len(max_lead) == 1:
    max_lead = max_lead[0]
ldt_step = args.ldt_step

pp_type = args.pp_type if args.pp_type.lower() != "none" else None
if pp_type is not None and "fourcastnetv2" in model_names:
    raise ValueError("fourcastnetv2 does not support post-processing yet. Please remove it from the model_names list.")

print('\n',args)
func_dict[args.func](tc_id=tc_id, model_names=model_names, max_lead_time=max_lead, ldt_step=ldt_step, pp_type=pp_type, pp_params=pp_params, 
                     data_path=args.data_path, df_path=args.df_path, plot_path=args.plot_path)
