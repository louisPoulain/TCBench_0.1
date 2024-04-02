from argparse import ArgumentParser
import subprocess
from utils.main_utils import str2list

parser = ArgumentParser()
    
# data params
parser.add_argument("--split_ratio", type=str, default="[0.7,0.2,0.1]")

# AI models params
parser.add_argument("--model", type=str, default="graphcast")
parser.add_argument("--seasons", type=str, default="[2000,2001,2002]", 
                    help="List of seasons to use for the train/val/test data. If --retrain, the seasons are taken to be ONLY train seasons")
parser.add_argument("--pres", help="whether to use pressure data", action="store_true")

# cnn model params
parser.add_argument("--model_args", type=str, default="[lr,0.01,epochs,10]",
                    help="List of arguments for the model, in the form [arg1,val1,arg2,val2, ...]")
#parser.add_argument("--type", type=str, default="mlp", choices=["mlp", "mlp_normal", "mlp_shash"])
parser.add_argument("--optim", type=str, default="adam")
parser.add_argument("--sched", type=str, default="cosine_annealing")
parser.add_argument("--crps", action="store_true")
parser.add_argument("--retrain", action="store_true", help="continue training from a previous model")

args = parser.parse_args()
print(args)

subprocess.run(["sbatch", "/users/lpoulain/louis/TCBench_0.1/slurms/cnn.slurm", args.split_ratio, args.model, args.seasons, 
                            str(args.pres), args.model_args, args.optim, args.sched, str(args.crps), str(args.retrain)])