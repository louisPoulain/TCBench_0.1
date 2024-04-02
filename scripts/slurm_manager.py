import subprocess, sys
from argparse import ArgumentParser
from params_writers import write_several_seasons, write_one_year

def str2list(li):
    if type(li)==list:
        li2=li
        return li2
    elif type(li)==str:
        
        if ', ' in li :
            li2=li[1:-1].split(', ')
        else :
            
            li2=li[1:-1].split(',')
        return li2
    
    else:
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))


parser = ArgumentParser()

parser.add_argument("--seasons", type=str2list, help="input seasons, pass it as a list", default=[2000])
parser.add_argument("--models", type=str2list, help="input models, pass it as a list", default=['graphcast'])
parser.add_argument("--range", help="whether seasons should be interpreted as a range", action="store_true")
parser.add_argument("--all_tcs", help="whether to use all tcs of the season or one at random per basin", action="store_true")
parser.add_argument("--input", help="'cds' or 'local'. Note that fcnv can only be run with cds currently", default="local")

args = parser.parse_args()
cds = "" if args.input=="local" else "cds"

print("Input args:\n", args)

seasons = args.seasons
if args.range:
    seasons = list(range(int(seasons[0]), int(seasons[-1])+1))
models = args.models
if "fourcastnetv2" in models:
    models.remove("fourcastnetv2")
    print("Currently need to find a better way for jobs with fcnv2")
    
if not args.all_tcs:
    inputs = write_several_seasons(output_path="/users/lpoulain/louis/TCBench_0.1/input_params/", 
                      seasons=seasons, step=6, max_lead=168, all_tcs=args.all_tcs,
                      ibtracs_path='/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv')

else:
    assert len(seasons)==1, "all_tcs can only be used for one season at a time"
    
    #key = lambda x: (os.path.basename(x).split("#")[1].split(".")[0])
    inputs = write_one_year(output_path="/users/lpoulain/louis/TCBench_0.1/input_params/", season=seasons[0], step=6, max_lead=168,
                   ibtracs_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv")
    #inputs = sorted(glob.glob(f"/users/lpoulain/louis/TCBench_0.1/input_params/{seasons[0]}/input_params_{seasons[0]}_*#*.txt"), key=key)
    
inputs_len = [len(open(input).readlines())-1 for input in inputs]

c = {}
for model in models:
    c[model] = 0

print(f"Number of inputs: {len(inputs)}")
# (fcn gcp, pgw) -> 2000: (0,1,1), 2001: (0,1,1), 2002: (0,1,1), 2003: (0,1,1), 2004: (0,1,1), 2005: (0,1,1), 2006: (0,1,1)
#                   2007: (0,1,1), 2008: (0,1,1), 2009: (0,[0,1,2,3,4],1), 2010: (0,0,1), 2011: (0,0,1), 2012: (0,0,1), 2013: (0,0,1), 2014: (0,0,1), 2015: (0,0,[1]), 2016: (0,0,[1])
#                   2017: (0,0,1), 2018: (0,0,1), 2019: (0,0,1), 2020: (0,0,0), 2021: (0,0,0), 2022: (0,0,0), 2023: (0,0,0)
k = 4
#inputs = [inputs[k]]
for i, input in enumerate(inputs):
    l = inputs_len[i] if len(inputs)>1 else inputs_len[k]
    for model in models:
        #sid = input.split("_")[2]
        model_name = "pangu" if model=="panguweather" else model
        
        subprocess.run(["bash", "/users/lpoulain/louis/TCBench_0.1/slurms/slurm_manager.sh", model, input, str(l), cds])
        c[model] += l
        
print(f"Number of jobs submitted: {c}")
