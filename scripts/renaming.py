import scripts.utils as ut
import argparse

parser = argparse.ArgumentParser(description='Renaming script')
parser.add_argument('--model',type=str,help='model name')
parser.add_argument('--remove',type=bool,help='whether to remove old files', default=False)
parser.add_argument('--year',type=int,help='rename for TC starting that year', default=2000)

args = parser.parse_args()
model, remove, year = args.model, args.remove, args.year

ut.renaming(folder_name="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/",
            model=model, year=year, remove_old=remove, cut=True)