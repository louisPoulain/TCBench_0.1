from cut_region import cut_and_save_rect
import pandas as pd
import argparse, subprocess



parser = argparse.ArgumentParser()

parser.add_argument("--params", type=str, help="Automatically created in the slurm, contains date_start, date_end and ldt")
parser.add_argument("--model", type=str, help="model name", default="graphcast")

args = parser.parse_args()

date_start, date_end, ldt = args.params.split("_")[0], args.params.split("_")[2], args.params.split("_")[4]


df = pd.read_csv("/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv", 
                 dtype="string", na_filter=False)

ds_folder = "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/"

tc_with_date_start = set(df[df["ISO_TIME"].astype("datetime64[ns]")==date_start]["SID"].unique())
tc_with_date_end = set(df[df["ISO_TIME"].astype("datetime64[ns]")==date_end]["SID"].unique())

tc_ids = list(tc_with_date_start.intersection(tc_with_date_end))

for tc_id in tc_ids:
    path = cut_and_save_rect(ds_folder, models=[args.model], df_tracks=df, date_start=date_start, date_end=date_end,
                      lead_time=ldt, tc_id=tc_id, output_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/")
print("Removing large file", flush=True)
subprocess.run(["rm", path])
print("Removed", flush=True)


