#!/bin/bash -l

# --mail-type ALL 
# --mail-user louis.poulain-auzeau@unil.ch

#SBATCH --chdir /work/FAC/FGSE/IDYST/tbeucler/default/louis/TCBench_0.1/
#SBATCH --job-name CNNTrain
#SBATCH --output /work/FAC/FGSE/IDYST/tbeucler/default/louis/TCBench_0.1/out_files/cnn_%A.out

#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task 4
#SBATCH --gpus 1
#SBATCH --mem 24G 
#SBATCH --time 24:00:00 
#SBATCH --export NONE

# clearing modules and loading python
module purge
module load gcc cuda cudnn proj cdo
eval "$(/dcsrsoft/spack/arolle/v1.0/spack/opt/spack/linux-rhel8-zen2/gcc-10.4.0/miniconda3-4.10.3-gpvric5au5ue2cp2qiiar6vijzx4ibnb/condabin/conda shell.bash hook)"
conda deactivate
conda activate /work/FAC/FGSE/IDYST/tbeucler/default/louis/ecmwf_venv/


DATA_PATH="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/"
DF_PATH="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv"
SAVE_PATH="/users/lpoulain/louis/plots/cnn/"

SPLIT_RATIO=$1
MODEL=$2
SEASONS=$3
PRES=$4

MODEL_ARGS=$5
OPTIM=$6
SCHED=$7
CRPS=$8
RETRAIN=$9

python3 -u /users/lpoulain/louis/TCBench_0.1/scripts/models/cnn_main.py --data_path="$DATA_PATH" --df_path="$DF_PATH" --save_path="$SAVE_PATH" \
                                                                --split_ratio="$SPLIT_RATIO" --model="$MODEL" --seasons="$SEASONS" --pres="$PRES" \
                                                                --model_args="$MODEL_ARGS" --optim="$OPTIM" --sched="$SCHED" --crps="$CRPS" --retrain="$RETRAIN"

conda deactivate
