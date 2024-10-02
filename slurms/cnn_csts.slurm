#!/bin/bash -l

# --mail-type ALL 
# --mail-user louis.poulain-auzeau@unil.ch

#SBATCH --chdir /work/FAC/FGSE/IDYST/tbeucler/default/louis/TCBench_0.1/
#SBATCH --job-name cnn_seasons_csts
#SBATCH --output /users/lpoulain/louis/plots/cnn/Constants/%A_%a.txt

#SBATCH --partition cpu
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task 1
#SBATCH --mem 32G 
#SBATCH --time 4:00:00 
#SBATCH --export NONE

# clearing modules and loading gcc
module purge
module load gcc
eval "$(/dcsrsoft/spack/arolle/v1.0/spack/opt/spack/linux-rhel8-zen2/gcc-10.4.0/miniconda3-4.10.3-gpvric5au5ue2cp2qiiar6vijzx4ibnb/condabin/conda shell.bash hook)"
conda deactivate
conda activate /work/FAC/FGSE/IDYST/tbeucler/default/louis/ecmwf_venv/

seasons=$1
model=$2
pres=$3

# Remove the brackets from the seasons string
seasons=${seasons:1:-1}

# Convert the seasons string to an array
IFS=',' read -r -a seasons_array <<< "$seasons"

# Iterate over the seasons array
season="[${seasons_array[$SLURM_ARRAY_TASK_ID]}]"

python3 -u scripts/models/cnn_csts.py --model_name $model --seasons $season --pres $pres
echo "Finished"
conda deactivate