#!/bin/bash -l

# --mail-type ALL 
# --mail-user louis.poulain-auzeau@unil.ch

#SBATCH --chdir /work/FAC/FGSE/IDYST/tbeucler/default/louis/TCBench_0.1/
#SBATCH --job-name renamer
#SBATCH --output /work/FAC/FGSE/IDYST/tbeucler/default/louis/TCBench_0.1/out_files/renaming_%A.out

#SBATCH --partition cpu
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task 8
#SBATCH --mem 64G 
#SBATCH --time 48:00:00 
#SBATCH --export NONE

# clearing modules and loading python
module purge
module load gcc cuda cudnn proj cdo
eval "$(/dcsrsoft/spack/arolle/v1.0/spack/opt/spack/linux-rhel8-zen2/gcc-10.4.0/miniconda3-4.10.3-gpvric5au5ue2cp2qiiar6vijzx4ibnb/condabin/conda shell.bash hook)"
conda deactivate
conda activate /work/FAC/FGSE/IDYST/tbeucler/default/louis/ecmwf_venv/


model=$1
remove=$2
year=$3

python3 -u scripts/renaming.py --model "$model" --remove "$remove" --year "$year"