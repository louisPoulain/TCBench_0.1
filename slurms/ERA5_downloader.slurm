#!/bin/bash -l

# --mail-type ALL 
# --mail-user louis.poulain-auzeau@unil.ch

#SBATCH --chdir /work/FAC/FGSE/IDYST/tbeucler/default/louis/TCBench_0.1/
#SBATCH --job-name era5_downloader
#SBATCH --output /users/lpoulain/louis/TCBench_0.1/era5_download_%a.txt

#SBATCH --partition cpu
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task 8
#SBATCH --mem 8G 
#SBATCH --time 72:00:00 
#SBATCH --export NONE
#SBATCH --array=0-12

# clearing modules and loading python
module purge
module load gcc cuda
eval "$(/dcsrsoft/spack/arolle/v1.0/spack/opt/spack/linux-rhel8-zen2/gcc-10.4.0/miniconda3-4.10.3-gpvric5au5ue2cp2qiiar6vijzx4ibnb/condabin/conda shell.bash hook)"
conda deactivate
conda activate /work/FAC/FGSE/IDYST/tbeucler/default/louis/e2mip_venv/

config=/users/lpoulain/louis/TCBench_0.1/input_params/input_era5_downloader.txt

# Extract the date, time and lead time for the current $SLURM_ARRAY_TASK_ID
min_year=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)

min_month=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

echo "Starting ERA5 download (upper) for year $min_year and month $min_month"
python3 scripts/ERA5_downloader.py --min_year "$min_year" --min_month "$min_month"
echo "Finished"
conda deactivate
