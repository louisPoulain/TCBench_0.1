#!/bin/bash -l

# --mail-type ALL 
# --mail-user louis.poulain-auzeau@unil.ch

#SBATCH --chdir /work/FAC/FGSE/IDYST/tbeucler/default/louis/
#SBATCH --job-name 4castv2
#SBATCH --output /work/FAC/FGSE/IDYST/tbeucler/default/louis/out_files/4castv2.out
# -- error /work/FAC/FGSE/IDYST/tbeucler/default/louis/repos/2021_TCGraphNet/dev/logs/script_12_err-%j.err

#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task 12
#SBATCH --mem 64G 
#SBATCH --time 05:00:00 
#SBATCH --export NONE

# clearing modules and loading python
module purge
module load gcc cuda cudnn
source activate /work/FAC/FGSE/IDYST/tbeucler/default/louis/ecmwf_venv/

# target_script=/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/2021_TCGraphNet/dev/percentile_sample_approach.py
ai-models --download-assets --assets ./fourcastnetv2/ --input cds --date 20180110 --time 0000 --lead-time 24 --path './fourcastnetv2/4castnetv2-{step}h.grib' fourcastnetv2

conda deactivate