#!/bin/bash -l

# --mail-type ALL 
# --mail-user louis.poulain-auzeau@unil.ch

#SBATCH --chdir /work/FAC/FGSE/IDYST/tbeucler/default/louis/TCBench_0.1/
#SBATCH --job-name panguTest
#SBATCH --output /work/FAC/FGSE/IDYST/tbeucler/default/louis/TCBench_0.1/out_files/pangu.out
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
eval "$(/dcsrsoft/spack/arolle/v1.0/spack/opt/spack/linux-rhel8-zen2/gcc-10.4.0/miniconda3-4.10.3-gpvric5au5ue2cp2qiiar6vijzx4ibnb/condabin/conda shell.bash hook)"
source activate /work/FAC/FGSE/IDYST/tbeucler/default/louis/ecmwf_venv/

# target_script=/work/FAC/FGSE/IDYST/tbeucler/default/milton/repos/2021_TCGraphNet/dev/percentile_sample_approach.py

ai-models --assets ../panguweather/ --input cds --date 20180110 --time 0000 --lead-time 30\
            --path '/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/panguweather/Testpangu_{step}h.grib' panguweather

conda deactivate