#!/bin/bash -l

# --mail-type ALL 
# --mail-user louis.poulain-auzeau@unil.ch

#SBATCH --chdir /work/FAC/FGSE/IDYST/tbeucler/default/louis/TCBench_0.1/
#SBATCH --job-name PanguMultiple
#SBATCH --output /work/FAC/FGSE/IDYST/tbeucler/default/louis/TCBench_0.1/out_files/pangu_multiple_%A_%a.out

#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task 1
#SBATCH --mem 32G 
#SBATCH --time 05:00:00 
#SBATCH --export NONE
# --array=0-50

# clearing modules and loading python
module purge
module load gcc cuda cudnn
eval "$(/dcsrsoft/spack/arolle/v1.0/spack/opt/spack/linux-rhel8-zen2/gcc-10.4.0/miniconda3-4.10.3-gpvric5au5ue2cp2qiiar6vijzx4ibnb/condabin/conda shell.bash hook)"
conda deactivate
conda activate /work/FAC/FGSE/IDYST/tbeucler/default/louis/ecmwf_venv/


# Specify the path to the config file
config=$1 #./input_params_2019326S08163.txt

# Extract the date, time and lead time for the current $SLURM_ARRAY_TASK_ID
date=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)

time=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

lead_time=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)

# Print to a file a message that includes the current $SLURM_ARRAY_TASK_ID, the same name, and the sex of the sample
# echo "This is array task ${SLURM_ARRAY_TASK_ID}, the date ${date} and the time is ${time}, the lt is ${lead_time}, file=${config}." >> "./output_pangu.txt"

if [[ -n "$time" ]] # -n tests if the string is NOT empty, -z if the string IS empty
then
    echo "$date $time $lead_time $config" | cat >> ./out_files/pangu_multiple_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
    ONNXRUNTIME=onnxruntime-gpu ai-models --assets ../panguweather/ --input cds --date "$date" --time "$time" --lead-time "$lead_time" \
  --path "/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/panguweather/pangu_d_${date}_t_${time}_{step}h.grib" panguweather
else
        echo "removed ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ($date $time $lead_time $config)" | cat >> ./out_files/removed_jobs.txt
        scancel ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
        rm ./out_files/pangu_multiple_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
fi

conda deactivate
