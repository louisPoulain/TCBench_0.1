#!/bin/bash -l

# --mail-type ALL 
# --mail-user louis.poulain-auzeau@unil.ch

#SBATCH --chdir /work/FAC/FGSE/IDYST/tbeucler/default/louis/TCBench_0.1/
#SBATCH --job-name fcnv2Multiple
#SBATCH --output /work/FAC/FGSE/IDYST/tbeucler/default/louis/TCBench_0.1/out_files/fcnv2_multiple_%A_%a.out

#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task 1
#SBATCH --mem 32G 
#SBATCH --time 01:40:00 
#SBATCH --export NONE

# clearing modules and loading python
module purge
module load gcc cuda proj cdo
eval "$(/dcsrsoft/spack/arolle/v1.0/spack/opt/spack/linux-rhel8-zen2/gcc-10.4.0/miniconda3-4.10.3-gpvric5au5ue2cp2qiiar6vijzx4ibnb/condabin/conda shell.bash hook)"
conda deactivate
conda activate /work/FAC/FGSE/IDYST/tbeucler/default/louis/e2mip_venv/


# Specify the path to the config file
config=$1 #./input_params_2019326S08163.txt
SAVE_DIR="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/fourcastnetv2"

# Extract the date, time and lead time for the current $SLURM_ARRAY_TASK_ID
date=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)

time=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

lead_time=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)

# Print to a file a message that includes the current $SLURM_ARRAY_TASK_ID, the same name, and the sex of the sample
# echo "This is array task ${SLURM_ARRAY_TASK_ID}, the date ${date} and the time is ${time}, the lt is ${lead_time}, file=${config}." >> "./output_gphcst.txt"
# echo ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

out=$(python3 /users/lpoulain/louis/TCBench_0.1/slurms/dates.py --date "$date" --time "$time" --lead_time "$lead_time")
#echo "test this is out: $out" | cat >> ./out_files/fcnv2_multiple_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
if ! compgen -G "${SAVE_DIR}/fourcastnetv2_${out}*.nc" > /dev/null; then

        echo "$date $time $lead_time $config $out" | cat >> ./out_files/fcnv2_multiple_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
        python3 slurms/fcnv2.py --date "$date" --time "$time" --ldt "$lead_time"
        wait $!
        echo "Filtering ..." | cat >> ./out_files/fcnv2_multiple_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
        python3 scripts/utils/filtering.py --params "$out" --model "fourcastnetv2"
        echo "Filtering done" | cat >> ./out_files/fcnv2_multiple_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
else
    rm ./out_files/fcnv2_multiple_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
fi

rm -rf /users/lpoulain/.cache/earth2mip/cds/
conda deactivate
