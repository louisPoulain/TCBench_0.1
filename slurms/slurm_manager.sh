#!/bin/bash 

# you can include #SBATCH comments here if you like, but any that are
# specified on the command line or in SBATCH_* environment variables
# will override whatever is defined in the comments.  You **can't**
# use positional parameters like $1 or $2 in a comment - they won't
# do anything.

# Load module
module load gcc
eval "$(/dcsrsoft/spack/arolle/v1.0/spack/opt/spack/linux-rhel8-zen2/gcc-10.4.0/miniconda3-4.10.3-gpvric5au5ue2cp2qiiar6vijzx4ibnb/condabin/conda shell.bash hook)"

model=$1
input=$2
array_len=$(($3 - 1))
cds_arg=$4


model_pangu="panguweather"
model_graphcast="graphcast"
model_fcnv2="fourcastnetv2"
#-$array_len
if [[ "$model" == "$model_graphcast" ]] ;
then
    echo "graphcast $input" "$cds_arg"
    sbatch --array=0-$array_len%4 /users/lpoulain/louis/TCBench_0.1/slurms/graphcast_array${cds_arg}.slurm $input
else 
    if [[ "$model" == "$model_fcnv2" ]] ;
    then
        echo "fourcastnetv2 $input" "$cds_arg"
        sbatch --array=0-$array_len /users/lpoulain/louis/TCBench_0.1/slurms/fcnv2_array.slurm $input
    else
        if [[ "$model" == "$model_pangu" ]] ;
        then
            echo "panguweather $input" "$cds_arg"
            sbatch --array=0-$array_len /users/lpoulain/louis/TCBench_0.1/slurms/pangu_array${cds_arg}.slurm $input
        else
            echo "model $model not recognized. Valid choices are $model_pangu, $model_graphcast and $model_fcnv2."
        fi
    fi
fi
