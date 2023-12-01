#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=runmxnet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4000m 
#SBATCH --time=03:00
#SBATCH --account=eecs471f23_class
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1

module load gcc/8.2.0
module load cudnn/11.6-v8.4.1
module load cuda/11.6.2
module load mkl/2022.0.2

#nsys profile -o nsys_log/report.${SLURM_JOB_ID} --force-overwrite true --trace cuda --stats true python2 submit/submission.py
nvprof python2 submit/submission.py
#python2 submit/submission.py
