#!/bin/bash
# The interpreter used to execute the script

#SBATCH --job-name=593_group5
#SBATCH --mail-user=kulkarnr@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --gpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-gpu=30g
#SBATCH --time=08:00:00
#SBATCH --account=bioinf593f23_class
#SBATCH --partition=gpu
#SBATCH --output=/nfs/turbo/dcmb-class/bioinf593/groups/group_05/training.log

eval "$(conda shell.bash hook)"
conda activate scvi-env

# The application(s) to execute along with its input arguments and options:
python3 Baseline_CNN.py
/bin/hostname
sleep 60
