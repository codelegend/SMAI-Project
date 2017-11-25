#!/bin/bash
#######################################
# Script for sbatch (Slurm batch run) #
########################################

#SBATCH -A $USER
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=aclImdb_init_run
#SBATCH --time=10:00
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=1024

module add cuda/8.0
module add cudnn/7-cuda-8.0

# wordvec generation
python make.py preprocess --dataset=aclImdb --parser=aclImdb --model=aclImdb

# training
python make.py train --dataset=aclImdb --parser=aclImdb --model=aclImdb --output=run_25_11_01

# testing
python make.py test --dataset=aclImdb --parser=aclImdb --model=aclImdb --load-from=run_25_11_01_final
