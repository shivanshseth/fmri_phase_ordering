#!/bin/bash
#SBATCH -n 32
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=2048
#SBATCH --time=72:00:00
#SBATCH --mail-user=shivansh.seth@research.iiit.ac.in


mkdir -p /scratch/shivansh.seth
mkdir -p /scratch/shivansh.seth/adni
scp -r ada:/share1/shivansh.seth/te_env.tar.gz /scratch/shivansh.seth/te_env.tar.gz
tar -xf /scratch/shivansh.seth/te_env.tar.gz -C /scratch/shivansh.seth

file="/scratch/shivansh.seth/adni/preproc_output.tar.gz"
if [ ! -f "$file" ]; then
    scp -r ada:/share1/shivansh.seth/ADNI/preproc_output.tar.gz /scratch/shivansh.seth/adni/preproc_output.tar.gz
    tar -xf /scratch/shivansh.seth/adni/preproc_output.tar.gz -C /scratch/shivansh.seth/adni
fi
# /scratch/shivansh.seth/preproc/bin/python spatial_correlations.py
