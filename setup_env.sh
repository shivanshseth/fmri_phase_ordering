#!/bin/bash
#SBATCH -n 20
#SBATCH --mem-per-cpu=2048
#SBATCH --time=72:00:00
#SBATCH --mail-user=shivansh.seth@research.iiit.ac.in

mkdir -p /scratch/shivansh.seth
mkdir -p /scratch/shivansh.seth/adni

# Env
 #scp -r ada:/share1/shivansh.seth/preproc.env.tar /scratch/shivansh.seth/preproc.env.tar
 #tar -xf /scratch/shivansh.seth/preproc.env.tar -C /scratch/shivansh.seth
 #scp -r ada:/share1/shivansh.seth/te_env.tar.gz /scratch/shivansh.seth/te_env.tar.gz
 #tar -xf /scratch/shivansh.seth/te_env.tar.gz -C /scratch/shivansh.seth

# Data
 #scp -r ada:/share1/shivansh.seth/ADNI/AD.zip /scratch/shivansh.seth/AD.zip
 #scp -r ada:/share1/shivansh.seth/ADNI/AD_anat.zip /scratch/shivansh.seth/AD_anat.zip
 #scp -r ada:/share1/shivansh.seth/ADNI/CN.zip /scratch/shivansh.seth/CN.zip
# scp -r ada:/share1/shivansh.seth/ADNI/CN_anat.zip /scratch/shivansh.seth/CN_anat.zip
 #cd /scratch/shivansh.seth
# unzip -q AD.zip
# unzip -q AD_anat.zip
# unzip -q CN.zip
# unzip -q CN_anat.zip
# cd -

# Preproced Data
 scp -r ada:/share1/shivansh.seth/ADNI/preproc_output.tar.gz /scratch/shivansh.seth/adni/preproc_output.tar.gz
 tar -xf /scratch/shivansh.seth/adni/preproc_output.tar.gz -C /scratch/shivansh.seth/adni
# cd ~/phase_diagram_analysis
#/scratch/shivansh.seth/preproc/bin/python calc_rts.py 
