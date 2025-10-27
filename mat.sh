#!/bin/bash
#SBATCH --output=output_2.txt
#SBATCH -n 30
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=72:00:00
#SBATCH --mail-user=shivansh.seth@research.iiit.ac.in

module load u18/matlab/R2022a
matlab -nodisplay -nosplash -nodesktop -r "run('example_02_PhaseDiagram.m'); exit"
