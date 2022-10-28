#!/bin/bash
#SBATCH --job-name=mae_pretr
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=2-00:00:00

#SBATCH --ntasks-per-gpu=5
#SBATCH --mem-per-cpu=16g
#SBATCH --account=precisionhealth_owned1
#SBATCH --partition=precisionhealth

##SBATCH --account=precisionhealth_project1
##SBATCH --partition=gpu
##SBATCH --ntasks-per-gpu=4
##SBATCH --mem-per-cpu=7g

##SBATCH --exclude=armis[28000,28001,28004]

#SBATCH --array=0-0
#SBATCH --output=./slurm_out/%x-%A_%a.out
source activate torchsrh

python main_simsiam.py \
  -a resnet50 \
  --dist-url 'tcp://localhost:32609' --multiprocessing-distributed --world-size 1 --rank 0 \
  --fix-pred-lr  
