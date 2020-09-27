#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=10G
#SBATCH --time=40:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=gendata
#SBATCH -o ./gendata.out
#SBATCH -e ./gendta.out

python run_domrand.py --gpu_render 1