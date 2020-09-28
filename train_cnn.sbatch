#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=10G
#SBATCH --time=40:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=gendata
#SBATCH -o ./traincnn.out
#SBATCH -e ./traincnn.out

python run_training.py --num_files=20 --output=xyz