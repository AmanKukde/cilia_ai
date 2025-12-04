#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/home/aman.kukde/cilia_ai/sam_runs/logs/%x_%j.log
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=SAM_Cilia_Segmentation
#SBATCH --time=12:00:00

cd /home/aman.kukde/cilia_ai/sam2/
source /home/aman.kukde/cilia_ai/.venv/bin/activate
# Run inference
python /home/aman.kukde/cilia_ai/run_notebook.py --notebook "/home/aman.kukde/cilia_ai/sam2/sam2_feature_space_aman.ipynb" --outputdir "/home/aman.kukde/cilia_ai/sam_runs/notebooks/"