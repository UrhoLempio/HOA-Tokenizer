#!/bin/bash
#SBATCH --job-name=myTestJob
#SBATCH --account=project_2013256
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=15
#SBATCH --gres=gpu:v100:1,nvme:10

module purge
module load ffmpeg
module load pytorch/2.6 
source /projappl/project_2013256/lempio/HOA-Tokenizer/new_env/bin/activate

srun python3 train.py configs/train_cluster.yaml

seff $SLURM_JOBID
