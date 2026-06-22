#!/bin/bash
#SBATCH --job-name=model_test_run
#SBATCH --account=project_2013256
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=2-0:00:00
#SBATCH --gres=gpu:v100:1,nvme:10
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module purge
module load pytorch/2.6

source /projappl/project_2013256/lempio/hoa_env1/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Running on $(hostname)"
nvidia-smi

# Run training (UNBUFFERED!)
srun python3 -u train.py configs/train_cluster.yaml