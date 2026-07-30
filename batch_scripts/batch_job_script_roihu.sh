#!/bin/bash
#SBATCH --job-name=foa_test_run
#SBATCH --account=project_2013256
#SBATCH --partition=gpumedium
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module purge
module load pytorch/2.6

source /projappl/project_2013256/lempio/hoa_env1/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Running on $(hostname)"
nvidia-smi

# Run training (UNBUFFERED!)
srun python3 -u train.py configs/train_cluster_foa.yaml