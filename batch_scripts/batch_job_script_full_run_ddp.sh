#!/bin/bash
#SBATCH --job-name=foa_test_run
#SBATCH --account=project_2013256
#SBATCH --partition=gpumedium
#SBATCH --time=00:30:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4 --cpus-per-task=72  # The product should be 288
#SBATCH --gres=gpu:gh200:4  # 4 GPUs per node
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module purge
module load pytorch/2.10

source /projappl/project_2013256/lempio/hoa_env1/bin/activate

# Set the number of CPU threads based on cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "Running on $(hostname)"
nvidia-smi

# Run training (UNBUFFERED!)
srun PYTHONUNBUFFERED=1 torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py configs/train_cluster_foa.yaml