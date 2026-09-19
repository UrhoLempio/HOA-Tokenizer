#!/bin/bash
#SBATCH --job-name=foa_multinode_test_run
#SBATCH --account=project_2013256
#SBATCH --partition=gpularge
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1 --cpus-per-task=288  # The product should be 288
#SBATCH --gres=gpu:gh200:4  # 4 GPUs per node
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module purge
module load python-pytorch/2.10

source /projappl/project_2013256/lempio/hoa_env1/bin/activate

# Set the number of CPU threads based on cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Run training (UNBUFFERED!)
export PYTHONUNBUFFERED=1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

export OMP_PLACES=cores
export OMP_PROC_BIND=spread

echo "Running on $(hostname)"
nvidia-smi

#debugging information for distributed training
echo "NODE:$HOSTNAME RANK:$SLURM_PROCID LOCAL_RANK:$LOCAL_RANK WORLD_SIZE:$WORLD_SIZE"

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --node-rank=$SLURM_NODEID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    train.py configs/train_cluster_foa_ddp_multinode.yaml