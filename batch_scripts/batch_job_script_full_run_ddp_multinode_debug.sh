#!/bin/bash
#SBATCH --job-name=foa_multinode_test_run
#SBATCH --account=project_2013256
#SBATCH --partition=gpularge
#SBATCH --time=36:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1 --cpus-per-task=64
#SBATCH --gres=gpu:gh200:4  # 4 GPUs per node
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module purge
module load python-pytorch/2.10

source /projappl/project_2013256/lempio/hoa_env1/bin/activate

# Set the number of CPU threads based on cpus-per-task.
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 4))
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# Run training unbuffered.
export PYTHONUNBUFFERED=1

# Distributed setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

echo "Running on $(hostname)"
nvidia-smi

echo "NODE:$HOSTNAME SLURM_PROCID:${SLURM_PROCID:-unset} SLURM_NODEID:${SLURM_NODEID:-unset} SLURM_NNODES:${SLURM_NNODES:-unset}"

# Keep all training outputs unique to this Slurm job and node count.
RUN_ROOT="${SCALING_OUTPUT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}/scaling_runs}"
RUN_NAME="${SLURM_JOB_NAME:-foa_scaling}_${SLURM_NNODES}n_${SLURM_JOB_ID}"
export HOA_RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
mkdir -p "$HOA_RUN_DIR"

if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    export HOA_RESUME_CHECKPOINT="$RESUME_CHECKPOINT"
fi

echo "Run output: $HOA_RUN_DIR"
echo "Resume checkpoint: ${HOA_RESUME_CHECKPOINT:-auto-resume from this run checkpoint directory}"

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --node-rank=$SLURM_NODEID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    train.py configs/train_cluster_foa_ddp_multinode.yaml
