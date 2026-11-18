#!/bin/bash
#SBATCH --job-name=foa_multinode_test_run
#SBATCH --account=project_2013256
#SBATCH --partition=gpularge
#SBATCH --time=01:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:gh200:4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module purge
module load python-pytorch/2.10

source /projappl/project_2013256/lempio/hoa_env1/bin/activate

# CPU threading
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 4))
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# Unbuffered logging
export PYTHONUNBUFFERED=1

# Distributed setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# Keep diagnostic logging quiet so the Slurm output stays readable.
export NCCL_DEBUG=WARN
unset NCCL_DEBUG_SUBSYS
unset TORCH_DISTRIBUTED_DEBUG

# DEBUG_MAX_STEPS is the absolute target step, not an additional step count.
# For a step-40000 checkpoint, set DEBUG_MAX_STEPS=40200 before submission
# to run 200 training steps.
export DEBUG_MAX_STEPS="${DEBUG_MAX_STEPS:-1000}"
export DEBUG_TIMING="${DEBUG_TIMING:-0}"

echo "Running on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"

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
echo "Absolute target step: $DEBUG_MAX_STEPS"

srun --ntasks="$SLURM_NNODES" \
    --ntasks-per-node=1 \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    --kill-on-bad-exit=1 \
    torchrun \
        --nnodes="$SLURM_NNODES" \
        --nproc_per_node=4 \
        --node_rank="$SLURM_NODEID" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
        train.py configs/train_cluster_foa_ddp_multinode.yaml

echo "Finished: $(date)"