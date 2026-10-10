#!/bin/bash
#SBATCH --job-name=foa_multinode_test_run
#SBATCH --account=project_2013256
#SBATCH --partition=gpularge
#SBATCH --time=00:10:00
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

# NCCL and Torch distributed diagnostics
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Limit the diagnostic run while retaining the normal training behavior.
export DEBUG_MAX_STEPS=100
export DEBUG_TIMING=1

echo "Running on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"

nvidia-smi

echo "NODE:$HOSTNAME RANK:$SLURM_PROCID LOCAL_RANK:$LOCAL_RANK WORLD_SIZE:$WORLD_SIZE"

# Create logging directory
LOGDIR="profiling_${SLURM_JOB_ID}"
mkdir -p "$LOGDIR"

###########################################################
# Monitoring (runs in background)
###########################################################

# GPU utilization and memory statistics, one monitor process per node.
srun --ntasks-per-node=1 --label bash -c '
  echo "Node $(hostname) monitor started: $(date)"
  nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv \
    -l 5 > "${1}/gpu_stats_$(hostname).csv" &
  GPUSTAT_PID=$!

  nvidia-smi dmon -s um -d 5 \
    > "${1}/gpu_dmon_$(hostname).log" &
  DMON_PID=$!

  vmstat 5 \
    > "${1}/vmstat_$(hostname).log" &
  VMSTAT_PID=$!

  if command -v mpstat &> /dev/null; then
    mpstat -P ALL 5 \
      > "${1}/cpu_$(hostname).log" &
    MPSTAT_PID=$!
  fi

  wait
' _ "$LOGDIR" &
MONITOR_PID=$!

###########################################################
# Training
###########################################################

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --node-rank=$SLURM_NODEID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    train.py configs/train_cluster_foa_ddp_multinode.yaml

TRAIN_EXIT_CODE=$?

###########################################################
# Cleanup
###########################################################

kill "$MONITOR_PID" 2>/dev/null
wait "$MONITOR_PID" 2>/dev/null

echo "Finished: $(date)"
exit $TRAIN_EXIT_CODE