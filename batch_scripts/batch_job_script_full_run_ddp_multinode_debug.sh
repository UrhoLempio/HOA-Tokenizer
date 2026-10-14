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

# Limit the diagnostic run while retaining the normal training behavior.
export DEBUG_MAX_STEPS=1000
export DEBUG_TIMING=0

echo "Running on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"

nvidia-smi

echo "NODE:$HOSTNAME RANK:$SLURM_PROCID LOCAL_RANK:$LOCAL_RANK WORLD_SIZE:$WORLD_SIZE"

# Create logging directory
LOGDIR="${SLURM_SUBMIT_DIR:-$PWD}/profiling_${SLURM_JOB_ID}"
mkdir -p "$LOGDIR"

###########################################################
# Monitoring (runs in background)
###########################################################

# Record one GPU-utility sample per second on every node. Separate Slurm output
# files make failures visible even if a node cannot access the shared log path.
srun --overlap \
    --ntasks="$SLURM_NNODES" \
    --ntasks-per-node=1 \
    --cpus-per-task=1 \
    --gres=gpu:gh200:4 \
    --label \
    --output="$LOGDIR/monitor_%N.out" \
    --error="$LOGDIR/monitor_%N.err" \
    bash -c '
      set -eu
      logdir="$1"
      host=$(hostname)
      gpu_file="$logdir/gpu_utilization_${host}.csv"
      ready_file="$logdir/ready_${host}"
      query="timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"

      printf "timestamp,index,uuid,gpu_util_pct,mem_util_pct,mem_used_mib,mem_total_mib,power_w\n" > "$gpu_file"
      if ! nvidia-smi --query-gpu="$query" --format=csv,noheader,nounits >> "$gpu_file"; then
        echo "ERROR: initial nvidia-smi query failed on $host" >&2
        exit 1
      fi

      touch "$ready_file"
      echo "MONITOR_READY host=$host time=$(date -Is) file=$gpu_file"
      exec nvidia-smi --query-gpu="$query" --format=csv,noheader,nounits --loop=1 >> "$gpu_file"
    ' _ "$LOGDIR" &
MONITOR_PID=$!

# Wait briefly for every node to prove that its first GPU sample was written.
mapfile -t NODE_HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
MONITORS_READY=0
for attempt in {1..30}; do
    READY_COUNT=0
    for node in "${NODE_HOSTS[@]}"; do
        if [[ -f "$LOGDIR/ready_${node}" ]]; then
            ((READY_COUNT += 1))
        fi
    done
    if (( READY_COUNT == SLURM_NNODES )); then
        MONITORS_READY=1
        break
    fi
    sleep 1
done

if (( MONITORS_READY )); then
    echo "GPU monitors ready on $READY_COUNT/$SLURM_NNODES nodes."
else
    echo "ERROR: GPU monitors ready on $READY_COUNT/$SLURM_NNODES nodes after 30 seconds. Check $LOGDIR/monitor_*.err"
    kill "$MONITOR_PID" 2>/dev/null
    wait "$MONITOR_PID" 2>/dev/null
    exit 1
fi

###########################################################
# Training
###########################################################

srun --overlap torchrun \
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