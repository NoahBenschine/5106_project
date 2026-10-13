#!/bin/bash

# Exit on error
set -e

# Determine the directory where this script resides (baselines/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYMARL_DIR="${SCRIPT_DIR}/pymarlzooplus/pymarlzooplus"

# Configuration variables
LAYOUT="cramped_room"         # Options: cramped_room, asymmetric_advantages, coordination_ring, forced_coordination, counter_circuit
REWARD_TYPE="sparse"          # Options: sparse, shaped
TIME_LIMIT=400
T_MAX=2000000                 # 2M timesteps is standard for Overcooked convergence
BATCH_SIZE_RUN=8              # Vectorized parallel workers
RUNNER="parallel"

echo "============================================================"
echo "Starting MAPPO training on Overcooked (${LAYOUT})"
echo "Working directory: ${PYMARL_DIR}"
echo "============================================================"

# Ensure the submodule directory exists
if [ ! -d "${PYMARL_DIR}" ]; then
    echo "Error: Submodule directory '${PYMARL_DIR}' not found."
    echo "Did you run 'git submodule update --init --recursive'?"
    exit 1
fi

# Change working directory to pymarlzooplus so Sacred loads local YAML configs correctly
cd "${PYMARL_DIR}"

# Execute MAPPO experiment
python main.py \
    --config=mappo \
    --env-config=overcooked \
    with \
    env_args.key="${LAYOUT}" \
    env_args.time_limit=${TIME_LIMIT} \
    env_args.reward_type="${REWARD_TYPE}" \
    runner="${RUNNER}" \
    batch_size_run=${BATCH_SIZE_RUN} \
    t_max=${T_MAX} \
    save_model=True \
    save_model_interval=500000 \
    log_interval=10000

echo "Training complete!"
