#!/bin/bash
#SBATCH --job-name=sac_reach
#SBATCH --output=outputs/slurm/%x_%j.log
#SBATCH --error=outputs/slurm/%x_%j.err
#SBATCH --partition=main-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# Submit từ project root, config truyền qua tham số (mặc định sac.yaml):
#   sbatch --job-name=sac_circle scripts/submit_train.sh configs/sac_circle.yaml
CONFIG="${1:-configs/sac.yaml}"
PROJECT_DIR=/home/s3002152/LeeHoang_/robot-arm-rl
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

# Mỗi session train có folder riêng trong outputs/runs/
RUN_DIR="outputs/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

echo "Job started on $(hostname)"
echo "Config: $CONFIG | Run dir: $RUN_DIR"
python -c 'import torch; print("torch:", torch.__version__, "| threads:", torch.get_num_threads())'

# Kỷ luật: test env phải pass trước khi train
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q || { echo "Tests FAILED — không train"; exit 1; }

python train.py --config "$CONFIG" --run-dir "$RUN_DIR"
status=$?

# Copy log SLURM vào run dir để session tự chứa đủ thông tin
cp "outputs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" "$RUN_DIR/" 2>/dev/null
echo "Job finished (exit $status)"
exit $status
