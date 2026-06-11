#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is missing. Accept the SORRY-Bench dataset terms, then export HF_TOKEN before running." >&2
  exit 1
fi

mkdir -p artifacts
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_path="artifacts/runpod_${run_stamp}.log"
ln -sf "$(basename "$log_path")" artifacts/runpod_latest.log

run_step() {
  echo "" | tee -a "$log_path"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$log_path"

  set +e
  "$@" 2>&1 | tee -a "$log_path"
  status="${PIPESTATUS[0]}"
  set -e

  if [[ "$status" -ne 0 ]]; then
    echo "step failed with exit code $status: $*" | tee -a "$log_path" >&2
    exit "$status"
  fi
}

save_pip_freeze() {
  freeze_path="artifacts/${1}_${run_stamp}.txt"
  echo "" | tee -a "$log_path"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] python -m pip freeze > $freeze_path" | tee -a "$log_path"

  set +e
  python -m pip freeze | tee "$freeze_path" | tee -a "$log_path"
  status="${PIPESTATUS[0]}"
  set -e

  if [[ "$status" -ne 0 ]]; then
    echo "failed to record pip freeze: $freeze_path" | tee -a "$log_path" >&2
    exit "$status"
  fi
}

if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi is missing; this RunPod experiment must run on a CUDA GPU pod." | tee -a "$log_path" >&2
  exit 1
fi
run_step nvidia-smi

python_bin="${PYTHON_BIN:-python3}"
venv_dir="${VENV_DIR:-.venv}"
run_step "$python_bin" --version
run_step "$python_bin" -m venv "$venv_dir"
# shellcheck source=/dev/null
source "$venv_dir/bin/activate"
run_step python --version
run_step python -m pip install --upgrade pip
run_step python -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0+cu128"
run_step python -m pip install -r requirements.txt
save_pip_freeze runpod_requirements_base
run_step python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else "torch cannot see CUDA")'
run_step python -m src.download_assets
run_step python -m src.fit_direction
run_step python -m src.run_gemma_sorry_bench
run_step python -m src.download_official_judge
run_step python -m pip install --use-pep517 -r requirements-official-evaluator.txt
save_pip_freeze runpod_requirements_official
run_step python -m src.score_sorry_bench_official

echo "" | tee -a "$log_path"
echo "finished RunPod experiment: $log_path" | tee -a "$log_path"
