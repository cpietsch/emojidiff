#!/bin/bash
#SBATCH --job-name=mojidiff-omnilottie-probe
#SBATCH --partition=hackathon
#SBATCH --account=eai-hackathon
#SBATCH --qos=eai-team11
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=800G
#SBATCH --time=01:00:00
#SBATCH --output=/home/eai-jdkxz/eai-team11/mojidiff/logs/%x-%j.out
# OmniLottie zero-shot probe: 8 shards, one per GPU; results to results/<job id>/ with DONE.
set -uo pipefail
B=$HOME/eai-team11/mojidiff/omniprobe
OUT=$HOME/eai-team11/mojidiff/results/$SLURM_JOB_ID
mkdir -p "$OUT"
export HF_HOME=$HOME/eai-team11/mojidiff/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 \
       TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 HF_HUB_DISABLE_PROGRESS_BARS=1
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee "$OUT/gpus.txt"
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
pids=()
for s in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=${GPUS[$s]} "$B/.venv/bin/python" "$B/in/probe.py" "$B/repo" "$B/weights" \
        "$B/qwen" "$B/in/items.jsonl" "$B/in/images" "$OUT/lottie" "$s" 8 > "$OUT/shard-$s.log" 2>&1 &
    pids+=($!)
done
for s in "${!pids[@]}"; do wait "${pids[$s]}"; echo "shard $s exit $?" | tee -a "$OUT/exits.txt"; done
date -u +%FT%TZ > "$OUT/DONE"
for attempt in 1 2 3; do
    HF_HUB_OFFLINE=0 HF_TOKEN=$(cat "$HOME/.config/mojidiff/hf_token") "$B/.venv/bin/python" -c "
import sys
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id='chrispie/mojidiff-transfer', repo_type='dataset', folder_path=sys.argv[1],
    path_in_repo='results/' + sys.argv[2], commit_message='OmniLottie probe ' + sys.argv[2])" "$OUT" "$SLURM_JOB_ID" && break
    sleep 60
done
