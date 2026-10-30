#!/bin/bash
#SBATCH --job-name=mojidiff-arms
#SBATCH --partition=hackathon
#SBATCH --account=eai-hackathon
#SBATCH --qos=eai-team11
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --mem=800G
#SBATCH --time=06:00:00
#SBATCH --output=/home/eai-jdkxz/eai-team11/mojidiff/logs/%x-%j.out
# One render-to-SVG run per GPU from one checkout made by setup.sh; when all have ended,
# the run records, logs and checkpoints go back through the private transfer repo under
# results/<job id>/, finished by a DONE file the workstation waits for.
#
#   sbatch --gres=gpu:N --export=ALL,COMMIT=<short>,ARMS="configs/a.yaml configs/b.yaml" \
#       scripts/curiosity/arms.slurm
set -uo pipefail
T=$HOME/eai-team11/mojidiff
R=$T/repo-$COMMIT
OUT=$T/results/$SLURM_JOB_ID
mkdir -p "$OUT"
HF_TOKEN=$(cat "$HOME/.config/mojidiff/hf_token")
export HF_TOKEN PYTHONUNBUFFERED=1 PYTHONPATH=$R/src MOJIDIFF_CACHE_ROOT=$T/cache HF_HOME=$T/hf \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
cd "$R" || exit 1
registry_lines=$(wc -l < state/runs.jsonl)
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

i=0
pids=()
names=()
for config in $ARMS; do
    name=$(basename "$config" .yaml)
    CUDA_VISIBLE_DEVICES=$i TRITON_CACHE_DIR=/tmp/tri-$SLURM_JOB_ID-$i \
        TORCHINDUCTOR_CACHE_DIR=/tmp/ind-$SLURM_JOB_ID-$i XDG_CACHE_HOME=/tmp/xdg-$SLURM_JOB_ID-$i \
        "$T/env/bin/python" -m mojidiff.learning.render2svg --config "$config" \
        > "$OUT/$name.log" 2>&1 &
    pids+=($!)
    names+=("$name")
    i=$((i + 1))
done
for k in "${!pids[@]}"; do
    wait "${pids[$k]}"
    echo "${names[$k]} exit $?" | tee -a "$OUT/exits.txt"
done

# Each new registry line names a run this job started; copy its record, log and checkpoints.
HF_HUB_OFFLINE=0
export HF_HUB_OFFLINE
tail -n +"$((registry_lines + 1))" state/runs.jsonl > "$OUT/runs.jsonl"
for run_id in $("$T/env/bin/python" -c "import json, sys
seen = []
for line in open(sys.argv[1]):
    run_id = json.loads(line)['run_id']
    if run_id not in seen:
        seen.append(run_id)
print(' '.join(seen))" "$OUT/runs.jsonl"); do
    mkdir -p "$OUT/runs/$run_id" "$OUT/checkpoints/$run_id"
    cp -r "runs/$run_id/." "$OUT/runs/$run_id/" 2>/dev/null
    slug=${run_id%-*-*-*}
    for log in "$OUT"/*.log; do
        grep -q "$run_id" "$log" && cp "$log" "$OUT/runs/$run_id/stdout.log"
    done
    for ckpt in best.pt latest.pt; do
        [ -f "$T/cache/runs/$run_id/$ckpt" ] && ln -f "$T/cache/runs/$run_id/$ckpt" "$OUT/checkpoints/$run_id/"
    done
    echo "$slug $run_id" >> "$OUT/index.txt"
done
date -u +%FT%TZ > "$OUT/DONE"
for attempt in 1 2 3; do
    "$T/env/bin/hf" upload chrispie/mojidiff-transfer "$OUT" "results/$SLURM_JOB_ID" \
        --repo-type dataset --commit-message "results of job $SLURM_JOB_ID ($COMMIT)" --quiet && break
    echo "upload attempt $attempt failed"
    sleep 60
done
