#!/usr/bin/env bash
# Milestone 0 browser matrix. Each config: fresh headless Chromium, cold icon, then the icons.
set -u
cd "$(dirname "$0")"
source ../env.sh
N=ort.webgpu.min.mjs
cfgs=()
for v in A B0 B C; do
  cfgs+=("${v}_jsep_b1=v=$v&ep=webgpu&b=1" "${v}_native_b1=v=$v&ep=webgpu&b=1&bundle=$N"
         "${v}_jsep_b8=v=$v&ep=webgpu&b=8" "${v}_native_b8=v=$v&ep=webgpu&b=8&bundle=$N")
done
cfgs+=("Bs_jsep_b1=v=Bs&ep=webgpu&b=1" "Bs_native_b1=v=Bs&ep=webgpu&b=1&bundle=$N")
for v in A B0 B C; do
  cfgs+=("${v}_wasm1_b1=v=$v&ep=wasm&threads=1&b=1" "${v}_wasm4_b1=v=$v&ep=wasm&threads=4&b=1")
done
for v in A B0 B C; do
  cfgs+=("${v}_wasm1_b8=v=$v&ep=wasm&threads=1&b=8&icons=8" "${v}_wasm4_b8=v=$v&ep=wasm&threads=4&b=8&icons=8")
done
for c in "${cfgs[@]}"; do
  tag=${c%%=*}
  if [ -f "results/replay_${tag}.json" ] && ! grep -q '"error"' "results/replay_${tag}.json"; then echo "skip $tag"; continue; fi
  timeout 7200 node run_replay.mjs "$c" 2>&1 | grep -v '^error: .*VerifyEach\|Rerunning' | cut -c1-900
done
# Pages-like server (no COOP/COEP): ask for 4 threads, see what ORT gets
PORT=8942 timeout 7200 node run_replay.mjs "B_wasm4_b1_nocoi=v=B&ep=wasm&threads=4&b=1&icons=8" 2>&1 | grep -v '^error: .*VerifyEach\|Rerunning' | cut -c1-900
echo MATRIX DONE
