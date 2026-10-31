# Piped over one ssh to the Curiosity login node: inputs from the transfer repo, the OmniLottie
# checkout at a fixed commit, its own environment, weights at fixed revisions, an import check,
# then the job. The token is read from ~/.config/mojidiff/hf_token.
set -euo pipefail
T=$HOME/eai-team11/mojidiff
B=$T/omniprobe
UV=/cm/shared/apps/py-uv/0.11.31/bin/uv
HF_TOKEN=$(cat "$HOME/.config/mojidiff/hf_token")
export HF_TOKEN TMPDIR=$T/tmp UV_CACHE_DIR=$T/tmp/uv HF_HOME=$T/hf HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "$B" "$T/logs"
HF="$UV tool run --from huggingface_hub>=1.28 hf"
rm -rf "$B/in-dl" && $HF download chrispie/mojidiff-transfer --repo-type dataset --include "probe/omnilottie/*" \
    --local-dir "$B/in-dl" --quiet
rm -rf "$B/in" && mv "$B/in-dl/probe/omnilottie" "$B/in"
(cd "$B/in" && sha256sum -c --quiet SHA256SUMS)
[ -d "$B/repo" ] || git clone -q https://github.com/OpenVGLab/OmniLottie "$B/repo"
git -C "$B/repo" checkout -q 26131278e7b46bc2f64f989b25bb816f9e07c528
$HF download OmniLottie/OmniLottie --revision 10b0d6bb443a2c70eb6a6fae6400f571d9df29d6 \
    --include "*.safetensors" --include "*.json" --local-dir "$B/weights" --quiet
$HF download Qwen/Qwen2.5-VL-3B-Instruct --revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
    --include "*.json" --include "*.txt" --local-dir "$B/qwen" --quiet
if [ ! -x "$B/.venv/bin/python" ]; then
    $UV venv -q --python 3.12 "$B/.venv"
    $UV pip install -q --python "$B/.venv/bin/python" torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
    $UV pip install -q --python "$B/.venv/bin/python" transformers==4.51.3 qwen-vl-utils==0.0.11 safetensors \
        pandas "numpy<2.3" "Pillow<12" "huggingface_hub<1.0"
fi
"$B/.venv/bin/python" -c "import torchvision" 2>/dev/null || $UV pip install -q --python "$B/.venv/bin/python" \
    torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu130
(cd /tmp && "$B/.venv/bin/python" -c "
import sys, types, torch, transformers, qwen_vl_utils
from transformers import Qwen2_5_VLForConditionalGeneration
sys.modules['decord'] = types.SimpleNamespace(VideoReader=None, cpu=None)
sys.modules['datasets'] = types.SimpleNamespace(load_dataset=None, load_from_disk=None)
sys.path.insert(0, '$B/repo')
import torch, transformers, inference
from transformers import AutoProcessor
AutoProcessor.from_pretrained('$B/qwen')
print('torch', torch.__version__, 'transformers', transformers.__version__, 'import ok')")
du -sh "$B/weights"
module load slurm
sbatch --parsable "$B/in/omni.slurm"
