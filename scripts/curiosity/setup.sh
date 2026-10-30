#!/bin/bash
# Bring one commit up on the Curiosity cluster (login node): fetch the private transfer
# repo from Hugging Face, verify every file, check the commit out, link the caches, and
# build the environment and the CLIP weights once.
#
#   bash -l scripts/curiosity/setup.sh <short commit>
#
# The token is read from ~/.config/mojidiff/hf_token (mode 600), never from arguments.
set -euo pipefail
COMMIT=${1:?usage: setup.sh <short commit>}
T=$HOME/eai-team11/mojidiff
REPO=chrispie/mojidiff-transfer
UV=/cm/shared/apps/py-uv/0.11.31/bin/uv
HF_TOKEN=$(cat "$HOME/.config/mojidiff/hf_token")
export HF_TOKEN TMPDIR=$T/tmp UV_CACHE_DIR=$T/tmp/uv HF_HOME=$T/hf HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "$T"/{tmp,transfer,cache,logs,hf,raw}

if [ ! -x "$T/env/bin/python" ]; then
    $UV venv -q --python 3.12 "$T/env"
    $UV pip install -q --python "$T/env/bin/python" torch==2.14.1 \
        --index-url https://download.pytorch.org/whl/cu130
    $UV pip install -q --python "$T/env/bin/python" "CairoSVG>=2.8.2" "Pillow>=11.3" \
        "PyYAML>=6.0.2" "defusedxml>=0.7.1" "numpy>=2.3" "picosvg>=0.23.0" "pyarrow>=21.0" \
        "svg.path>=7.0" transformers==5.17.0 "huggingface_hub>=1.28" pytest
fi
HF=$T/env/bin/hf

$HF download "$REPO" --repo-type dataset --local-dir "$T/transfer" --quiet \
    --include SHA256SUMS "code/mojidiff-$COMMIT.bundle" "data/*" "cache/*"
(cd "$T/transfer" && grep -E "code/mojidiff-$COMMIT.bundle|data/|cache/" SHA256SUMS \
    | sha256sum -c --quiet)

[ -d "$T/raw/openmoji/17.0.0" ] || tar -C "$T/raw" -xzf "$T/transfer/data/openmoji-17.0.0.tar.gz"
[ -d "$T/raw/external/twemoji" ] || tar -C "$T/raw" -xzf "$T/transfer/data/twemoji.tar.gz"
git config --global --get-all safe.directory | grep -qx "$T/raw/external/twemoji" \
    || git config --global --add safe.directory "$T/raw/external/twemoji"
for f in "$T"/transfer/cache/*.npz; do ln -sf "$f" "$T/cache/$(basename "$f")"; done

R=$T/repo-$COMMIT
if [ ! -d "$R" ]; then
    git clone -q "$T/transfer/code/mojidiff-$COMMIT.bundle" "$R"
    git -C "$R" checkout -q "$COMMIT"
    ln -s "$T/raw" "$R/data/raw"
fi
test "$(git -C "$R" rev-parse --short=7 HEAD)" = "$COMMIT"

$HF download openai/clip-vit-base-patch32 --revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 \
    --quiet >/dev/null
"$T/env/bin/python" -c "import torch, cairosvg, transformers; print('torch', torch.__version__, \
'transformers', transformers.__version__)"
echo "ready: $R"
