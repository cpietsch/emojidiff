#!/bin/bash
# Bring one commit up on the Curiosity cluster (login node): fetch the private transfer
# repo from Hugging Face, verify every file, check the commit out, link the caches, and
# build that checkout's environment and the CLIP weights.
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

# Downloads before any environment exists use the cluster's own uv-managed hf tool.
HF="$UV tool run --from huggingface_hub>=1.28 hf"
$HF download "$REPO" SHA256SUMS "code/mojidiff-$COMMIT.bundle" data/openmoji-17.0.0.tar.gz \
    data/twemoji.tar.gz cache/render2svg-corpus-5590a1948f5f.npz \
    cache/render2svg-corpus-35a6ce034e7e.npz cache/render2svg-extra-twemoji-23878da09a9c.npz \
    --repo-type dataset --local-dir "$T/transfer" --quiet
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

# One environment per checkout with the project installed: the isolated renderer starts
# `python -I`, which ignores PYTHONPATH, so mojidiff must be importable from site-packages.
E=$R/.venv
if [ ! -x "$E/bin/python" ]; then
    $UV venv -q --python 3.12 "$E"
    $UV pip install -q --python "$E/bin/python" torch==2.14.1 \
        --index-url https://download.pytorch.org/whl/cu130
    $UV pip install -q --python "$E/bin/python" -e "$R" transformers==5.17.0 \
        "huggingface_hub>=1.28" pytest
fi
"$E/bin/hf" download openai/clip-vit-base-patch32 --revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 \
    --quiet >/dev/null
(cd /tmp && env -i LC_ALL=C.UTF-8 "$E/bin/python" -I -c "import mojidiff.representation.render_worker")
"$E/bin/python" -c "import torch, cairosvg, transformers; print('torch', torch.__version__, \
'transformers', transformers.__version__)"
echo "ready: $R"
