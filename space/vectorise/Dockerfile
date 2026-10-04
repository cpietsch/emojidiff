# MojiDiff edit-pixels, re-vectorise demo as a Hugging Face Docker Space (free CPU:
# 2 vCPU, 16 GB RAM).
#
# This directory is assembled by `scripts/build_space.py vectorise <outdir>` from the
# MojiDiff repository (https://github.com/cpietsch/emojidiff); nothing is cloned from it.
# The build fetches the pinned OpenMoji 17.0.0 sources and the public checkpoints, then
# builds the icon cache the demo reads, so the container only loads at startup.

# ---------------------------------------------------------------- OpenMoji 17.0.0
# The immutable upstream checkout the corpus ledgers pin (`source_revision` in
# configs/learning/openmoji-g1-geometric-gate-v16.yaml): colour SVGs, data files and the
# licence only. Every source is sha256-checked against the tracked ledger when read.
FROM python:3.12-slim AS openmoji
ARG OPENMOJI_TAG=17.0.0
ARG OPENMOJI_REVISION=f9fc506a3f913be9897ab0181d611d4c910a4104
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN git clone --quiet --depth 1 --branch "$OPENMOJI_TAG" --filter=blob:none --no-checkout \
        https://github.com/hfg-gmuend/openmoji.git /openmoji \
 && cd /openmoji \
 && test "$(git rev-parse HEAD)" = "$OPENMOJI_REVISION" \
 && git sparse-checkout set --no-cone /color/svg/ /data/ /LICENSE.txt \
 && git checkout --quiet \
 && rm -rf .git

# ---------------------------------------------------------------- the Space
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /bin/uv
# libcairo2: CairoSVG renders every icon the model reads and every best-of-8 candidate.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libcairo2 \
 && rm -rf /var/lib/apt/lists/*
# Spaces run the container as uid 1000. Its home is /home/dev so the checkpoint sits
# where the training machine keeps it, /home/dev/.cache/mojidiff (render2svg.CACHE_ROOT).
RUN useradd --create-home --uid 1000 --home-dir /home/dev dev
USER dev
ENV HOME=/home/dev \
    PATH=/home/dev/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2
WORKDIR /home/dev/app

# Locked dependencies; torch is 2.8.0+cpu from the PyTorch CPU index (extra model-cpu).
COPY --chown=dev:dev pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra model-cpu

# v9's best.pt from the public model repo, sha256-checked.
COPY --chown=dev:dev space/fetch_checkpoints.py space/checkpoints.json space/
RUN uv venv -q /tmp/hf \
 && uv pip install -q -p /tmp/hf huggingface_hub==2.1.1 \
 && /tmp/hf/bin/python space/fetch_checkpoints.py space/checkpoints.json \
 && rm -rf /tmp/hf

COPY --chown=dev:dev --from=openmoji /openmoji data/raw/openmoji/17.0.0
COPY --chown=dev:dev . .
# The project itself, editable as on the training machine: the isolated render worker
# (`python -I -m mojidiff.representation.render_worker`) ignores PYTHONPATH.
RUN uv sync --frozen --extra model-cpu
# The canonical icon SVGs and their index (data/processed/kitbash), and a check that the
# checkpoint loads with this torch.
RUN python -m mojidiff.vectorise.icon_cache \
 && python space/warm_cache.py vectorise

EXPOSE 7860
# v9 colour, the gallery's `v9`: scripts/build_space.py checks this run id against
# configs/gallery/models.yaml.
CMD ["python", "scripts/serve_vectorise.py", "--host", "0.0.0.0", "--port", "7860", \
     "--checkpoint", "/home/dev/.cache/mojidiff/runs/r2s-full-v9-colour-621bc8e-b26ad95e-47646604/best.pt"]
