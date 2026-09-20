FROM nvcr.io/nvidia/pytorch@sha256:025d9b102b5436d4af8af58f12c6a46b7e5d16f19543b1d2cc4446bf2650b4f1

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libcairo2=1.18.0-3build1 \
    && rm -rf /var/lib/apt/lists/*

# Exact rendering/normalization closure from uv.lock. The NVIDIA base supplies PyTorch
# and NumPy; project source is mounted read-only from the immutable staged snapshot.
RUN python3 -m pip install --no-cache-dir \
    absl-py==2.5.0 \
    cairocffi==1.7.1 \
    cairosvg==2.9.0 \
    cffi==2.1.1 \
    cssselect2==0.9.0 \
    defusedxml==0.7.1 \
    lxml==6.1.2 \
    picosvg==0.23.0 \
    pillow==12.3.0 \
    pycparser==3.0 \
    pyyaml==6.0.3 \
    skia-pathops==0.9.2 \
    svg-path==7.1 \
    tinycss2==1.5.1 \
    webencodings==0.6.1
