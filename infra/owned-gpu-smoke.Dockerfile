FROM nvcr.io/nvidia/pytorch@sha256:025d9b102b5436d4af8af58f12c6a46b7e5d16f19543b1d2cc4446bf2650b4f1

# Exact CairoSVG closure from uv.lock. The NVIDIA base supplies PyTorch and NumPy;
# project source is mounted read-only from the immutable staged snapshot at runtime.
RUN python3 -m pip install --no-cache-dir \
    cairocffi==1.7.1 \
    cairosvg==2.9.0 \
    cffi==2.1.1 \
    cssselect2==0.9.0 \
    defusedxml==0.7.1 \
    pillow==12.3.0 \
    pycparser==3.0 \
    tinycss2==1.5.1 \
    webencodings==0.6.1
