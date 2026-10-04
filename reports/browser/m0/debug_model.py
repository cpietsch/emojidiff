"""Expose every float or int intermediate of a decoder-step variant as a graph output (for bisecting
WebGPU-vs-CPU differences). Usage: python debug_model.py B -> onnx/decoder_step_B_dbg.onnx"""
import sys
from pathlib import Path
import onnx
from onnx import shape_inference
HERE = Path(__file__).resolve().parent
v = sys.argv[1]
m = onnx.load(str(HERE / f"onnx/decoder_step_{v}.onnx"))
m = shape_inference.infer_shapes(m)
info = {vi.name: vi for vi in m.graph.value_info}
have = {o.name for o in m.graph.output}
order = []
for node in m.graph.node:
    for o in node.output:
        if o in info and o not in have and info[o].type.tensor_type.elem_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.INT32, onnx.TensorProto.INT64):
            m.graph.output.append(info[o]); order.append(o)
onnx.save(m, str(HERE / f"onnx/decoder_step_{v}_dbg.onnx"))
(HERE / f"onnx/decoder_step_{v}_dbg.order.txt").write_text("\n".join(order))
print(len(order), "extra outputs")
