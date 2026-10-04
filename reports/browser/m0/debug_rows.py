"""Which table rows did the browser's embedding Gathers return? (debug dump vs initializers)"""
import json, sys, numpy as np, onnx
from onnx import numpy_helper
v, tag = sys.argv[1], sys.argv[2]
m = onnx.load(f'onnx/decoder_step_{v}.onnx')
W = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
d = json.load(open(f'results/replay_{tag}.json'))['outputs']
for nd in m.graph.node:
    if nd.op_type == 'Gather' and nd.input[0] in W and W[nd.input[0]].ndim == 2 and nd.output[0] in d:
        a = np.array(d[nd.output[0]]['data'], np.float32).reshape(-1, W[nd.input[0]].shape[1])
        idx = d.get(nd.input[1], {}).get('data')
        rows = [int(np.argmin(np.abs(W[nd.input[0]] - x[None]).max(1))) for x in a]
        print(nd.input[0], 'index tensor', nd.input[1], idx, 'gathered rows', rows)
