// The exported model (python -m mojidiff.learning.onnx_export) as decode.mjs's abstract
// encode/step pair, on ONNX Runtime. The same code runs onnxruntime-web in a page or worker
// ("wasm" or "webgpu") and onnxruntime-node ("cpu"); the caller passes the ORT module, so
// this file has no dependency of its own.
//
//   encoder.onnx       pixels (1, S, S, 3) float32 RGB 0-255, rows (R,) int64
//                      -> memory_keys, memory_values (layers, R, heads, cells, head_dim)
//   decoder_step.onnx  ids (6, 1, L) float32, target (1,) float32,
//                      past_keys/values (layers, 1, heads, P, head_dim), memory K/V
//                      -> logits (1, vocabulary) of the last step, present_keys/values (P + L)
//
// On WebGPU the cache and the memory stay on the GPU ("gpu-buffer") and only the logits are
// read back; a cache tensor is disposed as soon as the call that consumed it has run, as in
// milestone 0's replay. Shapes come from manifest.json's constants.

/** Session options for both graphs on one backend ("cpu", "wasm" or "webgpu"). */
export function sessionOptions(backend, extra = {}) {
  const gpu = backend === "webgpu";
  return {
    encoder: {
      executionProviders: [backend], ...extra,
      ...(gpu ? { preferredOutputLocation: { memory_keys: "gpu-buffer", memory_values: "gpu-buffer" } } : {}),
    },
    decoder: {
      executionProviders: [backend], ...extra,
      ...(gpu ? { preferredOutputLocation: { logits: "cpu", present_keys: "gpu-buffer", present_values: "gpu-buffer" } } : {}),
    },
  };
}

/**
 * Bind two ONNX Runtime sessions to decode.mjs's contract. `constants` is manifest.json's
 * "constants". Returns {encode, step, release}: encode(Uint8Array S*S*3 RGB) -> memory;
 * step(ids, target, past, memory) -> {logits, present}; release(past, memory) frees the
 * last cache and the memory (GPU buffers on WebGPU; a no-op elsewhere).
 */
export function ortModel(ort, { encoder, decoder, constants }) {
  const layers = constants.decoder_layers, heads = constants.heads, headDim = constants.head_dim;
  const size = constants.image_size, vocabulary = constants.vocabulary;
  const empty = () => new ort.Tensor("float32", new Float32Array(0), [layers, 1, heads, 0, headDim]);
  const rows = () => new ort.Tensor("int64", new BigInt64Array(1), [1]);
  const free = tensor => { if (tensor && tensor.location === "gpu-buffer") tensor.dispose(); };

  async function encode(image) {
    if (image.length !== size * size * 3) throw new Error(`expected a ${size}x${size} RGB render`);
    const pixels = new Float32Array(image.length);
    for (let i = 0; i < image.length; i++) pixels[i] = image[i];
    const out = await encoder.run({ pixels: new ort.Tensor("float32", pixels, [1, size, size, 3]), rows: rows() });
    return { keys: out.memory_keys, values: out.memory_values };
  }

  async function step(ids, target, past, memory) {
    const out = await decoder.run({
      ids: new ort.Tensor("float32", ids, [6, 1, ids.length / 6]),
      target: new ort.Tensor("float32", target, [1]),
      past_keys: past ? past.keys : empty(),
      past_values: past ? past.values : empty(),
      memory_keys: memory.keys,
      memory_values: memory.values,
    });
    if (past) { free(past.keys); free(past.values); }
    const logits = out.logits.data;
    if (logits.length !== vocabulary) throw new Error(`logits of length ${logits.length}, expected ${vocabulary}`);
    return { logits, present: { keys: out.present_keys, values: out.present_values } };
  }

  function release(past, memory) {
    if (past) { free(past.keys); free(past.values); }
    if (memory) { free(memory.keys); free(memory.values); }
  }

  return { encode, step, release };
}
