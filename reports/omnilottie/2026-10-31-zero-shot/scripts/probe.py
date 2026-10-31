"""OmniLottie zero-shot probe: one process per GPU, model loaded once, every item in two modes.

    python probe.py <omnilottie repo> <weights dir> <qwen processor dir> <items.jsonl> <images dir>
        <out dir> <shard> <n shards>

Modes: "image" (render plus prompt, OmniLottie's Text-Image-to-Lottie) and "text" (prompt only).
Sampling settings are the repository's defaults (temperature 0.9, top-p 0.25, top-k 5,
repetition penalty 1.01, at most 4096 new tokens). Latency is wall time per item on one GPU,
including the vision encoder and token-to-JSON conversion.
"""

import json
import random
import sys
import time
import types
from pathlib import Path

repo, weights, qwen, items_path, images, out, shard, nshards = sys.argv[1:9]
shard, nshards = int(shard), int(nshards)
import qwen_vl_utils  # noqa: E402, F401
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402

# inference.py imports decord (video) and datasets (benchmark) at module level; neither path runs
# here. The stubs go in only after transformers and qwen_vl_utils have probed for real packages.
sys.modules["decord"] = types.SimpleNamespace(VideoReader=None, cpu=None)
sys.modules["datasets"] = types.SimpleNamespace(load_dataset=None, load_from_disk=None)
sys.path.insert(0, repo)
import inference as inf  # noqa: E402

device = torch.device("cuda:0")
out_dir = Path(out)
out_dir.mkdir(parents=True, exist_ok=True)
t0 = time.time()
processor = AutoProcessor.from_pretrained(qwen, padding_side="left")
processor.tokenizer.padding_side = "left"

config = AutoConfig.from_pretrained(qwen, vocab_size=192400, bos_token_id=192398,
                                    eos_token_id=192399, pad_token_id=151643)


class Wrapper(torch.nn.Module):
    """Same parameter names as decoder.LottieDecoder, without loading the base weights first."""

    def __init__(self) -> None:
        super().__init__()
        self.transformer = Qwen2_5_VLForConditionalGeneration._from_config(
            config, torch_dtype=torch.bfloat16, attn_implementation="sdpa")


model = Wrapper()
index = json.loads((Path(weights) / "model.safetensors.index.json").read_text())
state = {}
for shard_file in sorted(set(index["weight_map"].values())):
    state.update(load_file(str(Path(weights) / shard_file)))
missing, unexpected = model.load_state_dict(state, strict=False)
# Tied lm_head may be absent from the checkpoint; anything else missing is an error.
assert not unexpected and all("lm_head" in k for k in missing), (missing, unexpected)
if missing:
    model.transformer.tie_weights()
model = model.to(device).eval()
load_s = time.time() - t0
print(f"shard {shard}: loaded in {load_s:.1f}s, missing={missing}", flush=True)

cfg = {"tokenizer_name": qwen, "text_len": 1500, "pix_len": 4096, "num_candidates": 1}
items = [json.loads(line) for line in open(items_path)]
records = []
for k, item in enumerate(items):
    if k % nshards != shard:
        continue
    for mode in ("image", "text"):
        name = f"{item['hexcode']}-{mode}"
        random.seed(k)  # add_random_background picks a light background colour
        torch.manual_seed(k)
        torch.cuda.synchronize()
        start = time.time()
        try:
            lottie, info = inf.run_inference(
                model=model, processor=processor,
                task_type=inf.TASK_IMAGE if mode == "image" else inf.TASK_TEXT,
                device=device, cfg=cfg, uid=name,
                image_path=str(Path(images) / f"{item['hexcode']}.png") if mode == "image" else None,
                text_description=item["prompt"], use_sampling=True, temperature=0.9, top_p=0.25,
                top_k=5, repetition_penalty=1.01, output_path=str(out_dir / f"{name}.json"),
                verbose=False)
            error = None
        except Exception as exc:  # recorded, not fatal: one bad sample must not stop the shard
            lottie, info, error = None, {}, repr(exc)
        torch.cuda.synchronize()
        seconds = time.time() - start
        rec = {"name": name, "hexcode": item["hexcode"], "mode": mode, "seconds": round(seconds, 2),
               "generated_len": info.get("generated_len"), "has_eos": info.get("has_eos"),
               "is_valid": info.get("is_valid"), "issues": info.get("issues"),
               "ok": lottie is not None, "error": error,
               "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        records.append(rec)
        print(json.dumps(rec), flush=True)
(out_dir / f"records-{shard}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
print(f"shard {shard}: done, load {load_s:.1f}s", flush=True)
