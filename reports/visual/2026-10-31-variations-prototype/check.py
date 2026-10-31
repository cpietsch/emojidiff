"""Read-only feasibility check: flow-SDEdit variations and mixes on lfp-v2x (writes only to scratchpad)."""
import sys, time, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/home/dev/workspace/mojidiff/src")
from mojidiff.vectorise.server import PILOT_CONFIG
from mojidiff.learning.render2svg import load_corpus, _programs_to_renders, pixel_error, contact_sheet
from mojidiff.learning.openmoji_pilot import _load_program, load_pilot_index
from mojidiff.learning.latent_flow import load_flow_checkpoint, load_flow_parent
from mojidiff.learning.pixel_latent import ink_coverage
from mojidiff.learning.precision import strict_float32
from mojidiff.gallery.latent_backends import CanvasFlowBackend

OUT = Path(sys.argv[1])
dev = torch.device("cuda")
plain, _, layout, pilot, dhash = load_corpus(PILOT_CONFIG, 144)
by_split, _, _ = load_pilot_index(pilot)
template = _load_program(by_split["primary/train"][0], pilot, layout.codec)
state = torch.load("/home/dev/.cache/mojidiff/runs/lfp-v2x-exploratory-a8cbfdc-d2e9869c-47646604/best.pt", map_location="cpu")
vt = load_flow_parent(state, layout).to(dev)
prior = load_flow_checkpoint(state).to(dev).eval()
be = CanvasFlowBackend(vt, prior, graphs=False)
val = plain["primary/validation"]
idx = list(range(5, len(val.hexcodes), 34))[:10]
images = val.images[idx].to(dev)
with strict_float32(), torch.no_grad():
    mu, logvar = vt.posterior(images)
    x0 = prior.standardise(mu)
    sig_std = (prior.standardise(mu + (0.5*logvar).exp()) - x0)  # posterior sigma in standardised units
print("posterior sigma (standardised units) median", float(sig_std.median()), "mean", float(sig_std.mean()))
print("x0 per-icon norm / sqrt(D)", (x0.flatten(1).norm(dim=1)/x0[0].numel()**0.5).tolist())

def sdedit(x0, s, gen):
    eps = torch.randn(x0.shape, generator=gen).to(dev)
    if s <= 0: return prior.unstandardise(x0)
    x = (1 - s) * x0 + s * eps
    n = max(1, round(50 * s))
    times = torch.linspace(s, 0.0, n + 1, dtype=torch.float64).tolist()
    for a, b in zip(times[:-1], times[1:]):
        v = prior.velocity(x, torch.full((len(x),), a, device=dev))
        x = x + (b - a) * v
    return prior.unstandardise(x)

strengths = [0.0, 0.3, 0.45, 0.6, 0.8]
draws = 4
gen = torch.Generator().manual_seed(123)
lat, meta = [], []
with strict_float32(), torch.no_grad():
    for s in strengths:
        for d in range(1 if s == 0 else draws):
            z = sdedit(x0, s, gen)
            for i in range(len(idx)):
                lat.append(z[i]); meta.append(("vary", i, s, d))
    # mixes: pairs (i, i+1), global weight 0.5, strengths 0 / 0.45 / 0.6
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    a = torch.stack([x0[p] for p, _ in pairs]); b = torch.stack([x0[q] for _, q in pairs])
    for s in [0.0, 0.45, 0.6]:
        for d in range(1 if s == 0 else 2):
            z = sdedit(0.5 * a + 0.5 * b, s, gen)
            for k in range(len(pairs)):
                lat.append(z[k]); meta.append(("mix", k, s, d))
    Z = torch.stack(lat)
    t0 = time.perf_counter()
    toks = be.decode(Z)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
print(f"decoded {len(Z)} latents in {dt:.1f}s ({1000*dt/len(Z):.0f} ms/latent batched)")
renders = _programs_to_renders(toks.cpu(), layout, template)
targets = val.targets[idx].numpy()
rows = {}
stats = {}
for (kind, i, s, d), r in zip(meta, renders):
    key = (kind, s)
    st = stats.setdefault(f"{kind} s={s}", {"frag": [], "err_src": []})
    st["frag"].append(ink_coverage(r) < 0.10)
    if kind == "vary": st["err_src"].append(pixel_error(r, targets[i]))
    rows.setdefault((kind, i), []).append(r)
for k, v in stats.items():
    print(k, "fragments", np.mean(v["frag"]).round(3), "err_to_source", (np.mean(v["err_src"]).round(4) if v["err_src"] else "-"))
sheet_rows = [[targets[i]] + rows[("vary", i)] for i in range(len(idx))]
(OUT / "vary.png").write_bytes(contact_sheet(sheet_rows))
mix_rows = [[targets[p], targets[q]] + rows[("mix", k)] for k, (p, q) in enumerate(pairs)]
(OUT / "mix.png").write_bytes(contact_sheet(mix_rows))
print("icons", [val.hexcodes[i] for i in idx])
