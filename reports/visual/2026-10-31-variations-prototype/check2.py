"""Read-only check 2: spatial (part) mixing + SDEdit, and end-to-end latency of 8 variations on the 4080."""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/home/dev/workspace/mojidiff/src")
from mojidiff.vectorise.server import PILOT_CONFIG
from mojidiff.learning.render2svg import load_corpus, _programs_to_renders, contact_sheet, greedy_decode
from mojidiff.learning.openmoji_pilot import _load_program, load_pilot_index
from mojidiff.learning.latent_flow import load_flow_checkpoint, load_flow_parent
from mojidiff.learning.pixel_latent import ink_coverage
from mojidiff.learning.precision import strict_float32
from mojidiff.learning.fast_decode import GraphDecoder

OUT = Path(sys.argv[1]); dev = torch.device("cuda")
plain, _, layout, pilot, _ = load_corpus(PILOT_CONFIG, 144)
by_split, _, _ = load_pilot_index(pilot)
template = _load_program(by_split["primary/train"][0], pilot, layout.codec)
state = torch.load("/home/dev/.cache/mojidiff/runs/lfp-v2x-exploratory-a8cbfdc-d2e9869c-47646604/best.pt", map_location="cpu")
vt = load_flow_parent(state, layout).to(dev).eval()
prior = load_flow_checkpoint(state).to(dev).eval()
val = plain["primary/validation"]
hx = val.hexcodes
def find(prefix):
    return next(i for i, h in enumerate(hx) if h.startswith(prefix))
def sdedit(x0, s, gen):
    eps = torch.randn(x0.shape, generator=gen).to(dev)
    if s <= 0: return prior.unstandardise(x0)
    x = (1 - s) * x0 + s * eps
    n = max(1, round(50 * s)); times = torch.linspace(s, 0.0, n + 1, dtype=torch.float64).tolist()
    for a, b in zip(times[:-1], times[1:]):
        x = x + (b - a) * prior.velocity(x, torch.full((len(x),), a, device=dev))
    return prior.unstandardise(x)
# pairs of person/face icons where a top-half (hair/hat) or ink-mask swap means something
rng = np.random.default_rng(0)
people = [i for i, h in enumerate(hx) if h.startswith(("1F468", "1F469", "1F9D1", "1F471", "1F474", "1F475", "1F9D4"))]
faces = [i for i, h in enumerate(hx) if h.startswith(("1F60", "1F61", "1F62", "1F92", "1F97", "1F91"))]
print(len(people), len(faces))
pairs = [(int(a), int(b)) for a, b in rng.choice(people, size=(4, 2), replace=False)] + [(int(a), int(b)) for a, b in rng.choice(faces, size=(2, 2), replace=False)] if len(faces) >= 4 else []
pairs += [(find("1F9DA"), find("1F6D1")), (find("1F5A4"), find("1F943"))]
gen = torch.Generator().manual_seed(7)
lat, meta = [], []
with strict_float32(), torch.no_grad():
    ia = torch.tensor([a for a, _ in pairs]); ib = torch.tensor([b for _, b in pairs])
    mua, _ = vt.posterior(val.images[ia].to(dev)); mub, _ = vt.posterior(val.images[ib].to(dev))
    xa, xb = prior.standardise(mua), prior.standardise(mub)
    top = torch.zeros(1, 1, 18, 18, device=dev); top[..., :8, :] = 1.0  # top 8 of 18 rows from B
    # B's ink mask at cell resolution
    ink = (val.images[ib].min(dim=-1).values < 250).float()  # (P,144,144)
    cells = torch.nn.functional.avg_pool2d(ink[:, None], 8).to(dev)  # (P,1,18,18)
    inkm = (cells > 0.05).float()
    for name, m in [("top", top), ("lerp", torch.full_like(top, 0.5))]:
        for s in [0.0, 0.3, 0.45]:
            for d in range(1 if s == 0 else 2):
                z = sdedit((1 - m) * xa + m * xb, s, gen)
                for k in range(len(pairs)): lat.append(z[k]); meta.append((name, k, s))
    L = torch.stack(lat); toks = torch.cat([greedy_decode(vt, L[i:i+24]) for i in range(0, len(L), 24)])
    # latency: 8 variations, s=0.45, end to end incl. posterior, flow, batched greedy decode (no graph)
    img = val.images[ia[:1]].to(dev)
    def vary8():
        mu, _ = vt.posterior(img); x0 = prior.standardise(mu).expand(8, -1, -1, -1)
        z = sdedit(x0, 0.45, torch.Generator().manual_seed(1)); return greedy_decode(vt, z)
    vary8(); torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        t0 = time.perf_counter(); vary8(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    print("8 variations s=0.45 end to end ms (batched greedy, float32, no graph):", [round(1000 * t) for t in ts])
    def flow8():
        mu, _ = vt.posterior(img); x0 = prior.standardise(mu).expand(8, -1, -1, -1)
        return sdedit(x0, 0.45, torch.Generator().manual_seed(1))
    flow8(); torch.cuda.synchronize(); t0 = time.perf_counter(); flow8(); torch.cuda.synchronize()
    print("flow part only ms:", round(1000 * (time.perf_counter() - t0)))
renders = _programs_to_renders(toks.cpu(), layout, template)
tg = val.targets.numpy()
rows = {}
for (name, k, s), r in zip(meta, renders): rows.setdefault(k, []).append(r)
print("frag rate by kind/s:")
for key in sorted(set((n, s) for n, _, s in meta)):
    fr = [ink_coverage(r) < 0.10 for (n, k, s), r in zip(meta, renders) if (n, s) == key]
    print(key, round(float(np.mean(fr)), 3), len(fr))
sheet = [[tg[a], tg[b]] + rows[k] for k, (a, b) in enumerate(pairs)]
(OUT / "mix2.png").write_bytes(contact_sheet(sheet))
print([(hx[a], hx[b]) for a, b in pairs])
