"""Evidence for the slot-invariance defect, reproducible from the committed tree."""
import copy, json, torch
from pathlib import Path
from mojidiff.learning.openmoji_pilot import (
    load_openmoji_pilot_config, load_pilot_index, _selected_codec, _new_model,
    _load_program, _select_rows, _condition,
)
from mojidiff.learning.geometry import packed_batch

cfg = load_openmoji_pilot_config(Path("configs/learning/openmoji-g1-noise-conditioned-v4.yaml"))
by_split, groups, subgroups = load_pilot_index(cfg)
codec = _selected_codec(cfg); dev = torch.device("cpu")
rows = _select_rows(by_split["primary/validation"], cfg.validation_samples, cfg.seed + 1)
torch.manual_seed(0); model = _new_model(cfg, codec, groups, subgroups).eval().double()
out = []
for k in range(6):
    row = rows[k]; prog = _load_program(row, cfg, codec)
    sw = copy.deepcopy(prog); n = 0
    for i in range(sw.coordinates.shape[0]):
        a, b = int(sw.coordinates[i,0]), int(sw.coordinates[i,1])
        if a != b and a != 0 and b != 0:
            sw.coordinates[i,0], sw.coordinates[i,1] = b, a; n += 1
    cond = _condition((row,), groups, subgroups, dev, [0.35])
    with torch.no_grad():
        s1, c1 = model(packed_batch([prog], dev), cond)
        s2, c2 = model(packed_batch([sw], dev), cond)
    out.append({"hexcode": row.hexcode, "segments_swapped": n,
                "max_abs_logit_difference": float((c1-c2).abs().max()),
                "identical_prediction": bool(torch.equal(c1.argmax(-1), c2.argmax(-1))
                                             and torch.equal(s1.argmax(-1), s2.argmax(-1)))})
print(json.dumps({"precision": "float64", "rows": out,
                  "total_segments_swapped": sum(r["segments_swapped"] for r in out)}, indent=2))
