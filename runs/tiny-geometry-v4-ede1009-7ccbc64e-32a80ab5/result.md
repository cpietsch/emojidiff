# Tiny geometry v4 result

Completed reproducibly; the diverse corruption-coverage hypothesis passed.

At unchanged batch size, model, probability, and step count, deterministic per-step
resampling raised diverse-four held-out accuracy from 73.82% to 98.56%. Changed-token
accuracy rose from 58.17% to 96.65%; retained-token accuracy rose from 82.52% to 99.63%.
Checkpoint continuation and the full artifact rerun are exact. The paired renders are
recognizable at both target sizes.

The unchanged static one-icon control remains below its added held-out threshold, so the
generic combined gate flag remains false. Apply the same treatment there before closing
Gate E.
