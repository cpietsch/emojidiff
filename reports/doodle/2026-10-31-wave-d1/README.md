# Doodle to emoji, wave D1: fails its decisive criterion and does not work on real doodles

2026-10-31. Eight fine-tunes of the final model (`r2s-cur-doodle-ft-p05/p09-s1..4-*`, 30k steps at lr 3e-4 on
renders degraded by `mojidiff.learning.doodle` with probability 0.5 or 0.9), scored by `doodle_eval` on the 339
validation icons degraded once per kind (seed 1) against the clean targets. Criteria declared in
`state/CURRENT.md` at 16:00, before results.

| | clean | lineart | pencil (held out) |
| --- | --- | --- | --- |
| parent (final model) | 0.0645 | 0.1389 | 0.1445 |
| the degraded input itself, at 72 px | | **0.0969** | **0.1343** |
| vtracer compact on the degraded input | 0.0246 | 0.1070 | 0.1483 |
| doodle p0.5, mean of 4 seeds | 0.0739 | 0.1054 | 0.1376 |
| doodle p0.9, mean of 4 seeds | 0.0778 | 0.1054 | 0.1378 |

Paired over icons (per-icon mean of the arm's four seeds):

- (1) against the parent: lineart +0.034 [+0.029, +0.039], pencil +0.007 [+0.005, +0.009] (both arms). Passes, as
  expected.
- (2) decisive, against the degraded input: lineart -0.008 [-0.013, -0.004], pencil -0.003 [-0.006, -0.001]
  (both arms; every seed below zero on lineart); against vtracer: lineart +0.002 [-0.003, +0.006], pencil +0.011
  [+0.008, +0.014]. **Fails**: the model's drawing is further from the clean icon than the doodle it was given.
- (3) clean error rise: p0.5 +0.0094 [+0.0063, +0.0126] (passes at the point estimate), p0.9 +0.0133 (fails).
- Real doodles (exploratory; 130 QuickDraw drawings of 26 categories, CC BY 4.0, Google Creative Lab, drawn
  black at OpenMoji's 4 px outline width into the 22-122 px frame; `quickdraw_set.py`): the doodle model
  (p0.5 seed 2, the declared pick) fragments them like the parent; of the first drawing per category only the
  star comes out as a clean OpenMoji emoji (`quickdraw-doodle-vtracer-final-doodlemodel.png`). vtracer copies the
  doodle faithfully.

Reading: degradations of OpenMoji renders keep OpenMoji's shapes, proportions and framing, so the fine-tune
learns to undo the degradation of shapes it already knows; a real doodle has none of them and the model has no
template to snap to. The criteria (4) visual grades were not run: (2) fails and the real-doodle sheet is
unambiguous.
