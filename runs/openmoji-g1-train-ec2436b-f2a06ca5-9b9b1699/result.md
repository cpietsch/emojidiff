# Gate G dominant-bucket training run v1

Status: planned; no remote stage or GPU step yet.

First Gate G run whose outcome is a learning claim rather than pipeline liveness. It
trains the selected 577,552-parameter geometry denoiser for 600 bounded steps on a
256-icon family-disjoint subsample of the exact P32/T128 bucket and evaluates on 128
held-out icons from the disjoint validation split, tracing held-out recovery every 60
steps against an untrained step-0 control on the identical corruption draw.

The predeclared pass/fail criteria are recorded in `run.yaml` before launch. They are
deliberately falsifiable: the retained-token bar in particular is a real bet, because a
60-step CPU preflight reached only 0.0934 retained accuracy.
