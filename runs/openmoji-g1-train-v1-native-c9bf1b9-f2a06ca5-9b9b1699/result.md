# v1 config, native matched control on gpubox-4080

Status: completed in about 18 seconds.

Development moved off the `gtc` control plane. This run re-executes the **identical**
v1 config natively on the owned box under torch 2.14.0a0+4fdf77b940.nv26.08 and CUDA
13.4, replacing the container environment of torch
2.8.0a0+5228986c39.nv25.06 and CUDA 12.9.

| metric | container 2.8 / 12.9 | native 2.14 / 13.4 |
| --- | ---: | ---: |
| final train token accuracy | 0.455046275892 | 0.455046275892 |
| held-out changed accuracy | 0.055229646587 | 0.055229646587 |
| held-out aggregate accuracy | 0.231128774245 | 0.231153769246 |
| held-out retained accuracy | 0.325585862466 | 0.325624279677 |
| held-out loss | 11.148523330688 | 11.148344039917 |

Final train token accuracy and held-out changed-token accuracy are bit-identical.
Exactly one retained token of 26,030 differs, which moves aggregate accuracy in the
fifth decimal. Checkpoint bytes differ, as expected across framework versions; no
cross-environment artifact identity was claimed.

The migration therefore preserves every conclusion drawn from v1, including its
falsified retained-preservation criterion. This run, not the container parent, is the
matched baseline for the next data-scale experiment.
