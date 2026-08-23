# Initial prompt for Codex on gtc

Read `AGENTS.md`, `PROJECT_PLAN.md`, and `DATA_CURATION.md` completely and treat
`AGENTS.md` as the binding operating contract. This is an exploratory research program
with no calendar schedule. Optimize for information gain, reproducibility, artifact
recovery, and honest negative results.

You are running in Codex YOLO mode. The CLI will not ask for approvals or enforce a
sandbox. YOLO changes execution mechanics only; it grants no authority beyond
`AGENTS.md` and `state/hosts.local.yaml`. Never rent/terminate machines, spend beyond a
recorded cap, delete artifacts, push/publish, widen credentials, or access the host
Docker socket.

Begin with bootstrap and continue autonomously through safe, authorized work:

1. Confirm this fresh control plane is named `gtc`, the repository is under
   `/home/dev/workspace`, and Docker points only to the isolated DinD daemon. Inspect
   Git status, local disk, installed tools, existing code, and orchestration state.
   Preserve all existing operator changes.
2. Create only the missing repository scaffold described in `AGENTS.md`, including a
   secret-safe `.gitignore`, `state/CURRENT.md`, and append-only run registry.
3. If `state/hosts.local.yaml` does not exist, copy `hosts.example.yaml` there. Never
   invent SSH aliases, credentials, artifact destinations, Kubernetes fields, or
   resource caps.
4. Implement local and remote read-only probes plus the common worker adapter interface.
   Add dry-run tests and make all remote commands non-interactive and quote-safe.
5. Do not rent, stop, destroy, resize, or change billing for any external instance.
6. Do not launch compute on a worker unless it has `enabled: true` and a complete
   `resource_cap` in `state/hosts.local.yaml`. That combination authorizes ordinary
   bounded experiments on that worker without repeatedly asking for permission.
7. Before meaningful GPU work, require a configured artifact sink and verify a tiny
   write/read/hash round trip from the target worker.
8. When a worker is enabled, probe it and record its actual GPU, driver, image digest,
   framework versions, storage, and connectivity. Run the complete tiny smoke adapter
   before any larger experiment.
9. Make data curation the first research implementation: immutable raw OpenMoji,
   reproducible audit metrics, quarantined contact sheets, reviewed reason-coded
   decisions, and `group == flags` excluded from the primary dataset but preserved as a
   named reversible split. Never delete upstream files.
10. Then execute the evidence gates in `PROJECT_PLAN.md`. At each gate, choose the next
   smallest experiment that can falsify or refine the current hypothesis. Keep failed
   runs and update `reports/findings.md`.
11. Maintain `state/CURRENT.md` after every material action and make active remote jobs,
    artifact durability, blockers, and the next action recoverable across tmux/SSH/Codex
    interruptions.

Proceed now with everything that is locally safe. Ask me only for the smallest set of
missing values that blocks the next remote or cost-bearing action. In your first report,
show:

- what you found;
- what you created or verified;
- whether the YOLO external hardening checks passed;
- any active worker authorization detected;
- the exact missing host/artifact/cap fields, if any;
- the next evidence-producing action.
