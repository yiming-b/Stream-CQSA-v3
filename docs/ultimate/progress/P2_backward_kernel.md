# P2 progress log: backward kernel v12b (ALIGNED run-table backward + LEGACY v11 backward)

Package goal (task text, 2026-09-23): a CQS backward that on 128-aligned layouts runs at upstream speed on
live tiles and skips dead tiles for free, exact everywhere. Variant tree:
`/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/v12b` (not git; module `cqsa_cuda_next_v12b`, cloned from
v11x, forward keeps `-DCQSA_NO_ELEMENT_PATH`, backward WITHOUT `CQSA_BWD_NO_CQS`); diff against `next/kernel/v11`
committed as `docs/ultimate/kernel_diffs/v12b.diff`.

## Relayed status question (2026-09-23 13:12, answered here, package continues)

The user asked (relayed) to check the slurm jobs and report v2.2.3 and ultimate-kernel progress. squeue at
13:12: only 14312486 (cqsa_dist4_tele, pending Priority). sacct today: v2.2.3 side (main's unreleased
24a4ff7/5ba4c90/c4bc39b; the 12 formerly held jobs) -- paper mids 13908223/13908224/13915115/13915470 COMPLETED
(08:31-12:47), OOC forward 13952419/21/23/25 COMPLETED, OOC backward 13952418/20/22/24 OUT_OF_MEMORY (host
OOM, the known 4M-bwd-at-96G issue; the OOC harness fix is another package's). Ultimate program: P1.1 done
(v11x, job 14308401), I0 fixes landed (4ae02ea, edf0fb2, aeb401b) with fold_workers_ab 14313570 COMPLETED,
flash-attn 2.8.3 built (14240707), P3 forward v12f built (1926 s, 11:10) and its fwd_ab job not yet submitted,
P2 (this package) started 13:12.

## State (update at every milestone)

| step | status | notes |
|---|---|---|
| read design 1/3/4/7, exp_build_ab.md, cuobjdump_v11x.md, P3 log + census, v11/upstream bwd | done 13:10 | |
| clone v11x -> v12b, code written | - | |
| build b1 (detached, vis1) | - | |
| tests/test_ultimate_v12b.py on vis1 | - | |
| SXM job bwd_ab | - | |
| results, diff, commit | - | |
