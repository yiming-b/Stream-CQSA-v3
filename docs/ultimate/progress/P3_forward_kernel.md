# P3 progress log: forward kernel v12f (ALIGNED run-table loop + LEGACY v11 fallback)

Package goal (task text, 2026-09-23): a CQS forward whose CQS-on instantiation on 128-aligned layouts is
upstream's loop plus a run table (no in-loop verdict loads, no next-live search, no per-element mask path),
exact everywhere. Variant tree: `/scratch/gpfs/AKEY/yb2807/Stream-CQSA-dev/next/kernel/v12f` (not git; module
`cqsa_cuda_next_v12f`); diff against `next/kernel/v11` committed as `docs/ultimate/kernel_diffs/v12f.diff`.

## State (update at every milestone)

| step | status | notes |
|---|---|---|
| read design 1/3/4/7, exp_build_ab.md, cuobjdump_v11x.md, v11 kernel | done 2026-09-23 | |
| clone v11 -> v12f, design refined | in progress | |
| build 1 (detached, vis1) | - | |
| correctness on vis1 (tests/test_ultimate_v12f.py) | - | |
| SXM job fwd_ab (benchmarks/ultimate/fwd_ab.py) | - | |
| results fwd_ab.json/md, diff, commit | - | |

## Design as built (refinements of the task's sketch)

(filled in below as the code lands)

## Job ids / builds

(none yet)

## Next step for a resumed agent

Read this file, then `ls next/kernel/v12f` and the build log `next/logs/05_build_v12f*.log`.
