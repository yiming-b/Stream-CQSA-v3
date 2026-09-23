#!/bin/bash
# Freeze the worktree's current commit for a slurm job: prints the snapshot directory.
#   SNAP=$(bash slurm/ultimate/snapshot.sh) && sbatch --export=ALL,CQSA_SRC=$SNAP slurm/ultimate/<job>.slurm
# A queued job that imports the live worktree runs whatever is checked out when it starts; the
# snapshot (a `git archive` of one commit, so uncommitted edits are NOT included) cannot change.
set -euo pipefail
WT=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-ultimate
SHA=$(git -C "$WT" rev-parse --short HEAD)
SNAP=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-snapshots/ultimate-$SHA
if [ ! -f "$SNAP/.complete" ]; then
  mkdir -p "$SNAP"
  git -C "$WT" archive HEAD | tar -x -C "$SNAP"
  touch "$SNAP/.complete"
fi
echo "$SNAP"
