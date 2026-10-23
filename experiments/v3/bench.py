"""
The v3 experiment harness: one script for every row of experiments/v3/PLAN.md (E0-E6), one JSONL row per
(experiment, configuration, arm) under results/v3/, the tables / fits / figures generated from them.

    python -P experiments/v3/bench.py run --exp E1 --part fwd --out results/v3 [--budget-s 1500] [--only-missing]
    python -P experiments/v3/bench.py worker '<json row>'            (one configuration in its own process; prints RESULT ...)
    python -P experiments/v3/bench.py report --out results/v3        (report.md, fits.json, E*.png, STATUS.md)
    python -P experiments/v3/bench.py status --out results/v3        (STATUS.md's summary; last line "MISSING <n>")
    python -P experiments/v3/bench.py run --exp E1 --part fwd --smoke --out <scratch>   (vis1: plumbing only)

The plan (PLAN below, one entry per experiment) encodes the parameter grid of PLAN.md: the varied and the fixed
parameters, the arms (the four production series classic_CUDA / classic_Triton / wave_CUDA / wave_Triton and the
monolithic baselines FA-2 / SDPA), the parts a slurm job runs separately, the timing protocol (warm-up + timed
runs) and the reference policy. `run` enumerates the rows of one experiment (or part), skips the rows already in
results.jsonl (key = experiment, arm, N, c, itr, acc, qkv residency, dtype, D, direction, extra, node class),
runs each remaining row in a fresh subprocess (`worker`; allocator state cannot leak between rows), appends one
JSON row per configuration (status ok / oom / error, with the message), prints one line per row, and under
`--budget-s` runs only the rows whose estimated wall time still fits, so the job's time-box never kills a row
half-way: the next submission continues with `--only-missing`. Rows are ordered cheapest first within a part
(small N first, the baselines before the series).

Every Stream-CQSA row pins itr and c (tests/conftest.py's rule) and asserts the decomposition it reports
(tests/_decomp.py's assert_decomposed); the row records the engine / kernel / depth / subproblem count that ran.
E1 / E2 use the depth rule of PLAN.md: itr = the smallest depth >= 1 whose plan fits the device (the planner,
`attention(..., plan_only=True)`, with the series' engine and the pinned c); a row that still runs out of
memory at that depth retries at itr + 1 and records the failed attempt next to the depth that ran.

Metrics per row (PLAN.md): wall_s (median) and wall_min_s of the timed runs; peak_dev_gib (torch's
max_memory_allocated over the call) and workspace_gib (peak minus what was resident before the call);
peak_host_gib (VmHWM delta over the timed runs); rel_err / rel_err_grads vs the FA-2 reference computed on the
same inputs in the same worker (untimed; rows whose reference policy is "spot" and rows where FA-2 does not
fit skip it; without flash_attn the reference arm is torch SDPA on its flash backend, recorded and labelled
"SDPA-flash" in every row and table, never called FA-2; `--ref sdpa` selects it explicitly); fp64_err (output) and fp64_err_dq (gradient) on 128 sampled rows against an fp64 reference
computed on the device in chunks, next to FA-2's own fp64 error on the same rows (fp64_ratio_fa2); engine,
kernel, n_subproblems, n_waves, itr_ran from return_info; GPU name (nvidia-smi and torch), module names
(cqsa_cuda_v13 / cqsa_native_v2), git commit, slurm job id.

Residency: device-resident Q/K/V (fp16 / bf16 [B=1, H=8, N, D]) unless the row says qkv=host: then pinned
token-major host tensors ([B, N, H, D] storage viewed as [B, H, N, D], the layout the streaming engines gather
from) and a head-major pinned dO (the backward's cheaper layout, experiments/ooc_rivals). The FA-2 arm takes
its token-major copies untimed. fwd+bwd rows time one forward (inputs requiring grad) plus
torch.autograd.grad of the output; the backward's depth is pinned with bwd_itr=.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
K, M = 1024, 1 << 20
B, H = 1, 8
SERIES = ("classic_CUDA", "wave_CUDA", "classic_Triton", "wave_Triton")
BASELINES = ("FA-2", "SDPA")
KERNEL_OF = {"classic_CUDA": "cuda", "classic_Triton": "triton", "wave_CUDA": "wave-cuda", "wave_Triton": "wave-triton"}
ARM_ORDER = {a: i for i, a in enumerate(BASELINES + SERIES)}
N_LADDER = [128 * K, 256 * K, 512 * K, M, 2 * M, 4 * M]
C_LIST = [3, 7, 13, 21, 31, 57, 73, 91, 133]
MAX_SUBPROBLEMS = 4096          # the planner's cap on c**itr (autoconfig.plan): c=73 / 133 at itr=2 cannot run
SPOT_ROWS = 128
DEFAULT_OUT = os.path.join(ROOT, "results", "v3")
SMOKE_OUT = os.path.join(ROOT, ".smoke", "v3")
KEY_FIELDS = ("exp", "arm", "N", "c", "itr", "acc", "qkv", "dtype", "D", "direction", "extra", "node")

# ---------------------------------------------------------------------------------------------------------------
# the plan (experiments/v3/PLAN.md), one entry per experiment
# ---------------------------------------------------------------------------------------------------------------
# grid: the varied x fixed parameters (every key a list); extra_grids: additional partial grids (E0's bf16 / D=128
# rows); arms: the series and baselines; qkv: "device" | "host" | "by_acc" (acc=cpu rows are host-resident);
# parts: name -> filter on the row (the slurm job runs one part per call); warmup / reps: the timing protocol;
# ref: "fa2" (FA-2 output and gradients + the fp64 spot check) or "spot" (the fp64 spot check only: free at any N).
PLAN = {
    "E0": dict(
        question="correctness: every series and every (c, itr, acc) path is exact (output and gradients vs FA-2 and vs fp64)",
        grid=dict(N=[256 * K], c=[3, 7, 31, 133], itr=[1, 2], acc=["gpu", "cpu"], dtype=["fp16"], D=[64], direction=["fwd", "bwd"]),
        extra_grids=[dict(N=[256 * K], c=[7], itr=[1], acc=["gpu"], dtype=["bf16"], D=[64], direction=["fwd", "bwd"]),
                     dict(N=[256 * K], c=[7], itr=[1], acc=["gpu"], dtype=["fp16"], D=[128], direction=["fwd", "bwd"])],
        arms=SERIES, qkv="by_acc", warmup=0, reps=1, ref="fa2", node="sxm",
        parts={"all": {}}),
    "E1": dict(
        question="scaling: wall and peak memory vs N, device-resident, at the depth the memory allows",
        grid=dict(N=N_LADDER, c=[7], itr=["min"], acc=["gpu"], dtype=["fp16"], D=[64], direction=["fwd", "bwd"]),
        arms=SERIES + BASELINES, qkv="device", warmup=1, reps=1, ref="spot", node="sxm",
        parts={"fwd": dict(direction="fwd"), "bwd": dict(direction="bwd")}),
    "E2": dict(
        question="scaling, host-resident: Q/K/V and the accumulator in host memory (the recovery configuration)",
        grid=dict(N=N_LADDER, c=[7], itr=["min"], acc=["cpu"], dtype=["fp16"], D=[64], direction=["fwd", "bwd"]),
        # the PCIe part: 4M and 8M host-resident rows with their own resident baselines (never compared across node types)
        extra_grids=[dict(N=[4 * M, 8 * M], c=[7], itr=["min"], acc=["cpu"], dtype=["fp16"], D=[64], direction=["fwd", "bwd"],
                          node="pcie", arms=SERIES + BASELINES)],
        arms=SERIES, qkv="host", warmup=1, reps=1, ref="spot", node="sxm",
        parts={"fwd": dict(direction="fwd", node="sxm"), "bwd": dict(direction="bwd", node="sxm"),
               "pcie_fwd": dict(direction="fwd", node="pcie"), "pcie_bwd": dict(direction="bwd", node="pcie")}),
    "E3": dict(
        question="parameters: how the quorum size c shapes time and memory",
        grid=dict(N=[M], c=C_LIST, itr=[1], acc=["gpu"], dtype=["fp16"], D=[64], direction=["fwd", "bwd"]),
        arms=SERIES, qkv="device", warmup=1, reps=3, ref="fa2", node="sxm",
        parts={"fwd": dict(direction="fwd"), "bwd": dict(direction="bwd")}),
    "E4": dict(
        question="parameters: depth x accumulator placement at one size (device-resident Q/K/V)",
        grid=dict(N=[M], c=[7], itr=[1, 2], acc=["gpu", "cpu"], dtype=["fp16"], D=[64], direction=["fwd", "bwd"]),
        arms=SERIES, qkv="device", warmup=1, reps=3, ref="fa2", node="sxm",
        parts={"all": {}}),
    "E5": dict(
        question="parameters: the wave budget W (subproblems per launch)",
        grid=dict(N=[M], c=[73], itr=[1], acc=["gpu"], dtype=["fp16"], D=[64], direction=["fwd", "bwd"],
                  extra=["W=1x", "W=2x", "W=4x", "W=8x", "W=16x"]),      # multiples of the largest subproblem's tokens
        arms=("wave_CUDA", "wave_Triton"), qkv="device", warmup=1, reps=3, ref="fa2", node="sxm",
        parts={"all": {}}),
    "E6": dict(
        question="parameters: precision and head dim (performance; E0 has their correctness)",
        grid=dict(N=[M], c=[7], itr=[1], acc=["gpu"], dtype=["fp16", "bf16"], D=[64, 128], direction=["fwd", "bwd"]),
        arms=SERIES + BASELINES, qkv="device", warmup=1, reps=3, ref="fa2", node="sxm",
        parts={"all": {}}),
}
EXPERIMENTS = tuple(PLAN)


def _rows_of_grid(exp, spec, grid, arms, node):
    axes = dict(grid)
    axes.setdefault("extra", [None])
    names = list(axes)
    for values in itertools.product(*(axes[n] for n in names)):
        g = dict(zip(names, values))
        for arm in arms:
            r = dict(exp=exp, arm=arm, kind="series" if arm in SERIES else "baseline", N=int(g["N"]), dtype=g["dtype"],
                     D=int(g["D"]), direction=g["direction"], extra=g["extra"], node=node,
                     warmup=spec["warmup"], reps=spec["reps"], ref=spec["ref"])
            if arm in SERIES:
                r.update(c=g["c"], itr=g["itr"], acc=g["acc"])
                r["qkv"] = ("host" if g["acc"] == "cpu" else "device") if spec["qkv"] == "by_acc" else spec["qkv"]
                if isinstance(g["itr"], int) and g["c"] ** g["itr"] > MAX_SUBPROBLEMS:
                    continue                                   # the planner refuses it (c**itr > 4096 tasks)
                if arm not in ("wave_CUDA", "wave_Triton") and g["extra"] and str(g["extra"]).startswith("W="):
                    continue
            else:
                r.update(c=None, itr=None, acc=None, qkv="device")
            yield r


def plan_rows(exp, part=None, smoke=False):
    """The rows of one experiment (one part of it), in the order they run."""
    spec = PLAN[exp]
    grids = [dict(spec["grid"])] + [dict(g) for g in spec.get("extra_grids", [])]
    rows, seen = [], set()
    for g in grids:
        node = g.pop("node", spec["node"])
        arms = g.pop("arms", spec["arms"])
        if smoke:
            if len(g["N"]) > 1:
                g["N"] = [16384, 32768]
            else:
                g["N"] = [32768]
            if len(g.get("c", [])) > 2:
                g["c"] = [g["c"][0], g["c"][-1]]
        for r in _rows_of_grid(exp, spec, g, arms, node):
            if smoke:
                r.update(warmup=0, reps=1, smoke=True)
            if known_exclusion(r):
                continue
            k = row_key(r)
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)
    if part is not None:
        if part not in spec["parts"]:
            raise SystemExit(f"{exp} has parts {list(spec['parts'])}, not {part!r}")
        flt = spec["parts"][part]
        rows = [r for r in rows if all(r.get(k_) == v_ for k_, v_ in flt.items())]
        for r in rows:
            r["part"] = part
    else:
        for r in rows:
            r["part"] = next((p for p, flt in spec["parts"].items() if all(r.get(k_) == v_ for k_, v_ in flt.items())), "all")
    rows.sort(key=row_order)
    return rows


# Rows that are not measurements but findings: configurations the current code cannot run to completion.
# They are left out of the plan (STATUS.md counts them as excluded, not missing) and reported in report.md.
KNOWN_EXCLUSIONS = [
    # The Triton backward at head dim 128 never finishes at N=1M: every other E6 cell completed (Triton D=64
    # backward 37 s, CUDA D=128 backward 38 s) while every Triton D=128 backward row hit the 33-minute part
    # timeout with nothing recorded, in five consecutive jobs (16084365..16090304, A100-SXM4-80GB). The map
    # records the Triton backward's 128-wide key tiles as pathological (results/triton_bwd_bench.json: up to
    # 1955 ms per tile); the CUDA backward is the default, so this path only matters when nothing is compiled.
    dict(arm=("classic_Triton", "wave_Triton"), D=128, direction="bwd",
         reason="Triton backward at D=128 does not finish (known 128-wide key-tile pathology); CUDA backward is the default"),
]


def known_exclusion(r):
    """The reason this row is excluded from the plan, or None."""
    for x in KNOWN_EXCLUSIONS:
        if all((r.get(k) in v) if isinstance(v, tuple) else (r.get(k) == v) for k, v in x.items() if k != "reason"):
            return x["reason"]
    return None


def row_order(r):
    itr = r["itr"] if isinstance(r["itr"], int) else (0 if r["itr"] is None else 9)
    extra = str(r["extra"] or "")
    if extra.startswith("W=") and extra.endswith("x"):
        extra = f"W={int(extra[2:-1]):04d}"                 # W=1x < W=2x < ... < W=16x
    return (r["node"], r["N"], r["D"], r["dtype"], 0 if r["direction"] == "fwd" else 1, ARM_ORDER[r["arm"]],
            r["c"] or 0, itr, r["acc"] or "", extra)


def row_key(r):
    return tuple(r.get(f) for f in KEY_FIELDS)


def estimate_row_s(r, history=None):
    """Rough wall time of one worker (startup + inputs + references + warm-up + timed runs), for the budget: a quadratic
    model anchored at 6 s per forward at N=1M on an A100-SXM4-80GB (results/ultimate/engine_ab.md), 3.6x for fwd+bwd,
    1.15x (fwd) / 1.8x (fwd+bwd) for the Triton kernel, 1.3x for host-resident / host-accumulator rows; the measured wall
    of the same configuration (any status) overrides it when larger."""
    x = r["N"] / M
    t = 6.0 * x * x * (r["D"] / 64)
    if r["direction"] == "bwd":
        t *= 3.6
    if r["arm"] in SERIES:
        if "Triton" in r["arm"]:
            t *= 1.8 if r["direction"] == "bwd" else 1.15      # engine_ab: the Triton backward is ~1.9x the v13 CUDA one
        if r["qkv"] == "host" or r["acc"] == "cpu":
            t *= 1.3
        if r["itr"] == "min":
            t *= 1.1                                           # a planned depth > 1 costs a little more
    if r["node"] == "pcie":
        t *= 1.25
    calls = r["warmup"] + r["reps"] + (1 if r["ref"] == "fa2" else 0)
    est = 30.0 + calls * t + (12.0 * x if r.get("qkv") == "host" else 2.0 * x)   # host: pinned allocation + generation
    if history:
        same = [h.get("row_wall_s", 0.0) for h in history
                if (h["exp"], h["arm"], h["N"], h["D"], h["dtype"], h["direction"], h.get("qkv")) ==
                (r["exp"], r["arm"], r["N"], r["D"], r["dtype"], r["direction"], r.get("qkv")) and h.get("row_wall_s")]
        if same:
            est = max(est, 1.15 * max(same))
    return est


# ---------------------------------------------------------------------------------------------------------------
# results file
# ---------------------------------------------------------------------------------------------------------------
def read_rows(out_dir):
    path = os.path.join(out_dir, "results.jsonl")
    rows = []
    if os.path.exists(path):
        for ln in open(path):
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
    return rows


def latest_by_key(rows):
    d = {}
    for r in rows:
        d[row_key(r)] = r                   # the last row per key wins
    return d


def append_row(out_dir, r):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.jsonl"), "a") as f:
        f.write(json.dumps(r, default=str) + "\n")
        f.flush()


# ---------------------------------------------------------------------------------------------------------------
# the worker: one configuration in its own process
# ---------------------------------------------------------------------------------------------------------------
def gpu_info():
    import torch
    info = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, host=socket.gethostname(),
                slurm_job=os.environ.get("SLURM_JOB_ID"))
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        info["nvidia_smi"] = r.stdout.strip()
    except Exception as e:                                       # noqa: BLE001
        info["nvidia_smi"] = f"unavailable: {e!r}"
    try:
        info["git_commit"] = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True,
                                            text=True, timeout=30).stdout.strip() or None
    except Exception:                                            # noqa: BLE001
        info["git_commit"] = None
    if not info["git_commit"]:
        # a frozen snapshot (git archive) has no .git: a GIT_COMMIT file (dist_cita/make_bundle.sh writes one) or the
        # snapshot directory name carries the commit
        gc_file = os.path.join(ROOT, "GIT_COMMIT")
        if os.path.exists(gc_file):
            words = open(gc_file).read().split()
            info["git_commit"] = words[0] if words else None
        else:
            info["git_commit"] = os.path.basename(ROOT).split("-")[-1] if "-" in os.path.basename(ROOT) else None
    info["code_root"] = ROOT
    return info


def module_info():
    import stream_cqsa
    from stream_cqsa import interface as I
    out = dict(package=os.path.dirname(stream_cqsa.__file__), version=getattr(stream_cqsa, "__version__", None),
               cuda_module=I.CQSA_CUDA_MODULE, cuda_module_file=getattr(I.cqsa_cuda, "__file__", None),
               backward_kernel=I.backward_kernel() if hasattr(I, "backward_kernel") else None)
    try:
        from stream_cqsa import native_wave as W
        ext = W.native_ext()
        out.update(native_module=ext.__name__ if ext is not None else None,
                   native_module_file=getattr(ext, "__file__", None) if ext is not None else None)
    except Exception as e:                                       # noqa: BLE001
        out.update(native_module=None, native_module_error=repr(e))
    try:
        import flash_attn
        out["flash_attn"] = flash_attn.__version__
    except Exception:                                            # noqa: BLE001
        out["flash_attn"] = None
    return out


def vm_hwm_bytes():
    try:
        for ln in open("/proc/self/status"):
            if ln.startswith("VmHWM:"):
                return int(ln.split()[1]) * 1024
    except Exception:                                            # noqa: BLE001
        pass
    return 0


def is_oom(exc):
    import torch
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    s = str(exc).lower()
    return ("out of memory" in s or "cuda_error_out_of_memory" in s or "cudaErrorMemoryAllocation".lower() in s
            or "cannot allocate memory" in s or "cuda error: out of memory" in s)


def sanitize(info, depth=0):
    """JSON-serialisable copy of an engine info dict (tensors dropped, tuples listed)."""
    if info is None:
        return None
    out = {}
    for k_, v_ in info.items():
        if hasattr(v_, "device") or k_ == "lse":
            continue
        if isinstance(v_, dict):
            out[k_] = sanitize(v_, depth + 1) if depth < 3 else str(v_)[:200]
        elif isinstance(v_, (list, tuple)):
            out[k_] = [x if isinstance(x, (int, float, str, bool, type(None))) else str(x) for x in list(v_)[:64]]
        elif isinstance(v_, (int, float, str, bool, type(None))):
            out[k_] = v_
        else:
            out[k_] = str(v_)[:200]
    return out


def _decomp_helpers():
    """tests/_decomp.py's assert_decomposed (imported by path; the snapshot carries tests/), or a replica."""
    try:
        import importlib.util
        p = os.path.join(ROOT, "tests", "_decomp.py")
        spec = importlib.util.spec_from_file_location("_decomp_v3", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.assert_decomposed
    except Exception:                                            # noqa: BLE001
        def assert_decomposed(info, *, itr=None, c=None, engine=None, n_subproblems=None):
            assert isinstance(info, dict)
            if "engine" in info:
                assert info["engine"] in ("classic", "wave"), f"the monolithic path ran (engine={info['engine']!r}); info={info}"
            assert not info.get("monolithic", False), f"the engine took its monolithic flash branch; info={info}"
            n = info.get("n_subproblems")
            if n is not None:
                assert n > 1, f"n_subproblems={n}: no decomposition ran; info={info}"
            else:
                it = info.get("itr")
                assert isinstance(it, int) and it >= 1, f"no subproblem count and itr={it!r}; info={info}"
            if itr is not None:
                assert info.get("itr") == itr, f"itr={info.get('itr')!r}, pinned {itr}; info={info}"
            if c is not None and "c" in info:
                assert info["c"] == c, f"c={info['c']!r}, pinned {c}; info={info}"
            return info
        return assert_decomposed


def make_tensor(N, D, dtype, *, seed, device, layout, pin, requires_grad=False, chunk=M):
    """Seeded [B, H, N, D] tensor, generated on the device in chunks of the sequence (so device- and host-resident
    rows hold the same values). layout "head": [B, H, N, D] contiguous; "token": [B, N, H, D] storage viewed as
    [B, H, N, D]. device "cuda" or "cpu" (pinned when `pin`; falls back to pageable memory with a note)."""
    import torch
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    g = torch.Generator(device="cuda").manual_seed(int(seed))
    pinned = False
    if device == "cpu":
        shape = (B, N, H, D) if layout == "token" else (B, H, N, D)
        try:
            t = torch.empty(shape, dtype=dt, device="cpu", pin_memory=True) if pin else torch.empty(shape, dtype=dt)
            pinned = bool(pin)
        except Exception:                                        # noqa: BLE001  (pinned memory exhausted)
            t = torch.empty(shape, dtype=dt)
    else:
        shape = (B, N, H, D) if layout == "token" else (B, H, N, D)
        t = torch.empty(shape, dtype=dt, device="cuda")
    view = t.transpose(1, 2) if layout == "token" else t
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        x = torch.randn(B, H, e - s, D, generator=g, device="cuda", dtype=dt)
        view[:, :, s:e].copy_(x)
        del x
    torch.cuda.synchronize()
    if requires_grad:
        view.requires_grad_(True)
    return view, pinned


def _chunk_to(t, s, e, device, dtype):
    """t[:, :, s:e] on `device` as `dtype`; host token-major storage is sliced contiguously (one DMA)."""
    tm = t.transpose(1, 2)
    if not t.is_cuda and tm.is_contiguous():
        return tm[:, s:e].to(device=device, non_blocking=False).transpose(1, 2).to(dtype)
    return t[:, :, s:e].to(device=device).to(dtype)


def spot_row_indices(N, n=SPOT_ROWS, seed=1234):
    import torch
    g = torch.Generator().manual_seed(seed)
    rows = set(torch.randint(0, N, (n,), generator=g).tolist()) | {0, N - 1}
    return sorted(rows)


def ref64_forward(q, k, v, rows, scale, chunk=16 * K):
    """fp64 causal attention of the sampled query rows against every key, on the device in chunks of keys.
    Returns (out64 [B, H, R, D], lse64 [B, H, R])."""
    import torch
    N = q.shape[2]
    dev = torch.device("cuda")
    ridx = torch.tensor(rows, device=dev)
    qs = q.detach()[:, :, rows].to(dev).double() if q.is_cuda else q.detach()[:, :, rows].to(dev).double()
    R = len(rows)
    m = torch.full((B, H, R), -math.inf, device=dev, dtype=torch.float64)
    l = torch.zeros((B, H, R), device=dev, dtype=torch.float64)
    acc = torch.zeros((B, H, R, q.shape[3]), device=dev, dtype=torch.float64)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        kc = _chunk_to(k.detach(), s, e, dev, torch.float64)
        vc = _chunk_to(v.detach(), s, e, dev, torch.float64)
        S = torch.matmul(qs, kc.transpose(-1, -2)) * scale                        # [B, H, R, e-s]
        j = torch.arange(s, e, device=dev)
        S.masked_fill_(j[None, None, None, :] > ridx[None, None, :, None], -math.inf)
        m_new = torch.maximum(m, S.amax(-1))
        p = torch.exp(S - m_new[..., None])
        alpha = torch.exp(m - m_new)
        l = l * alpha + p.sum(-1)
        acc = acc * alpha[..., None] + torch.matmul(p, vc)
        m = m_new
        del kc, vc, S, p
    return acc / l[..., None], m + torch.log(l)


def ref64_dq(q, k, v, dout, out64, lse64, rows, scale, chunk=16 * K):
    """fp64 dQ of the sampled rows: dq_i = scale * sum_j P_ij (dP_ij - delta_i) k_j with P from the fp64 lse."""
    import torch
    N = q.shape[2]
    dev = torch.device("cuda")
    ridx = torch.tensor(rows, device=dev)
    qs = q.detach()[:, :, rows].to(dev).double()
    ds = dout.detach()[:, :, rows].to(dev).double()
    delta = (ds * out64).sum(-1)                                                   # [B, H, R]
    dq = torch.zeros_like(qs)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        kc = _chunk_to(k.detach(), s, e, dev, torch.float64)
        vc = _chunk_to(v.detach(), s, e, dev, torch.float64)
        S = torch.matmul(qs, kc.transpose(-1, -2)) * scale
        j = torch.arange(s, e, device=dev)
        S.masked_fill_(j[None, None, None, :] > ridx[None, None, :, None], -math.inf)
        P = torch.exp(S - lse64[..., None])
        dP = torch.matmul(ds, vc.transpose(-1, -2))
        dS = P * (dP - delta[..., None])
        dq += torch.matmul(dS, kc)
        del kc, vc, S, P, dP, dS
    return dq * scale


def rel_rows(a, rows, ref64):
    """norm-relative error of a[:, :, rows] (any device / dtype) against the fp64 reference rows."""
    import torch
    x = a.detach()[:, :, rows].to(ref64.device).double()
    return ((x - ref64).norm() / ref64.norm()).item()


def rel_full(a, b, chunk=256 * K):
    """norm-relative error ||a - b|| / ||b|| over the whole [B, H, N, D] tensors, in fp64 on the device, chunked."""
    import torch
    N = a.shape[2]
    dev = torch.device("cuda")
    num = den = 0.0
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        ac = _chunk_to(a.detach(), s, e, dev, torch.float64)
        bc = _chunk_to(b.detach(), s, e, dev, torch.float64)
        num += (ac - bc).pow(2).sum().item()
        den += bc.pow(2).sum().item()
        del ac, bc
    return math.sqrt(num / den) if den > 0 else float("nan")


def flash_attn_importable():
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:                                            # noqa: BLE001
        return False


def resolve_ref_arm(policy):
    """The reference arm of a row's reference policy: "fa2" -> "FA-2" when flash_attn is importable, else
    "SDPA-flash" (torch SDPA on its flash backend: the same algorithm, a different build; recorded as such in the
    row, never called FA-2); "sdpa" -> "SDPA-flash"; "spot" / None -> None (the fp64 spot check only)."""
    if policy == "fa2":
        return "FA-2" if flash_attn_importable() else "SDPA-flash"
    if policy == "sdpa":
        return "SDPA-flash"
    return None


def sdpa_flash_reference(q, k, v, dout, scale, want_grads):
    """torch SDPA (flash backend) on head-major device copies of the inputs; returns (out [B,H,N,D] on the host,
    grads on the host or None). The device copies are released before returning."""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend
    dev = torch.device("cuda")
    N, D = q.shape[2], q.shape[3]

    def hm(t):          # head-major contiguous device copy
        x = torch.empty((B, H, N, D), dtype=t.dtype, device=dev)
        for s in range(0, N, M):
            e = min(N, s + M)
            x[:, :, s:e].copy_(_chunk_to(t.detach(), s, e, dev, t.dtype))
        return x
    qf, kf, vf = (hm(t).requires_grad_(want_grads) for t in (q, k, v))
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        of = F.scaled_dot_product_attention(qf, kf, vf, is_causal=True, scale=scale)
    grads = None
    if want_grads:
        df = hm(dout)
        gf = torch.autograd.grad(of, (qf, kf, vf), df)
        grads = [x.detach().cpu() for x in gf]
        del gf, df
    out = of.detach().cpu()
    del qf, kf, vf, of
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return out, grads


def arm_reference(q, k, v, dout, scale, want_grads, ref_arm):
    if ref_arm == "FA-2":
        return fa2_reference(q, k, v, dout, scale, want_grads)
    if ref_arm == "SDPA-flash":
        return sdpa_flash_reference(q, k, v, dout, scale, want_grads)
    raise ValueError(f"unknown reference arm {ref_arm!r}")


def fa2_reference(q, k, v, dout, scale, want_grads):
    """FA-2 (flash_attn_func) on token-major copies of the inputs; returns (out [B,H,N,D] view on the host, grads on
    the host or None). The device copies are released before returning."""
    import torch
    from flash_attn import flash_attn_func
    dev = torch.device("cuda")
    N, D = q.shape[2], q.shape[3]

    def tm(t):          # token-major device copy
        x = torch.empty((B, N, H, D), dtype=t.dtype, device=dev)
        for s in range(0, N, M):
            e = min(N, s + M)
            x[:, s:e].copy_(_chunk_to(t.detach(), s, e, dev, t.dtype).transpose(1, 2))
        return x
    qf, kf, vf = (tm(t).requires_grad_(want_grads) for t in (q, k, v))
    of = flash_attn_func(qf, kf, vf, causal=True, softmax_scale=scale)
    grads = None
    if want_grads:
        df = tm(dout)
        gf = torch.autograd.grad(of, (qf, kf, vf), df)
        grads = [x.detach().cpu().transpose(1, 2) for x in gf]
        del gf, df
    out = of.detach().cpu().transpose(1, 2)
    del qf, kf, vf, of
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return out, grads


def _patch_classic_autograd(capture):
    """Make the classic engine's autograd op report what ran: capture the forward's info and the backward's
    bwd_info (depth reached, escalations, OOM retries) and forbid silent depth escalations (allow_escalation=False)
    so a pinned depth that does not fit is an OOM row, not a different configuration."""
    import stream_cqsa.native_autograd as NA
    fwd0, bwd0 = NA.stream_cqsa_forward, NA.stream_cqsa_backward

    def fwd(*a, **kw):
        kw.setdefault("allow_escalation", False)
        out, info = fwd0(*a, **kw)
        capture["forward"] = sanitize(info)
        return out, info

    def bwd(*a, **kw):
        binfo = {}
        kw.setdefault("bwd_info", binfo)
        kw.setdefault("allow_escalation", False)
        try:
            return bwd0(*a, **kw)
        finally:
            capture["backward"] = sanitize(binfo)
    NA.stream_cqsa_forward, NA.stream_cqsa_backward = fwd, bwd


def smallest_depth_that_fits(q, k, v, kernel, c, iset, acc, max_itr=4):
    """PLAN.md's depth rule for E1 / E2: the smallest itr >= 1 whose plan (the planner with this series' engine and
    the pinned c, at this residency) fits the device. Returns (itr, plan name, reason); the deepest depth tried
    when nothing fits (the row then runs and records its OOM)."""
    import stream_cqsa
    last = None
    for itr in range(1, max_itr + 1):
        try:
            p = stream_cqsa.attention(q, k, v, is_causal=True, kernel=kernel, plan_only=True, itr=itr, c=c, interest_set=iset,
                                      accumulate_on_gpu=(acc == "gpu"), low_memory=(acc == "cpu"))
        except ValueError as e:                                   # no decomposition at this depth (task cap)
            last = (itr, None, f"planner: {e}")
            continue
        host = not q.is_cuda
        fits = any(x["ok"] and x["mode"] == "cqsa" and x["itr"] == itr and x["c"] == c and x["acc"] == acc
                   and bool(x["stream_from_host"]) == host for x in p.candidates)
        last = (itr, p.name(), p.reason)
        if fits:
            return itr, p.name(), p.reason, True
    return last[0], last[1], last[2], False


def run_timed(call, warmup, reps, *, bwd_fn=None):
    """1 + warm-up + timed runs of `call` (and `bwd_fn(out)` for fwd+bwd rows); wall around the call(s) + synchronize,
    peak device memory per run. Returns (last out, last grads, stats)."""
    import torch
    times, peaks, resident = [], [], []
    out = grads = None
    for i in range(warmup + reps):
        del out, grads
        out = grads = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        resident.append(torch.cuda.memory_allocated())
        t0 = time.perf_counter()
        out = call()
        if bwd_fn is not None:
            grads = bwd_fn(out)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peaks.append(torch.cuda.max_memory_allocated())
        if i >= warmup:
            times.append(dt)
    st = dict(times_s=times, wall_s=statistics.median(times), wall_min_s=min(times), peak_dev_gib=max(peaks) / 2**30,
              resident_before_gib=max(resident) / 2**30, workspace_gib=(max(peaks) - max(resident)) / 2**30)
    return out, grads, st


def worker(row):
    import torch
    t_start = time.time()
    res = dict(row)
    res.update(gpu_info())
    try:
        res.update(module_info())
    except Exception as e:                                       # noqa: BLE001
        res["module_info_error"] = repr(e)
    try:
        _worker(row, res)
        res.setdefault("status", "ok")
    except Exception as e:                                       # noqa: BLE001
        if is_oom(e):
            res["status"] = "oom"
            res["error"] = str(e)[:3000]
        else:
            res["status"] = "error"
            res["error"] = f"{type(e).__name__}: {e}"[:3000]
            res["traceback"] = traceback.format_exc()[-6000:]
    res["worker_wall_s"] = time.time() - t_start          # inside the worker; the run loop records the subprocess' wall as row_wall_s
    res["vmhwm_gib"] = vm_hwm_bytes() / 2**30
    return res


def _worker(row, res):
    import torch
    from stream_cqsa.autoconfig import QUORUM_SETS
    arm, N, D, dtype, direction = row["arm"], int(row["N"]), int(row["D"]), row["dtype"], row["direction"]
    bwd = direction == "bwd"
    scale = float(D) ** -0.5
    host = row.get("qkv") == "host"
    seed = N * 1000 + D + (7 if dtype == "bf16" else 0)
    torch.manual_seed(0)
    if row.get("mem_cap_gib"):                               # plumbing tests of the OOM paths on a shared GPU
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(row["mem_cap_gib"]) * 2**30 / total))

    # ---- JIT / module warm-up on a tiny call (Triton compiles, module and kernel loads, streams): never timed -----
    tdt = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    qw, kw_, vw = (torch.randn(B, H, 16384, D, device="cuda", dtype=tdt).requires_grad_(bwd) for _ in range(3))
    if arm in SERIES:
        import stream_cqsa
        ow = stream_cqsa.attention(qw, kw_, vw, is_causal=True, kernel=KERNEL_OF[arm], itr=1, c=7, interest_set=QUORUM_SETS[7],
                                   bwd_itr=1)
    elif arm == "FA-2":
        from flash_attn import flash_attn_func
        ow = flash_attn_func(qw.transpose(1, 2), kw_.transpose(1, 2), vw.transpose(1, 2), causal=True, softmax_scale=scale)
    else:
        import torch.nn.functional as F
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            ow = F.scaled_dot_product_attention(qw, kw_, vw, is_causal=True, scale=scale)
    if bwd:
        torch.autograd.grad(ow, (qw, kw_, vw), torch.randn_like(ow))
    del qw, kw_, vw, ow
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # ---- inputs --------------------------------------------------------------------------------------------------
    dev = "cpu" if host else "cuda"
    layout = "token" if (host or arm == "FA-2") else "head"
    q, pinned = make_tensor(N, D, dtype, seed=seed + 1, device=dev, layout=layout, pin=True, requires_grad=bwd)
    k, _ = make_tensor(N, D, dtype, seed=seed + 2, device=dev, layout=layout, pin=True, requires_grad=bwd)
    v, _ = make_tensor(N, D, dtype, seed=seed + 3, device=dev, layout=layout, pin=True, requires_grad=bwd)
    dout = None
    if bwd:
        dout, _ = make_tensor(N, D, dtype, seed=seed + 4, device=dev, layout="token" if arm == "FA-2" else "head", pin=True)
    res.update(inputs_gib=(3 + (1 if bwd else 0)) * q.numel() * q.element_size() / 2**30, pinned=pinned if host else None,
               layout=layout)

    # ---- references (untimed): fp64 spot rows always; FA-2 output + gradients under ref="fa2" -------------------
    rows = spot_row_indices(N)
    out64, lse64 = ref64_forward(q, k, v, rows, scale)
    dq64 = ref64_dq(q, k, v, dout, out64, lse64, rows, scale) if bwd else None
    ref_out = ref_grads = None
    res["ref_fa2"] = "skipped"                                  # the status of the reference arm's run (key kept for old rows)
    ref_arm = resolve_ref_arm(row.get("ref"))
    res["ref_arm"] = ref_arm                                    # "FA-2" | "SDPA-flash" | None: what rel_err / fp64_*_fa2 compare to
    if ref_arm is not None and not (arm == "FA-2" and ref_arm == "FA-2") and not (arm == "SDPA" and ref_arm == "SDPA-flash"):
        try:
            ref_out, ref_grads = arm_reference(q, k, v, dout, scale, bwd, ref_arm)
            res["ref_fa2"] = "ok"
            res["fp64_err_fa2"] = rel_rows(ref_out, rows, out64)
            if bwd:
                res["fp64_err_dq_fa2"] = rel_rows(ref_grads[0], rows, dq64)
        except Exception as e:                                   # noqa: BLE001
            if not is_oom(e):
                raise
            res["ref_fa2"] = "oom"
            torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # ---- the arm -------------------------------------------------------------------------------------------------
    info_box = {}
    capture = {}
    if arm == "FA-2":
        from flash_attn import flash_attn_func
        qf, kf, vf = (t.transpose(1, 2) for t in (q, k, v))          # the token-major storage: no copy
        df = dout.transpose(1, 2) if bwd else None

        def call():
            return flash_attn_func(qf, kf, vf, causal=True, softmax_scale=scale).transpose(1, 2)

        def bwd_fn(out):
            return [g.transpose(1, 2) for g in torch.autograd.grad(out.transpose(1, 2), (qf, kf, vf), df)]
    elif arm == "SDPA":
        import torch.nn.functional as F
        from torch.nn.attention import sdpa_kernel, SDPBackend

        def call():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)

        def bwd_fn(out):
            return list(torch.autograd.grad(out, (q, k, v), dout))
    else:
        import stream_cqsa
        c, acc, kernel = int(row["c"]), row["acc"], KERNEL_OF[arm]
        iset = tuple(QUORUM_SETS[c])
        if bwd:
            _patch_classic_autograd(capture)
        # the depth: pinned, or PLAN.md's rule for E1 / E2 (the smallest depth >= 1 whose plan fits)
        if row["itr"] == "min":
            itr, pname, preason, fits = smallest_depth_that_fits(q, k, v, kernel, c, iset, acc)
            res.update(itr_planned=itr, plan_name=pname, plan_reason=(preason or "")[:600], plan_fits=fits)
        else:
            itr = int(row["itr"])
            res.update(itr_planned=itr)
        extra = {}
        if row.get("extra") and str(row["extra"]).startswith("W="):
            from stream_cqsa.native_wave import wave_token_floor
            mult = int(str(row["extra"])[2:].rstrip("x"))
            floor = wave_token_floor(N, itr, c, iset, H=H, D=D, itemsize=q.element_size())
            extra["max_wave_tokens"] = int(mult * floor)
            res.update(wave_floor_tokens=int(floor), max_wave_tokens_requested=extra["max_wave_tokens"])
        res["engine_expected"] = ("wave" if (arm.startswith("wave") and (not bwd or (arm == "wave_CUDA" and not host)))
                                  else "classic")
        res["kernel_requested"] = kernel

        def make_call(itr_):
            pins = dict(itr=itr_, c=c, interest_set=iset, accumulate_on_gpu=(acc == "gpu"), low_memory=(acc == "cpu"), **extra)
            if bwd:
                pins["bwd_itr"] = itr_
            else:
                pins["allow_escalation"] = False                 # a pinned depth that does not fit is an OOM row

            def call():
                out, inf = stream_cqsa.attention(q, k, v, is_causal=True, kernel=kernel, return_info=True, **pins)
                info_box.clear()
                info_box.update(inf)
                return out
            return call

        def bwd_fn(out):
            return list(torch.autograd.grad(out, (q, k, v), dout))

    hwm0 = vm_hwm_bytes()
    attempts = []
    out = grads = st = None
    if arm in SERIES:
        for itr_try in ([itr] if row["itr"] != "min" else [itr, itr + 1, itr + 2]):
            try:
                out, grads, st = run_timed(make_call(itr_try), int(row["warmup"]), int(row["reps"]), bwd_fn=bwd_fn if bwd else None)
                itr = itr_try
                break
            except Exception as e:                               # noqa: BLE001
                if not is_oom(e):
                    raise
                attempts.append(dict(itr=itr_try, status="oom", error=str(e)[:800]))
                out = grads = None
                gc.collect()
                torch.cuda.empty_cache()
        res["itr_attempts"] = attempts
        if st is None:
            raise torch.cuda.OutOfMemoryError(f"out of memory at itr {[a['itr'] for a in attempts]}: {attempts[-1]['error']}")
    else:
        out, grads, st = run_timed(call, int(row["warmup"]), int(row["reps"]), bwd_fn=bwd_fn if bwd else None)
    res.update(st)
    res["peak_host_gib"] = max(0, vm_hwm_bytes() - hwm0) / 2**30
    res["tokens_per_s"] = N / st["wall_s"]

    # ---- what ran --------------------------------------------------------------------------------------------------
    if arm in SERIES:
        info = sanitize(info_box)
        if bwd and capture:
            if "forward" in capture:
                f = capture["forward"]
                for k_ in ("n_subproblems", "n_parallel", "oom_retries", "depth_escalations", "itr_max_reached", "monolithic"):
                    if k_ in f:
                        info[k_] = f[k_]
            if "backward" in capture:
                info["backward"] = capture["backward"]
        res["info"] = info
        res["engine"] = info.get("engine")
        res["kernel"] = info.get("kernel")
        res["itr_ran"] = info.get("itr") if isinstance(info.get("itr"), int) else itr
        res["n_subproblems"] = info.get("n_subproblems")
        res["n_waves"] = info.get("n_waves") or (info.get("backward") or {}).get("n_waves")
        res["wave_sizes"] = info.get("wave_sizes")
        res["max_wave_tokens"] = info.get("max_wave_tokens")
        res["acc_ran"] = ("gpu" if (res["engine"] == "wave" and bwd) else row["acc"])   # the wave backward is device-resident only
        res["engine_as_expected"] = res["engine"] == res["engine_expected"]
        assert_decomposed = _decomp_helpers()
        try:
            chk = dict(info)
            if chk.get("n_subproblems") is None and isinstance(chk.get("itr"), int):
                pass                                             # attention()'s backward info: itr >= 1 is the evidence
            assert_decomposed(chk, itr=itr, c=c)
            res["decomposition_asserted"] = True
            # the pinned depth either engine runs is exactly c**itr subproblems at depth itr
            if res["n_subproblems"] is None:
                res["n_subproblems"] = c ** itr
                res["n_subproblems_inferred"] = True
        except AssertionError as e:
            res["decomposition_asserted"] = False
            res["status"] = "error"
            res["error"] = f"decomposition assertion failed: {e}"[:3000]
    else:
        res.update(engine="monolithic", kernel=arm.lower(), itr_ran=0, n_subproblems=1, n_waves=None)

    # ---- errors ------------------------------------------------------------------------------------------------------
    res["fp64_err"] = rel_rows(out, rows, out64)
    if res.get("fp64_err_fa2"):
        res["fp64_ratio_fa2"] = res["fp64_err"] / res["fp64_err_fa2"]
    if bwd and grads is not None:
        res["fp64_err_dq"] = rel_rows(grads[0], rows, dq64)
        if res.get("fp64_err_dq_fa2"):
            res["fp64_dq_ratio_fa2"] = res["fp64_err_dq"] / res["fp64_err_dq_fa2"]
    if ref_out is not None:
        res["rel_err"] = rel_full(out, ref_out)
        if bwd and ref_grads is not None:
            res["rel_err_grads_each"] = [rel_full(g, r) for g, r in zip(grads, ref_grads)]
            res["rel_err_grads"] = max(res["rel_err_grads_each"])
    res["out_device"] = str(out.device)
    res["nan_in_out"] = bool(torch.isnan(out[:, :, rows]).any().item())


# ---------------------------------------------------------------------------------------------------------------
# run: the rows of one experiment / part, each in a fresh subprocess
# ---------------------------------------------------------------------------------------------------------------
def worker_env():
    # attention(kernel="triton") sets CQSA_FORWARD / CQSA_BACKWARD=triton in its own process and never unsets them; a parent
    # that called it (a notebook) must not hand them down, or a CUDA series row would run the Triton backward under a CUDA label
    env = {k_: v_ for k_, v_ in os.environ.items() if k_ not in ("CQSA_FORWARD", "CQSA_BACKWARD")}
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def run_worker(row, log_dir=None):
    cmd = [sys.executable, "-P", os.path.abspath(__file__), "worker", json.dumps(row)]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=worker_env(), cwd=ROOT)
    except KeyboardInterrupt:
        return None
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    if line:
        r = json.loads(line[7:])
        r["row_wall_s"] = time.time() - t0
    else:
        killed = proc.returncode < 0 or proc.returncode == 137
        r = dict(row, status="error", rc=proc.returncode, row_wall_s=time.time() - t0,
                 error=(f"worker killed (rc={proc.returncode}): host memory / signal?" if killed else f"worker exited rc={proc.returncode}"),
                 stdout_tail=proc.stdout[-4000:], slurm_job=os.environ.get("SLURM_JOB_ID"))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        name = "_".join(str(row.get(f)) for f in KEY_FIELDS).replace("/", "_")
        with open(os.path.join(log_dir, f"{name}.log"), "w") as f:
            f.write(proc.stdout)
    return r


def describe(r):
    itr = r.get("itr")
    return (f"{r['exp']}:{r.get('part', '-'):8s} {r['arm']:14s} N={r['N']:8d} D={r['D']:3d} {r['dtype']} {r['direction']:3s} "
            f"c={str(r.get('c') or '-'):>3s} itr={str(itr if itr is not None else '-'):>3s} acc={str(r.get('acc') or '-'):3s} "
            f"qkv={r.get('qkv', '-'):6s}" + (f" {r['extra']}" if r.get("extra") else ""))


def one_line(r):
    if r.get("status") != "ok":
        return f"{r['status'].upper()}: {str(r.get('error', ''))[:160]}"
    s = (f"wall {r['wall_s']:9.3f} s (min {r['wall_min_s']:.3f})  peak {r['peak_dev_gib']:6.2f} GiB  host +{r.get('peak_host_gib', 0):.2f} GiB  "
         f"fp64 {r.get('fp64_err', float('nan')):.1e}")
    if r.get("rel_err") is not None:
        s += f"  rel {r['rel_err']:.1e}"
    if r.get("rel_err_grads") is not None:
        s += f" grads {r['rel_err_grads']:.1e}"
    if r.get("fp64_err_dq") is not None:
        s += f"  fp64 dq {r['fp64_err_dq']:.1e}"
    s += f"  engine={r.get('engine')}/{r.get('kernel')} itr={r.get('itr_ran')} n_sub={r.get('n_subproblems')}"
    if r.get("n_waves"):
        s += f" waves={r['n_waves']}"
    if r.get("itr_attempts"):
        s += f" attempts={[a['itr'] for a in r['itr_attempts']]}"
    return s + f"  ({r.get('row_wall_s', 0):.0f} s)"


def apply_filters(rows, a):
    def lst(s, conv=str):
        return [conv(x) for x in s.split(",") if x] if s else None
    f = dict(arm=lst(a.series), N=lst(a.N, int), c=lst(a.c, int), itr=lst(a.itr, lambda x: x if x == "min" else int(x)),
             acc=lst(a.acc), direction=lst(a.direction), dtype=lst(a.dtype), D=lst(a.D, int), extra=lst(a.extra))
    out = []
    for r in rows:
        ok = True
        for k_, vals in f.items():
            if vals is None:
                continue
            if r["kind"] == "baseline" and k_ in ("c", "itr", "acc", "extra"):
                continue                                         # baselines carry no decomposition parameters
            if r.get(k_) not in vals:
                ok = False
                break
        if ok:
            out.append(r)
    return out


def cmd_run(a):
    out_dir = a.out or (SMOKE_OUT if a.smoke else DEFAULT_OUT)
    os.makedirs(out_dir, exist_ok=True)
    exps = [e for e in a.exp.split(",") if e]
    rows = []
    for exp in exps:
        if exp not in PLAN:
            raise SystemExit(f"unknown experiment {exp!r}; the plan has {list(PLAN)}")
        rows += plan_rows(exp, a.part, smoke=a.smoke)
    rows = apply_filters(rows, a)
    for r in rows:
        if a.reps is not None:
            r["reps"] = a.reps
        if a.warmup is not None:
            r["warmup"] = a.warmup
        if a.ref:
            r["ref"] = a.ref
        if a.mem_cap_gib:
            r["mem_cap_gib"] = a.mem_cap_gib
        if a.set_N:
            r["N"] = a.set_N                                  # plumbing tests of a fixed-N experiment at another N
    if any(r["arm"] == "FA-2" for r in rows) and not flash_attn_importable():
        n_fa2 = sum(1 for r in rows if r["arm"] == "FA-2")
        print(f"== flash_attn is not importable: the {n_fa2} FA-2 arm rows are dropped (SDPA = torch's flash backend stays)", flush=True)
        rows = [r for r in rows if r["arm"] != "FA-2"]
    pols = sorted({r.get("ref") for r in rows if r.get("ref") in ("fa2", "sdpa")})
    if pols:
        print("== reference arm: " + ", ".join(f"{p} -> {resolve_ref_arm(p)}" for p in pols)
              + ("" if flash_attn_importable() else " (flash_attn not importable: SDPA-flash stands in and is labelled so)"), flush=True)
    history = read_rows(out_dir)
    done = latest_by_key(history)
    t_start = time.time()
    n_run = n_skip = n_defer = 0
    simulated = 0.0                                              # --dry-run: the estimates of the rows that would have run
    print(f"== bench {a.exp}{(':' + a.part) if a.part else ''}: {len(rows)} rows planned, {len(done)} keys in {out_dir}/results.jsonl, "
          f"budget {a.budget_s or 'none'} s, job {os.environ.get('SLURM_JOB_ID')}", flush=True)
    for r in rows:
        key = row_key(r)
        prev = done.get(key)
        if prev is not None and not a.redo:
            if prev.get("status") in ("ok", "oom") or (prev.get("status") == "error" and not a.retry_errors):
                n_skip += 1
                continue
        est = estimate_row_s(r, history)
        elapsed = (time.time() - t_start) + simulated
        if a.budget_s and elapsed + 1.25 * est > a.budget_s:
            n_defer += 1
            print(f"{describe(r)}: deferred (budget: {elapsed:.0f} s elapsed + ~{est:.0f} s estimated > {a.budget_s:.0f} s)", flush=True)
            continue
        if a.dry_run:
            simulated += est
            print(f"{describe(r)}: would run (~{est:.0f} s; ~{simulated:.0f} s cumulative)", flush=True)
            continue
        res = run_worker(r, log_dir=os.path.join(out_dir, "logs"))
        if res is None:                                          # interrupted: nothing recorded, the row runs next time
            print(f"{describe(r)}: INTERRUPTED (not recorded)", flush=True)
            break
        res.setdefault("slurm_job", os.environ.get("SLURM_JOB_ID"))
        res["recorded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        append_row(out_dir, res)
        history.append(res)
        done[key] = res
        n_run += 1
        print(f"{describe(r)}: {one_line(res)}", flush=True)
    print(f"== done: {n_run} rows run, {n_skip} already present, {n_defer} deferred by the budget, "
          f"{(time.time() - t_start):.0f} s", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# report: tables, fits, figures, STATUS.md
# ---------------------------------------------------------------------------------------------------------------
COLORS = {"classic_CUDA": "#2a78d6", "wave_CUDA": "#eb6834", "classic_Triton": "#1baf7a", "wave_Triton": "#eda100",
          "FA-2": "#52514e", "SDPA": "#9a9893"}
STYLES = {"FA-2": dict(linestyle="--"), "SDPA": dict(linestyle=":")}
MARKERS = {"classic_CUDA": "o", "wave_CUDA": "s", "classic_Triton": "^", "wave_Triton": "D", "FA-2": "x", "SDPA": "+"}


def fmt(x, nd=3, unit=""):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{nd}f}{unit}"


def fmt_e(x):
    return "-" if x is None else f"{x:.1e}"


def n_label(N):
    return f"{N / M:g}M" if N >= M else f"{N // K}K"


def baseline_lookup(rows_ok):
    """FA-2 / SDPA rows by (node, N, D, dtype, direction): the experiment's own baselines first, any experiment's next."""
    own, any_ = {}, {}
    for r in rows_ok:
        if r["arm"] in BASELINES:
            own[(r["exp"], r["arm"], r["node"], r["N"], r["D"], r["dtype"], r["direction"])] = r
            any_.setdefault((r["arm"], r["node"], r["N"], r["D"], r["dtype"], r["direction"]), r)

    def get(r, arm):
        return (own.get((r["exp"], arm, r["node"], r["N"], r["D"], r["dtype"], r["direction"]))
                or any_.get((arm, r["node"], r["N"], r["D"], r["dtype"], r["direction"])))
    return get


def engine_cell(r):
    if r["arm"] in BASELINES:
        return "monolithic"
    s = f"{r.get('engine')}/{r.get('kernel')} itr={r.get('itr_ran')} n_sub={r.get('n_subproblems')}"
    if r.get("n_waves"):
        s += f" waves={r['n_waves']}"
    if r.get("itr_attempts"):
        s += " (oom at itr " + ",".join(str(x["itr"]) for x in r["itr_attempts"]) + ")"
    if r.get("engine_as_expected") is False:
        s += " [*]"
    if r["arm"] == "classic_CUDA" and r.get("direction") == "bwd" and r.get("backward_kernel") not in (None, "cuda"):
        s += f" [bwd kernel {r['backward_kernel']}]"             # the module's default backward was not the CUDA one
    return s


def ref_arm_of(r):
    """The arm the row's rel_err / fp64_*_fa2 fields compare to: FA-2 (rows before the ref_arm field), SDPA-flash, or -."""
    if r.get("ref_fa2") != "ok":
        return "-"
    return r.get("ref_arm") or "FA-2"


def ref_arms_label(rows):
    arms = sorted({ref_arm_of(r) for r in rows} - {"-"})
    return " / ".join(arms) if arms else "ref"


def table_for(exp, rows, get_base):
    lines = ["| N | D | dtype | dir | c | itr | acc | qkv | W | arm | engine that ran | wall s | min s | peak GiB | workspace | host GiB | "
             "/FA-2 | /SDPA | ref | rel vs ref | grads | fp64 (x ref) | fp64 dq (x ref) | status |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        base = {}
        for arm in BASELINES:
            b = get_base(r, arm)
            base[arm] = (r["wall_s"] / b["wall_s"]) if (b and r.get("status") == "ok" and b.get("status") == "ok") else None
        st = r.get("status")
        what = engine_cell(r) if st == "ok" else f"{st}: {str(r.get('error', ''))[:120]}"
        fp64 = fmt_e(r.get("fp64_err")) + (f" ({r['fp64_ratio_fa2']:.2f})" if r.get("fp64_ratio_fa2") else "")
        fp64dq = fmt_e(r.get("fp64_err_dq")) + (f" ({r['fp64_dq_ratio_fa2']:.2f})" if r.get("fp64_dq_ratio_fa2") else "")
        lines.append(f"| {n_label(r['N'])} | {r['D']} | {r['dtype']} | {r['direction']} | {r.get('c') or '-'} | {r.get('itr') if r.get('itr') is not None else '-'} | "
                     f"{r.get('acc') or '-'} | {r.get('qkv') or '-'} | {r.get('extra') or '-'} | {r['arm']} | {what} | "
                     f"{fmt(r.get('wall_s'))} | {fmt(r.get('wall_min_s'))} | {fmt(r.get('peak_dev_gib'), 2)} | {fmt(r.get('workspace_gib'), 2)} | "
                     f"{fmt(r.get('peak_host_gib'), 2)} | {fmt(base['FA-2'], 3)} | {fmt(base['SDPA'], 3)} | {ref_arm_of(r)} | {fmt_e(r.get('rel_err'))} | "
                     f"{fmt_e(r.get('rel_err_grads'))} | {fp64} | {fp64dq} | {st} |")
    return lines


def summary_e0(rows):
    """E0's accuracy table: per series the worst errors over every (c, itr, acc, dtype, D) row."""
    ref = ref_arms_label(rows)
    lines = [f"Worst case per series over all E0 rows (ok rows only; the reference arm is {ref}):", "",
             f"| series | rows | max rel vs {ref} (output) | max rel vs {ref} (grads) | max fp64 err (output) | max fp64 err / {ref}'s | max fp64 err (dQ) | max fp64 dQ err / {ref}'s |",
             "|---|---|---|---|---|---|---|---|"]
    for arm in SERIES:
        rs = [r for r in rows if r["arm"] == arm and r.get("status") == "ok"]
        if not rs:
            continue

        def mx(key):
            vals = [r[key] for r in rs if r.get(key) is not None]
            return max(vals) if vals else None
        lines.append(f"| {arm} | {len(rs)} | {fmt_e(mx('rel_err'))} | {fmt_e(mx('rel_err_grads'))} | {fmt_e(mx('fp64_err'))} | "
                     f"{fmt(mx('fp64_ratio_fa2'), 2)} | {fmt_e(mx('fp64_err_dq'))} | {fmt(mx('fp64_dq_ratio_fa2'), 2)} |")
    return lines + [""]


def r2(y, yh):
    import numpy as np
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    ss = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((y - yh) ** 2) / ss if ss > 0 else float("nan")


def fits_for(exp, rows):
    """Per (series, direction): quadratic fit of time vs N and linear fit of peak memory vs N (>= 3 points), as
    benchmarks/profile_sweep.py's fit step."""
    import numpy as np
    fits = {}
    for arm in SERIES + BASELINES:
        for direction in ("fwd", "bwd"):
            rs = sorted([r for r in rows if r["arm"] == arm and r["direction"] == direction and r.get("status") == "ok"], key=lambda r: r["N"])
            if len(rs) < 3:
                continue
            x = np.array([r["N"] for r in rs], float) / 1e6
            t = np.array([r["wall_s"] for r in rs])
            m = np.array([r["peak_dev_gib"] for r in rs])
            pt = np.polyfit(x, t, 2)
            pm = np.polyfit(x, m, 1)
            fits[f"{arm}/{direction}"] = dict(
                N=[int(r["N"]) for r in rs], wall_s=t.tolist(), peak_gib=m.tolist(),
                time_fit="t[s] = %.4g*(N/1e6)^2 + %.4g*(N/1e6) + %.4g" % tuple(pt), time_coef=pt.tolist(), time_r2=float(r2(t, np.polyval(pt, x))),
                mem_fit="peak[GiB] = %.4g*(N/1e6) + %.4g" % tuple(pm), mem_coef=pm.tolist(), mem_r2=float(r2(m, np.polyval(pm, x))))
    return fits


def _style(arm):
    return dict(color=COLORS[arm], marker=MARKERS[arm], markersize=6, linewidth=2, **STYLES.get(arm, {}))


def _legend(ax, **kw):
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(frameon=False, **kw)


def _axes_cosmetics(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def figure_scaling(exp, rows, fits, path, base_rows=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for j, direction in enumerate(("fwd", "bwd")):
        for arm in SERIES + BASELINES:
            src = rows if arm in SERIES or any(r["arm"] == arm for r in rows) else (base_rows or [])
            rs = sorted([r for r in src if r["arm"] == arm and r["direction"] == direction and r.get("status") == "ok"], key=lambda r: r["N"])
            if not rs:
                continue
            x = np.array([r["N"] for r in rs], float) / 1e6
            lab = arm + (" (resident, from E1)" if src is not rows else "")
            if arm in SERIES and any(r.get("itr") == "min" for r in rs):
                depths = sorted({r.get("itr_ran") for r in rs} - {None})
                if depths:
                    lab += f" (itr {'/'.join(map(str, depths))})"
            axes[0, j].plot(x, [r["wall_s"] for r in rs], label=lab, **_style(arm))
            axes[1, j].plot(x, [r["peak_dev_gib"] for r in rs], label=lab, **_style(arm))
            f = fits.get(f"{arm}/{direction}")
            if f and src is rows and arm in SERIES:
                xx = np.linspace(x.min(), x.max(), 100)
                axes[0, j].plot(xx, np.polyval(f["time_coef"], xx), color=COLORS[arm], linewidth=0.8, alpha=0.6)
        ooms = [r for r in rows if r["direction"] == direction and r.get("status") == "oom"]
        for r in ooms:
            axes[0, j].annotate("OOM", (r["N"] / 1e6, axes[0, j].get_ylim()[0] if axes[0, j].lines else 1), color=COLORS[r["arm"]], fontsize=7)
        Ns = sorted({r["N"] for r in rows if r["direction"] == direction})
        for ax in axes[:, j]:
            ax.set_xscale("log", base=2)
            if Ns:
                ax.set_xticks([n / 1e6 for n in Ns])
                ax.set_xticklabels([n_label(n) for n in Ns])
                ax.minorticks_off()
        axes[0, j].set_yscale("log")
        _axes_cosmetics(axes[0, j], "N (tokens)", "wall time (s)", f"{exp} {direction}: wall vs N (quadratic fits)")
        _axes_cosmetics(axes[1, j], "N (tokens)", "peak device memory (GiB)", f"{exp} {direction}: peak vs N")
        _legend(axes[0, j], fontsize=7)
        _legend(axes[1, j], fontsize=7)
    fig.suptitle(f"{exp}: {PLAN[exp.split()[0]]['question']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def figure_vs_c(exp, rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for j, direction in enumerate(("fwd", "bwd")):
        for arm in SERIES:
            rs = sorted([r for r in rows if r["arm"] == arm and r["direction"] == direction and r.get("status") == "ok"], key=lambda r: r["c"])
            if not rs:
                continue
            axes[0, j].plot([r["c"] for r in rs], [r["wall_s"] for r in rs], label=arm, **_style(arm))
            axes[1, j].plot([r["c"] for r in rs], [r["peak_dev_gib"] for r in rs], label=arm, **_style(arm))
        for ax in axes[:, j]:
            ax.set_xscale("log")
            ax.set_xticks(C_LIST)
            ax.set_xticklabels(C_LIST)
            _legend(ax, fontsize=7)
        _axes_cosmetics(axes[0, j], "c (quorum size)", "wall time (s)", f"{exp} {direction}: wall vs c (N=1M, itr=1)")
        _axes_cosmetics(axes[1, j], "c (quorum size)", "peak device memory (GiB)", f"{exp} {direction}: peak vs c")
    fig.suptitle(f"{exp}: {PLAN[exp]['question']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def figure_bars(exp, rows, path, group_of, group_label, group_order):
    """Grouped bars (one group per configuration, one bar per arm): wall time and peak memory, fwd and bwd."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    arms = [a for a in BASELINES + SERIES if any(r["arm"] == a for r in rows)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for j, direction in enumerate(("fwd", "bwd")):
        groups = [g for g in group_order if any(group_of(r) == g and r["direction"] == direction for r in rows)]
        if not groups:
            for ax in axes[:, j]:
                ax.text(0.5, 0.5, "no rows yet", ha="center", va="center", transform=ax.transAxes, color="#52514e")
                _axes_cosmetics(ax, "", "", f"{exp} {direction}")
            continue
        xs = np.arange(len(groups))
        w = 0.8 / max(1, len(arms))
        for i, arm in enumerate(arms):
            vals_t, vals_m = [], []
            for g in groups:
                r = next((r for r in rows if r["arm"] == arm and r["direction"] == direction and group_of(r) == g), None)
                ok = r is not None and r.get("status") == "ok"
                vals_t.append(r["wall_s"] if ok else 0)
                vals_m.append(r["peak_dev_gib"] if ok else 0)
            axes[0, j].bar(xs + (i - len(arms) / 2 + 0.5) * w, vals_t, w * 0.92, color=COLORS[arm], label=arm)
            axes[1, j].bar(xs + (i - len(arms) / 2 + 0.5) * w, vals_m, w * 0.92, color=COLORS[arm], label=arm)
        for ax in axes[:, j]:
            ax.set_xticks(xs)
            ax.set_xticklabels([group_label(g) for g in groups], fontsize=8)
            _legend(ax, fontsize=7)
        _axes_cosmetics(axes[0, j], "", "wall time (s)", f"{exp} {direction}: wall time")
        _axes_cosmetics(axes[1, j], "", "peak device memory (GiB)", f"{exp} {direction}: peak device memory")
    fig.suptitle(f"{exp}: {PLAN[exp]['question']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def figure_errors(exp, rows, path):
    """E0: fp64 error of the output (and dq) per series vs c, next to FA-2's own on the same rows; one line per
    (series, itr, acc): itr=2 hollow markers, acc=cpu (host-resident) dash-dotted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    panels = ((axes[0], "fwd", "fp64_err"), (axes[1], "fwd", "rel_err"), (axes[2], "bwd", "fp64_err_dq"))
    for arm in SERIES:
        for itr in (1, 2):
            for acc in ("gpu", "cpu"):
                for ax, direction, key in panels:
                    rs = sorted([r for r in rows if r["arm"] == arm and r.get("itr") == itr and r.get("acc") == acc and r.get("status") == "ok"
                                 and r["direction"] == direction and r["dtype"] == "fp16" and r["D"] == 64 and r.get(key) is not None],
                                key=lambda r: r["c"])
                    if not rs:
                        continue
                    st = _style(arm)
                    if itr == 2:
                        st.update(markerfacecolor="white")
                    if acc == "cpu":
                        st.update(linestyle="-.", alpha=0.75)
                    ax.plot([r["c"] for r in rs], [r[key] for r in rs], label=f"{arm} itr={itr} acc={acc}", **st)
    fa = [r for r in rows if r.get("fp64_err_fa2") and r["direction"] == "fwd" and r["dtype"] == "fp16" and r["D"] == 64]
    if fa:
        axes[0].axhline(statistics.median(r["fp64_err_fa2"] for r in fa), color=COLORS["FA-2"], linestyle="--", linewidth=1.5, label=f"{ref_arms_label(fa)} (own fp64 error)")
    fb = [r for r in rows if r.get("fp64_err_dq_fa2") and r["direction"] == "bwd" and r["dtype"] == "fp16" and r["D"] == 64]
    if fb:
        axes[2].axhline(statistics.median(r["fp64_err_dq_fa2"] for r in fb), color=COLORS["FA-2"], linestyle="--", linewidth=1.5, label=f"{ref_arms_label(fb)} (own fp64 error)")
    for ax, t, yl in ((axes[0], "output vs fp64 (128 rows)", "norm-relative error"), (axes[1], f"output vs {ref_arms_label(rows)} (full tensor)", "norm-relative error"),
                      (axes[2], "dQ vs fp64 (128 rows)", "norm-relative error")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([3, 7, 31, 133])
        ax.set_xticklabels([3, 7, 31, 133])
        ax.minorticks_off()
        _axes_cosmetics(ax, "c", yl, f"{exp}: {t}")
        _legend(ax, fontsize=6)
    fig.suptitle(f"{exp}: {PLAN[exp]['question']} (fp16, D=64 rows; the bf16 / D=128 rows are in the table)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def figure_wave_budget(exp, rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for direction, alpha in (("fwd", 1.0), ("bwd", 0.55)):
        for arm in ("wave_CUDA", "wave_Triton"):
            rs = sorted([r for r in rows if r["arm"] == arm and r["direction"] == direction and r.get("status") == "ok"
                         and r.get("max_wave_tokens")], key=lambda r: r["max_wave_tokens"])
            if not rs:
                continue
            xs = [r["max_wave_tokens"] / 1e6 for r in rs]
            axes[0].plot(xs, [r["wall_s"] for r in rs], label=f"{arm} {direction}", alpha=alpha, **_style(arm))
            axes[1].plot(xs, [r["peak_dev_gib"] for r in rs], label=f"{arm} {direction}", alpha=alpha, **_style(arm))
    ticks = {}
    for r in rows:
        if r.get("status") == "ok" and r.get("max_wave_tokens") and r.get("extra"):
            ticks.setdefault(r["max_wave_tokens"], (r["extra"], r.get("n_waves")))
    for ax in axes:
        ax.set_xscale("log", base=2)
        if ticks:
            ws = sorted(ticks)
            ax.set_xticks([w / 1e6 for w in ws])
            ax.set_xticklabels([f"{ticks[w][0][2:]}\n{w / 1e6:.2f}M\n{ticks[w][1]} waves" for w in ws], fontsize=8)
            ax.minorticks_off()
        _legend(ax, fontsize=7)
    _axes_cosmetics(axes[0], "wave budget W (x the largest subproblem; packed tokens; waves)", "wall time (s)", f"{exp}: wall vs W (N=1M, c=73)")
    _axes_cosmetics(axes[1], "wave budget W (x the largest subproblem; packed tokens; waves)", "peak device memory (GiB)", f"{exp}: peak vs W")
    fig.suptitle(f"{exp}: {PLAN[exp]['question']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def status_doc(out_dir, latest, smoke=False):
    """STATUS.md: per experiment and part the rows done / oom / error / missing against the plan, with job ids."""
    lines = [f"# results/v3 status ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"`{os.path.join(out_dir, 'results.jsonl')}`: {len(latest)} distinct configurations recorded (last row per key).", "",
             "| experiment | part | planned | ok | oom | error | missing | jobs |", "|---|---|---|---|---|---|---|---|"]
    missing_total = 0
    failed = []
    missing_rows = []
    for exp in EXPERIMENTS:
        for part in PLAN[exp]["parts"]:
            rows = plan_rows(exp, part, smoke=smoke)
            cnt = dict(ok=0, oom=0, error=0, missing=0)
            jobs = set()
            for r in rows:
                got = latest.get(row_key(r))
                if got is None:
                    cnt["missing"] += 1
                    missing_rows.append(r)
                    continue
                cnt[got.get("status", "error") if got.get("status") in cnt else "error"] += 1
                if got.get("slurm_job"):
                    jobs.add(str(got["slurm_job"]))
                if got.get("status") in ("oom", "error"):
                    failed.append((exp, part, got))
            missing_total += cnt["missing"]
            lines.append(f"| {exp} | {part} | {len(rows)} | {cnt['ok']} | {cnt['oom']} | {cnt['error']} | {cnt['missing']} | {', '.join(sorted(jobs)) or '-'} |")
    lines += ["", f"**Missing rows: {missing_total}**", ""]
    if KNOWN_EXCLUSIONS:
        lines += ["## Excluded from the plan (findings, not measurements)", ""]
        for x in KNOWN_EXCLUSIONS:
            where = ", ".join(f"{k}={'/'.join(v) if isinstance(v, tuple) else v}" for k, v in x.items() if k != "reason")
            lines.append(f"- {where}: {x['reason']}")
        lines.append("")
    if failed:
        lines += ["## Failed rows (oom / error)", "", "| experiment | part | row | status | job | message |", "|---|---|---|---|---|---|"]
        for exp, part, g in failed:
            lines.append(f"| {exp} | {part} | {describe(g).strip()} | {g.get('status')} | {g.get('slurm_job') or '-'} | "
                         f"{str(g.get('error', ''))[:160].replace('|', '/')} |")
        lines.append("")
    if missing_rows:
        lines += ["## Missing rows", ""]
        by = {}
        for r in missing_rows:
            by.setdefault((r["exp"], r["part"]), []).append(r)
        for (exp, part), rs in by.items():
            lines.append(f"- {exp}:{part}: {len(rs)} rows, e.g. " + "; ".join(describe(r).strip() for r in rs[:3]))
        lines.append("")
    return "\n".join(lines), missing_total


def cmd_report(a):
    out_dir = a.out or (SMOKE_OUT if a.smoke else DEFAULT_OUT)
    all_rows = read_rows(out_dir)
    latest = latest_by_key(all_rows)
    rows = sorted(latest.values(), key=row_order)
    ok = [r for r in rows if r.get("status") == "ok"]
    get_base = baseline_lookup(ok)
    gpus = sorted({r.get("gpu", "?") for r in rows})
    jobs = sorted({str(r.get("slurm_job")) for r in rows if r.get("slurm_job")})
    mods = sorted({f"{r.get('cuda_module')} / {r.get('native_module')}" for r in rows if r.get("cuda_module")})
    lines = ["# Stream-CQSA v3 experiments (experiments/v3/PLAN.md): report", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `experiments/v3/bench.py report` from `{os.path.relpath(os.path.join(out_dir, 'results.jsonl'), ROOT)}` "
             f"({len(all_rows)} rows, {len(latest)} configurations; the last row per key). GPUs: {', '.join(gpus)}. Jobs: {', '.join(jobs) or '-'}. "
             f"Modules: {'; '.join(mods) or '-'}.", "",
             "B=1 H=8 causal; wall = median of the timed runs (fwd: one attention() call; bwd: forward + torch.autograd.grad), min alongside; "
             "peak = torch.cuda.max_memory_allocated over the call, workspace = peak minus what was resident before it; host = VmHWM delta "
             "over the timed runs; /FA-2 and /SDPA = wall ratios against the resident monolithic baselines of the same (node type, N, D, dtype, "
             "direction) (the experiment's own first, E1's / E6's otherwise); ref = the reference arm of the row (FA-2 = flash_attn, or SDPA-flash = "
             "torch SDPA on its flash backend where flash_attn is not installed); rel = norm-relative error against that arm on the same inputs "
             "(grads: the worst of dq/dk/dv); fp64 = norm-relative error on 128 sampled rows against an fp64 reference, with the ratio to the "
             "reference arm's own fp64 error on the same rows in parentheses (PLAN.md asks <= 1.2). Every Stream-CQSA row pins itr and c and asserts the "
             "decomposition it reports; `[*]` marks an engine other than the series' nominal one (documented routing: the wave backward runs on "
             "the CUDA kernel and on device-resident inputs only, so wave_Triton fwd+bwd and host-resident wave backwards run on the classic "
             "engine; the wave backward keeps a device accumulator, so wave_CUDA acc=cpu fwd+bwd rows ran with acc=gpu in the backward).", ""]
    if a.smoke or any(r.get("smoke") for r in rows):
        lines += ["**SMOKE rows (plumbing checks on a shared GPU, not measurements).**", ""]
    fits_all = {}
    figs = []
    for exp in EXPERIMENTS:
        ex = [r for r in rows if r["exp"] == exp]
        lines += [f"## {exp}: {PLAN[exp]['question']}", ""]
        if not ex:
            lines += ["_no rows yet_", ""]
            continue
        lines += table_for(exp, ex, get_base) + [""]
        png = os.path.join(out_dir, f"{exp}.png")
        try:
            if exp in ("E1", "E2"):
                # one node type per figure and fit set (never compare across node types): the SXM rows, then E2's PCIe rows
                sxm = [r for r in ex if r["node"] == "sxm"]
                pcie = [r for r in ex if r["node"] == "pcie"]
                fits = fits_for(exp, sxm)
                fits_all[exp] = fits
                if fits:
                    lines += ["Fits (profile_sweep's: time quadratic in N, peak linear in N; >= 3 points; SXM rows):", ""]
                    for key, f in fits.items():
                        lines.append(f"- {key}: {f['time_fit']} (R2 {f['time_r2']:.4f}); {f['mem_fit']} (R2 {f['mem_r2']:.4f})")
                    lines.append("")
                base_rows = [r for r in ok if r["exp"] == "E1" and r["arm"] in BASELINES and r["node"] == "sxm"] if exp == "E2" else None
                if sxm:
                    figure_scaling(exp, sxm, fits, png, base_rows=base_rows)
                if pcie:
                    png2 = os.path.join(out_dir, f"{exp}_pcie.png")
                    figure_scaling(exp + " (PCIe)", pcie, {}, png2)
                    figs.append(png2)
                    lines += [f"![{exp} PCIe]({os.path.basename(png2)})", ""]
            elif exp == "E0":
                lines += summary_e0(ex)
                figure_errors(exp, ex, png)
                worst = [r for r in ex if r.get("status") == "ok" and (r.get("fp64_ratio_fa2") or 0) > 1.2 or (r.get("fp64_dq_ratio_fa2") or 0) > 1.2]
                lines.append(f"Rows above 1.2x FA-2's own fp64 error: {len(worst)}" + (" -- " + "; ".join(describe(r).strip() for r in worst[:8]) if worst else "") + "\n")
            elif exp == "E3":
                figure_vs_c(exp, ex, png)
            elif exp == "E4":
                figure_bars(exp, ex, png, lambda r: (r.get("itr"), r.get("acc")), lambda g: f"itr={g[0]} acc={g[1]}",
                            [(1, "gpu"), (1, "cpu"), (2, "gpu"), (2, "cpu")])
            elif exp == "E5":
                figure_wave_budget(exp, ex, png)
            elif exp == "E6":
                figure_bars(exp, ex, png, lambda r: (r["dtype"], r["D"]), lambda g: f"{g[0]} D={g[1]}",
                            [("fp16", 64), ("bf16", 64), ("fp16", 128), ("bf16", 128)])
            figs.append(png)
            lines += [f"![{exp}]({os.path.basename(png)})", ""]
        except Exception as e:                                   # noqa: BLE001
            lines += [f"_figure failed: {type(e).__name__}: {e}_", ""]
            traceback.print_exc()
    status_md, missing = status_doc(out_dir, latest, smoke=a.smoke or any(r.get("smoke") for r in rows))
    open(os.path.join(out_dir, "report.md"), "w").write("\n".join(lines))
    json.dump(fits_all, open(os.path.join(out_dir, "fits.json"), "w"), indent=1)
    open(os.path.join(out_dir, "STATUS.md"), "w").write(status_md)
    print(f"report: {os.path.join(out_dir, 'report.md')} ({len(lines)} lines), figures {[os.path.basename(f) for f in figs]}, "
          f"fits.json, STATUS.md (missing {missing})")
    return missing


def cmd_status(a):
    out_dir = a.out or (SMOKE_OUT if a.smoke else DEFAULT_OUT)
    latest = latest_by_key(read_rows(out_dir))
    smoke = a.smoke or any(r.get("smoke") for r in latest.values())
    status_md, missing = status_doc(out_dir, latest, smoke=smoke)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "STATUS.md"), "w").write(status_md)
    print(status_md)
    print(f"MISSING {missing}")
    return missing


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["run", "worker", "report", "status", "plan"])
    ap.add_argument("arg", nargs="?", help="worker: the row as JSON")
    ap.add_argument("--exp", default="E1", help="experiment id(s), comma-separated (run / plan)")
    ap.add_argument("--part", default=None, help="one part of the experiment (fwd / bwd / all / pcie_fwd / pcie_bwd)")
    ap.add_argument("--out", default=None, help=f"results directory (default {DEFAULT_OUT}; --smoke: {SMOKE_OUT})")
    ap.add_argument("--budget-s", type=float, default=0.0, help="wall-clock budget of this run; rows that would exceed it are deferred")
    ap.add_argument("--only-missing", action="store_true", help="(the default behaviour) skip rows already in results.jsonl")
    ap.add_argument("--retry-errors", action="store_true", help="re-run rows whose last status is error (oom rows are final)")
    ap.add_argument("--redo", action="store_true", help="re-run every selected row")
    ap.add_argument("--dry-run", action="store_true", help="print the rows that would run, with their estimates")
    ap.add_argument("--smoke", action="store_true", help="plumbing on a shared GPU: N=16384 / 32768, one run, scratch output")
    ap.add_argument("--series", default="", help="comma list of arms to run (classic_CUDA, ..., FA-2, SDPA)")
    ap.add_argument("--N", default="", help="comma list of N")
    ap.add_argument("--c", default="")
    ap.add_argument("--itr", default="")
    ap.add_argument("--acc", default="")
    ap.add_argument("--direction", default="")
    ap.add_argument("--dtype", default="")
    ap.add_argument("--D", default="")
    ap.add_argument("--extra", default="")
    ap.add_argument("--reps", type=int, default=None, help="override the plan's timed runs per row")
    ap.add_argument("--warmup", type=int, default=None, help="override the plan's warm-up runs per row")
    ap.add_argument("--ref", default=None, choices=[None, "fa2", "sdpa", "spot"],
                    help="override the plan's reference policy: fa2 (FA-2, or SDPA-flash when flash_attn is missing), sdpa (SDPA-flash), spot")
    ap.add_argument("--mem-cap-gib", type=float, default=0.0, help="cap the allocator (plumbing tests of the OOM paths; recorded in the row)")
    ap.add_argument("--set-N", type=int, default=0, help="run the selected rows at this N instead (plumbing tests on a shared GPU)")
    a = ap.parse_args()
    if a.mode == "worker":
        print("RESULT " + json.dumps(worker(json.loads(a.arg)), default=str), flush=True)
    elif a.mode == "run":
        cmd_run(a)
    elif a.mode == "report":
        cmd_report(a)
    elif a.mode == "status":
        cmd_status(a)
    elif a.mode == "plan":
        rows = apply_filters(sum((plan_rows(e, a.part, smoke=a.smoke) for e in a.exp.split(",")), []), a)
        tot = 0.0
        for r in rows:
            est = estimate_row_s(r)
            tot += est
            print(f"{describe(r)}  ~{est:.0f} s")
        print(f"{len(rows)} rows, ~{tot / 60:.0f} GPU-min estimated")


if __name__ == "__main__":
    main()
